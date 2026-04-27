#include "error.cuh"
#include <stdio.h>

/*
定义了一个别名 real：
* 如果编译时定义了 USE_DP，那么 real = double
* 否则 real = float
也就是说，这个程序既可以测 单精度 float，也可以测 双精度 double
*/
#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

/*定义两个常量，在核函数中可以使用在函数外部由#define或const定义的常量，包括整型常量和浮点型常量*/
const int NUM_REPEATS = 10;//每个任务会跑 11 次，其中第 1 次通常当作“热身”，后 10 次参与均值和误差计算
const int TILE_DIM = 32;//block 大小是 32 x 32，即一个 block 有 1024 个线程

void timing(const real *d_A, real *d_B, const int N, const int task);
__global__ void copy(const real *A, real *B, const int N);
__global__ void transpose1(const real *A, real *B, const int N);
__global__ void transpose2(const real *A, real *B, const int N);
__global__ void transpose3(const real *A, real *B, const int N);
void print_matrix(const int N, const real *A);

int main(int argc, char **argv)
{
    /*程序要求运行时传一个参数N，表示矩阵大小，矩阵是N*N，元素总数是N^2*/
    if (argc != 2)
    {
        printf("usage: %s N\n", argv[0]);
        exit(1);
    }
    const int N = atoi(argv[1]);

    const int N2 = N * N //N2是元素总个数
    const int M = sizeof(real) * N2; //M是总字节数，这里用int存字节数，若N很大，可能溢出，更稳妥的写法应当是：size_t M = sizeof(real) * (size_t)N2;
    real *h_A = (real *) malloc(M); //主机端输入矩阵
    real *h_B = (real *) malloc(M); //主机端输出矩阵
    for (int n = 0; n < N2; ++n) //初始化输入矩阵，把A填成0，1，2，3，...，如果把它按 N x N 看，就是按行递增。这样初始化的好处是：1，很容易看出是否转置了；2，很容易检查索引是否写对；
    {
        h_A[n] = n;
    }
    real *d_A, *d_B; //申请设备内存并拷贝
    CHECK(cudaMalloc(&d_A, M)); //GPU设备上的输入矩阵
    CHECK(cudaMalloc(&d_B, M)); //GPU设备上的输出矩阵
    CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice)); //将主机上的矩阵数据拷贝到设备输入矩阵d_A

    /*依次测四个任务*/
    printf("\ncopy:\n");
    timing(d_A, d_B, N, 0);
    printf("\ntranspose with coalesced read:\n");
    timing(d_A, d_B, N, 1);
    printf("\ntranspose with coalesced write:\n");
    timing(d_A, d_B, N, 2);
    printf("\ntranspose with coalesced write and __ldg read:\n");
    timing(d_A, d_B, N, 3);

    CHECK(cudaMemcpy(h_B, d_B, M, cudaMemcpyDeviceToHost)); //拷回结果并打印
    if (N <= 10) //最后把d_B拷回主机到h_B，如果矩阵很小就打印，这里只打印最后一次任务留下的 B。因为前面 4 次 timing() 都在写同一个 d_B，最后打印看到的是 task=3 的结果
    {
        printf("A =\n");
        print_matrix(N, h_A);
        printf("\nB =\n");
        print_matrix(N, h_B);
    }

    free(h_A); //释放内存
    free(h_B);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    return 0;
}

void timing(const real *d_A, real *d_B, const int N, const int task)
{
    /*网格和线程块配置*/
    const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM; //grid 大小采用向上取整：(N + 31) / 32，这样即使 N 不是 32 的倍数，也能覆盖整个矩阵。多出来的线程通过 if (nx < N && ny < N) 做边界保护
    const int grid_size_y = grid_size_x;
    /*block_size = (32, 32)的优点：
        1，32和warp大小一致；
           CUDA 一个 warp 通常是 32 个线程。如果线程布局是 threadIdx.x 为快变维度，那么一行正好 32 个线程，常常对应一个 warp；
           这意味着：同一 warp 的线程正好可以访问一整行的 32 个元素，
                    是否“连续访问”会非常清楚；
        2，方便观察 coalescing；
           例如：
            * A[ny * N + nx]：同一 warp 的 nx 连续，所以访问连续
            * A[nx * N + ny]：同一 warp 的 nx 连续，但地址按列跳跃，所以不连续

            这个实验设计非常典型，就是为了直观展示 coalesced vs non-coalesced。
    */
    const dim3 block_size(TILE_DIM, TILE_DIM); //block_size = (32, 32)，所以每个 block 覆盖矩阵中的一个 32x32 区域
    const dim3 grid_size(grid_size_x, grid_size_y); // grid 大小

    float t_sum = 0;
    float t2_sum = 0; //时间统计变量，用于计算平均值/标准差
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) //重复执行，总共跑11次：第 0 次：热身，不计入统计；第 1~10 次：计入统计；因为第一次运行通常会有额外开销，例如：kernel 首次启动开销，上下文初始化残余影响，cache 状态未稳定
    {
        //用 CUDA Event 计时
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start)); //创建两个事件
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start)); //记录开始事件
        cudaEventQuery(start);

        switch (task) //根据task启动不同kernel，kernel 启动本身是异步的。所以如果没有后面的 cudaEventSynchronize(stop)，CPU 可能会在 GPU 还没算完时就继续往下走，导致计时不准。
        {
            case 0:
                copy<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 1:
                transpose1<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 2:
                transpose2<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 3:
                transpose3<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            default:
                printf("Error: wrong task\n");
                exit(1);
                break;
        }

        //得到kernel执行时间（毫秒），不包括主机初始化矩阵时间，不包括cudaMemcpy H2D/D2H 时间，不包括malloc/free 时间，这里测的是纯kernel时间
        CHECK(cudaEventRecord(stop)); //记录结束事件
        CHECK(cudaEventSynchronize(stop)); //等GPU完成
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop)); //计算两事件之间的时间
        printf("Time = %g ms.\n", elapsed_time);

        //统计均值和误差
        if (repeat > 0) //跳过第一次热身
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    //标准差
    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}

/*
对于一个 warp 来说，通常 threadIdx.x 连续变化，所以：
* 读取 A[index]：地址连续
* 写入 B[index]：地址连续
这就是典型的 coalesced read + coalesced write
也就是 合并读 + 合并写
这是最理想的全局内存访问模式之一，因此通常最快。
*/
__global__ void copy(const real *A, real *B, const int N)/*线程映射关系：每个线程处理矩阵中一个元素 (ny, nx)。*/
{
    const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
    const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
    const int index = ny * N + nx; //矩阵元素线性下标公式A[ny * N + nx] 表示第 ny 行第 nx 列
    if (nx < N && ny < N)
    {
        B[index] = A[index];/*也就是直接拷贝：B(y, x) = A(y, x)*/
    }
}

/*
A[ny * N + nx] 对于同一行中 nx 连续的线程，访问地址连续。
所以 读是合并的（coalesced read）
B[nx * N + ny]
这里对于 warp 中相邻线程，nx 在变，而 ny 通常固定，因此地址差是 N 个元素，而不是 1 个元素。
也就是线程写的是跨行跳跃位置。
所以 写是不合并的（non-coalesced write）
GPU 全局内存最喜欢的是“相邻线程访问相邻地址”。
这里写入变成了“相邻线程访问跨度很大的地址”，会导致：
* 内存事务数量增多
* 带宽利用率下降
* 性能明显差于 copy

*/
__global__ void transpose1(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[nx * N + ny] = A[ny * N + nx]; //矩阵元素线性下标公式A[ny * N + nx] 表示第 ny 行第 nx 列 / B(x, y) = A(y, x) 这就是标准矩阵转置。
    }
}

/*
B(y, x) = A(x, y)
这个结果和真正的转置矩阵内容是一样的。
因为标准转置也是：
B = A^T, B(y,x)=A(x,y)
所以 transpose2 也是在做转置，只是从“输出坐标”的角度写法不同。
* transpose1：从 A 的正常布局顺序读，往 B 的转置位置写
* transpose2：从 A 的转置位置读，往 B 的正常布局顺序写
相邻线程写相邻地址
所以 写是合并的（coalesced write）
相邻线程读的地址相隔 N
所以 读是不合并的（non-coalesced read）
和 transpose1 谁更快
这个取决于 GPU 架构、cache、内存系统等。
很多情况下：

* 不合并读和不合并写都会慢
* 但有时“不合并读 + 合并写”会比“合并读 + 不合并写”稍好
* 也有可能差不多
这个程序就是在实验这种差异。
*/
__global__ void transpose2(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = A[nx * N + ny]; //矩阵元素线性下标公式 B[ny * N + nx] 表示第 ny 行第 nx 列
    }
}

/*
它和 transpose2 完全一样，只是把读取方式从普通加载改成了：
__ldg(&A[...])

__ldg() 是 CUDA 提供的只读加载指令接口。
它提示编译器/硬件：

* 这个地址的数据只读
* 可以走只读数据缓存路径

通常适用于：
* 输入数组不会被修改
* 存在重复读取
* 普通全局内存访问效率不高
在这里为什么可能有帮助
因为 transpose2/3 的问题在于 读不连续。
如果这些不连续读能更好地利用只读缓存，那么 transpose3 可能比 transpose2 快一点。
但要注意：

* 在较新的 GPU 架构上，编译器有时会自动做优化
* __ldg() 的收益可能不明显
* 甚至和普通读几乎一样

所以它的实验意义大于“必然更快”。
*/
__global__ void transpose3(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = __ldg(&A[nx * N + ny]); //矩阵元素线性下标公式 B[ny * N + nx] 表示第 ny 行第 nx 列
    }
}

void print_matrix(const int N, const real *A)
{
    for (int ny = 0; ny < N; ny++)
    {
        for (int nx = 0; nx < N; nx++)
        {
            printf("%g\t", A[ny * N + nx]); //矩阵元素线性下标公式 A[ny * N + nx] 表示第 ny 行第 nx 列
        }
        printf("\n");
    }
}

