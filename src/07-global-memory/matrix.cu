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
           这意味着：同一 warp 的线程正好可以访问一整行的 32 个元素，是否“连续访问”会非常清楚；
        2，方便观察 coalescing；
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

__global__ void copy(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
    const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
    const int index = ny * N + nx; //矩阵元素线性下标公式A[ny * N + nx] 表示第 ny 行第 nx 列
    if (nx < N && ny < N)
    {
        B[index] = A[index];
    }
}

__global__ void transpose1(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[nx * N + ny] = A[ny * N + nx]; //矩阵元素线性下标公式A[ny * N + nx] 表示第 ny 行第 nx 列
    }
}

__global__ void transpose2(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = A[nx * N + ny]; //矩阵元素线性下标公式 B[ny * N + nx] 表示第 ny 行第 nx 列
    }
}

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

