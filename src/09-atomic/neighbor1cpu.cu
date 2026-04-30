#include "error.cuh"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// 这个文件的核心功能是：从 xy.txt 读取一组二维原子坐标，在 CPU 上用暴力两两比较的方式找出距离小于 cutoff = 1.9 的原子对，
// 构建每个原子的邻居列表，重复计时 20 次，最后把邻居表写入 neighbor.txt

// 这个程序是“CPU 版本”的邻居列表构建示例。
//
// 它要解决的问题是：
// 给定 xy.txt 中的一批二维原子坐标，找出每个原子附近有哪些原子。
// 如果两个原子之间的距离小于 cutoff，就认为它们互为邻居。
//
// 程序最终会生成 neighbor.txt：
// 每一行对应一个原子，第一列是邻居数量，后面是这些邻居的原子编号。

// 编译时如果定义了 USE_DP，就使用 double；否则默认使用 float。
// 这样同一份代码可以在单精度和双精度之间切换：
//   nvcc neighbor1cpu.cu          -> real 是 float
//   nvcc -DUSE_DP neighbor1cpu.cu -> real 是 double
#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

// 原子总数。这里先声明，等 read_xy 读完坐标后再赋值。
int N;//原子总数，运行时从 xy.txt 得到。

// 重复计时次数。邻居搜索会连续运行 20 次，每次都打印耗时。
// 多跑几次可以减少偶然波动带来的影响，便于和 GPU 版本做性能对比。
const int NUM_REPEATS = 20;//NUM_REPEATS = 20：为了测性能，重复执行 20 次。

// 每个原子最多保存 MN 个邻居。
// NL 数组会按照 N * MN 的大小分配空间，所以 MN 不能小于真实最大邻居数。
const int MN = 10;//MN = 10：每个原子最多保存 10 个邻居。

// 截断半径，单位为 Angstrom。
// 两个原子的距离小于 cutoff 时，程序认为它们是邻居。
const real cutoff = 1.9;//cutoff = 1.9：邻居判定距离阈值。

// 实际比较时使用距离平方：
//   dx * dx + dy * dy < cutoff * cutoff
// 这样可以避免计算平方根 sqrt，速度更快，判断结果也等价。
const real cutoff_square = cutoff * cutoff;//cutoff_square：用距离平方比较，避免调用 sqrt。

// 从 xy.txt 读取二维坐标，分别存入 x 和 y 两个数组。
void read_xy(std::vector<real>& x, std::vector<real>& y);

// 重复调用 find_neighbor，并用 CUDA event 统计每次运行时间。
void timing(int *NN, int *NL, std::vector<real> x, std::vector<real> y);

// 把邻居数量 NN 和邻居列表 NL 写入 neighbor.txt。
void print_neighbor(const int *NN, const int *NL);

int main(void)
{
    // x 和 y 分别保存所有原子的二维坐标。
    // 第 n 个原子的坐标是：
    //   (x[n], y[n])
    // 也就是说，x 和 y 的下标都是原子编号。
    std::vector<real> x, y;
    read_xy(x, y);//read_xy(x, y) 读取坐标文件。

    // 每读入一行坐标，就对应一个原子。
    // read_xy 会同时向 x 和 y 里 push_back，因此正常情况下 x.size() == y.size()。
    N = x.size();//N = x.size() 得到原子数量。

    // NN[n] 表示第 n 个原子实际有多少个邻居。
    // 例如 NN[7] = 3，表示 7 号原子有 3 个邻居。
    int *NN = (int*) malloc(N * sizeof(int));//数组 NN[n] 第 n 个原子的邻居数量。

    // NL 是一张一维数组形式的邻居表。
    // 第 n 个原子的第 k 个邻居存放在 NL[n * MN + k]。
    //
    // 可以把它想象成一个 N 行、MN 列的二维表：
    //   第 0 行：0 号原子的邻居
    //   第 1 行：1 号原子的邻居
    //   第 n 行：n 号原子的邻居
    //
    // 只是 C/C++ 里这里用一维数组手动展开：
    //   NL[n * MN + 0], NL[n * MN + 1], ..., NL[n * MN + MN - 1]
    // 就是第 n 个原子的所有邻居槽位。
    int *NL = (int*) malloc(N * MN * sizeof(int));//NL[n * MN + k]：第 n 个原子的第 k 个邻居编号。
    
    // 多次运行邻居搜索并打印每次耗时。
    timing(NN, NL, x, y);

    // 把最终得到的邻居数量和邻居编号写到 neighbor.txt。
    print_neighbor(NN, NL);//print_neighbor(...) 输出邻居表。

    free(NN);
    free(NL);
    return 0;
}

/*
它打开当前运行目录下的 xy.txt，每个非空行读取两个数，分别放入 v_x 和 v_y。注意这里路径是相对运行目录的，不是相对源码文件目录的。
所以通常需要在 src/09-atomic 目录下运行，或者确保运行目录里有 xy.txt。
当前仓库里的 xy.txt 有 22467 行，其中最后 3 行为空行；程序会跳过空行，所以实际读入 22464 个原子坐标。
*/
void read_xy(std::vector<real>& v_x, std::vector<real>& v_y)//read_xy(x, y) 读取坐标文件。

{
    // xy.txt 中每个非空行应包含两个数：x 坐标和 y 坐标。
    // 例如某一行是：
    //   28.06444083882057 241.17654747387496
    // 那么第一个数会进入 v_x，第二个数会进入 v_y。
    //
    // 注意：这里打开的是相对路径 "xy.txt"。
    // 程序运行时，需要保证当前工作目录下存在 xy.txt。
    std::ifstream infile("xy.txt");
    std::string line, word;
    if(!infile)
    {
        std::cout << "Cannot open xy.txt" << std::endl;
        exit(1);
    }
    while (std::getline(infile, line))
    {
        // 把一整行字符串包装成输入流，后面就可以像读文件一样逐个读取单词。
        std::istringstream words(line);

        // 跳过空行，避免把空白行当成错误数据。
        if(line.length() == 0)
        {
            continue;
        }

        // 每一行只读取前两个数据：第一个作为 x，第二个作为 y。
        // i == 0 时读 x 坐标，i == 1 时读 y 坐标。
        for (int i = 0; i < 2; i++)
        {
            if(words >> word)
            {
                if(i == 0)
                {
                    // std::stod 把字符串转换成 double。
                    // 如果 real 是 float，push_back 时会再转换成 float。
                    v_x.push_back(std::stod(word));
                }
                if(i==1)
                {
                    v_y.push_back(std::stod(word));
                }
            }
            else
            {
                // 如果一行里不足两个数，就说明 xy.txt 格式不符合要求。
                std::cout << "Error for reading xy.txt" << std::endl;
                exit(1);
            }
        }
    }
    infile.close();
}

void find_neighbor(int *NN, int *NL, const real* x, const real* y)
{
    // 每次搜索前先把所有原子的邻居数量清零。
    // timing 会重复调用 find_neighbor 多次，如果不清零，邻居数量会累加到上一次结果上。
    for (int n = 0; n < N; n++)
    {
        NN[n] = 0;
    }

    // 逐对检查原子之间的距离。
    // n2 从 n1 + 1 开始，是为了每个原子对只检查一次：
    // 检查了 (n1, n2) 后，就不再重复检查 (n2, n1)。
    //
    // 举例：
    //   n1 = 0 时，检查 (0, 1)、(0, 2)、(0, 3) ...
    //   n1 = 1 时，从 (1, 2) 开始，不再检查 (1, 0)
    //
    // 这样总比较次数是 N * (N - 1) / 2，而不是 N * N。
    for (int n1 = 0; n1 < N; ++n1)
    {
        // 先把 n1 原子的坐标取出来。
        // 内层循环会反复用到它，放在局部变量里可以少访问几次数组。
        real x1 = x[n1];
        real y1 = y[n1];
        for (int n2 = n1 + 1; n2 < N; ++n2)
        {
            // 计算两个原子在 x、y 方向上的距离差。
            // x12、y12 分别表示从 n1 指向 n2 的坐标差。
            real x12 = x[n2] - x1;
            real y12 = y[n2] - y1;

            // 使用距离的平方进行比较，避免调用 sqrt，提高速度。
            // 二维欧氏距离本来是：
            //   distance = sqrt(x12 * x12 + y12 * y12)
            // 判断 distance < cutoff 等价于判断 distance_square < cutoff_square。
            real distance_square = x12 * x12 + y12 * y12;

            // 如果距离小于截断半径，就把这两个原子互相加入对方的邻居表。
            if (distance_square < cutoff_square)
            {
                // 把 n2 作为 n1 的一个邻居保存起来，然后 NN[n1] 自增。
                //
                // 假设当前 NN[n1] 是 2，说明 n1 已经有 2 个邻居，
                // 那么新邻居应该写到第 3 个槽位，也就是 k = 2 的位置：
                //   NL[n1 * MN + 2] = n2
                //
                // 表达式 NN[n1]++ 的含义是：
                //   先使用 NN[n1] 当前的值作为下标，再把 NN[n1] 加 1。
                NL[n1 * MN + NN[n1]++] = n2;

                // 邻居关系是对称的：n1 是 n2 的邻居，n2 也是 n1 的邻居。
                // 所以找到一对近邻时，需要写入两次：
                //   n2 写进 n1 的邻居表
                //   n1 写进 n2 的邻居表
                NL[n2 * MN + NN[n2]++] = n1;
            }
        }
    }
}

void timing(int *NN, int *NL, std::vector<real> x, std::vector<real> y)//timing(...) 重复运行 CPU 近邻搜索 20 次并打印耗时。
{
    // 重复运行 NUM_REPEATS 次，只是为了测量耗时，不会改变算法结果。
    // 每次 find_neighbor 开始时都会把 NN 清零，所以每一轮都是重新计算邻居表。
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;

        // 创建两个 CUDA event：
        //   start 记录开始时刻
        //   stop  记录结束时刻
        // CHECK 宏来自 error.cuh，用于检查 CUDA API 调用是否成功。
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        // 这里用 CUDA event 计时，方便和 GPU 版本的程序保持同一种计时方式。
        // 虽然本文件的邻居搜索在 CPU 上执行，但计时工具仍然复用了 CUDA event。
        CHECK(cudaEventRecord(start));

        // 等待 start 事件真正被记录完成，确保后面的计时区间从这里开始。
        while(cudaEventQuery(start)!=cudaSuccess){}

        // 真正的邻居搜索发生在 CPU 上。
        find_neighbor(NN, NL, x.data(), y.data());

        CHECK(cudaEventRecord(stop));

        // 等待 stop 事件完成，然后才能计算 start 到 stop 之间的时间。
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        std::cout << "Time = " << elapsed_time << " ms." << std::endl;

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
}

void print_neighbor(const int *NN, const int *NL)//print_neighbor(...) 输出邻居表。
{
    // 输出文件中每一行对应一个原子：
    // 第一列是邻居数量，后面 MN 列是邻居编号，不足的位置用 NaN 填充。
    //
    // 例如某一行：
    //   2 5 18 NaN NaN NaN NaN NaN NaN NaN NaN
    // 表示该原子有 2 个邻居，邻居编号分别是 5 和 18。
    std::ofstream outfile("neighbor.txt");
    if (!outfile)
    {
        std::cout << "Cannot open neighbor.txt" << std::endl;
    }
    for (int n = 0; n < N; ++n)
    {
        // 如果某个原子的邻居数超过 MN，说明预留的邻居表空间不够。
        //
        // 重要提醒：
        // 这个检查发生在输出阶段。严格来说，如果 find_neighbor 中已经写入超过
        // MN 个邻居，越界写入可能已经发生。这里主要用于发现 MN 设得太小的问题。
        if (NN[n] > MN)
        {
            std::cout << "Error: MN is too small." << std::endl;
            exit(1);
        }

        // 先输出第 n 个原子的邻居数量。
        outfile << NN[n];

        // 固定输出 MN 个邻居槽位。
        // 这样 neighbor.txt 每一行的列数一致，后续脚本读取和画图会更方便。
        for (int k = 0; k < MN; ++k)
        {
            if(k < NN[n])
            {
                // 输出第 n 个原子的第 k 个邻居编号。
                outfile << " " << NL[n * MN + k];
            }
            else
            {
                // 没有邻居的位置用 NaN 占位，保证每一行列数一致。
                outfile << " NaN";
            }
        }
        outfile << std::endl;
    }
    outfile.close();
}

/*
这个文件本身没有使用 atomicAdd，因为 CPU 是串行执行 find_neighbor，不会有多个线程同时修改同一个 NN[n]。
它主要是作为基准版本，对应 GPU 版本 neighbor2gpu.cu (line 1)。GPU 版本里，如果多个线程同时给同一个原子增加邻居数，
就需要用 atomicAdd 避免竞态条件。因此这个 CPU 文件的作用是：提供正确性参考和性能对照。
*/