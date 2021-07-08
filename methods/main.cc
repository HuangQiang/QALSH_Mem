#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

#include "def.h"
#include "util.h"
#include "ann.h"

using namespace nns;

// -----------------------------------------------------------------------------
void usage()                        // display usage of this package
{
    printf("\n"
        "--------------------------------------------------------------------\n"
        " Usage of the Package for c-k-Approximate Nearest Neighbor Search   \n"
        "--------------------------------------------------------------------\n"
        "    -alg  (integer)   options of algorithms\n"
        "    -n    (integer)   number of data points\n"
        "    -qn   (integer)   number of query points\n"
        "    -d    (integer)   data dimension\n"
        "    -p    (real)      l_p distance <==> p-stable distr. (0, 2]\n"
        "    -z    (real)      symmetric factor of p-stable distr. [-1, 1]\n"
        "    -c    (real)      approximation ratio (c > 1)\n"
        "    -lf   (integer)   leaf size of kd-tree (QALSH+)\n"
        "    -L    (integer)   #projections for drusilla select (QALSH+)\n"
        "    -M    (integer)   #candidates  for drusilla select (QALSH+)\n"
        "    -dt   (string)    data type\n"
        "    -pf   (string)    prefix folder\n"
        "    -of   (string)    output folder\n"
        "\n"
        "--------------------------------------------------------------------\n"
        " The Options of Algorithms (-alg) are:                              \n"
        "--------------------------------------------------------------------\n"
        "    0 - Ground-Truth\n"
        "        Params: -alg 0 -n -qn -d -p -dt -pf\n"
        "\n"
        "    1 - QALSH+\n"
        "        Params: -alg 1 -n -qn -d -p -z -c -lf -L -M -dt -pf -of\n"
        "\n"
        "    2 - QALSH\n"
        "        Params: -alg 2 -n -qn -d -p -z -c -dt -pf -of\n"
        "\n"
        "    3 - Linear-Scan\n"
        "        Params: -alg 3 -n -qn -d -p -dt -pf -of\n"
        "\n"
        "--------------------------------------------------------------------\n"
        " Author: Qiang Huang (huangq@comp.nus.edu.sg, huangq2011@gmail.com) \n"
        "--------------------------------------------------------------------\n"
        "\n\n\n");
}

// -----------------------------------------------------------------------------
template<class DType>
void interface(                     // interface for calling function
    int   alg,                          // which algorithm
    int   n,                            // number of data points
    int   qn,                           // number of query points
    int   d,                            // dimensionality
    int   leaf,                         // leaf size of kd-tree
    int   L,                            // number of projection (drusilla)
    int   M,                            // number of candidates (drusilla)
    float p,                            // p-stable distr. (0,2]
    float zeta,                         // symmetric factor of p-distr. [-1,1]
    float c,                            // approximation ratio
    const char *prefix,                 // prefix of data, query, and truth
    const char *folder)                 // output folder
{
    // read data set, query set, and ground truth file
    gettimeofday(&g_start_time, NULL);
    DType  *data  = new DType[(uint64_t) n*d];
    DType  *query = new DType[(uint64_t) qn*d];
    Result *truth = NULL;

    if (read_data<DType>(n,  d, 0, p, prefix, data)) exit(1);
    if (read_data<DType>(qn, d, 1, p, prefix, query)) exit(1);
    if (alg > 0) {
        truth = new Result[(uint64_t) qn*MAXK];
        if (read_data<Result>(qn, MAXK, 2, p, prefix, truth)) exit(1);
    }
    gettimeofday(&g_end_time, NULL);
    float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Load data and query: %f Seconds\n\n", running_time);
    
    // methods
    switch (alg) {
    case 0:
        ground_truth<DType>(n, qn, d, p, prefix, (const DType*) data, 
            (const DType*) query);
        break;
    case 1:
        qalsh_plus<DType>(n, qn, d, leaf, L, M, p, zeta, c, folder, 
            (const DType*) data, (const DType*) query, (const Result*) truth);
        break;
    case 2:
        qalsh<DType>(n, qn, d, p, zeta, c, folder, (const DType*) data, 
            (const DType*) query, (const Result*) truth);
        break;
    case 3:
        linear_scan<DType>(n, qn, d, p, folder, (const DType*) data, 
            (const DType*) query, (const Result*) truth);
        break;
    default:
        printf("Parameters error!\n");
        usage();
    }
    // release space
    delete[] data;
    delete[] query;
    if (alg > 0) delete[] truth;
}

// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
    srand(6);                       // use a fixed seed instead of time(NULL)

    int   cnt  = 1;                 // parameter counter
    int   alg  = -1;                // which algorithm
    int   n    = -1;                // number of data points
    int   qn   = -1;                // number of query points
    int   d    = -1;                // dimensionality
    float p    = -1.0f;             // p-stable distr. (0,2]
    float zeta = -2.0f;             // symmetric factor of p-distr. [-1,1]
    float c    = -1.0f;             // approximation ratio
    int   leaf = -1;                // leaf size of kd-tree (QALSH+)
    int   L    = -1;                // #projections for drusilla-select (QALSH+)
    int   M    = -1;                // #candidates  for drusilla-select (QALSH+)
    char  dtype[20];                // data type
    char  prefix[200];              // prefix of data, query, and truth set
    char  folder[200];              // output folder
    
    while (cnt < nargs) {
        if (strcmp(args[cnt], "-alg") == 0) {
            alg = atoi(args[++cnt]); assert(alg >= 0);
            printf("alg    = %d\n", alg);
        }
        else if (strcmp(args[cnt], "-n") == 0) {
            n = atoi(args[++cnt]); assert(n > 0);
            printf("n      = %d\n", n);
        }
        else if (strcmp(args[cnt], "-qn") == 0) {
            qn = atoi(args[++cnt]); assert(qn > 0);
            printf("qn     = %d\n", qn); 
        }
        else if (strcmp(args[cnt], "-d") == 0) {
            d = atoi(args[++cnt]); assert(d > 0);
            printf("d      = %d\n", d); 
        }
        else if (strcmp(args[cnt], "-lf") == 0) {
            leaf = atoi(args[++cnt]); assert(leaf > 0);
            printf("leaf   = %d\n", leaf);
        }
        else if (strcmp(args[cnt], "-L") == 0) {
            L = atoi(args[++cnt]); assert(L > 0);
            printf("L      = %d\n", L);
        }
        else if (strcmp(args[cnt], "-M") == 0) {
            M = atoi(args[++cnt]); assert(M > 0);
            printf("M      = %d\n", M);
        }
        else if (strcmp(args[cnt], "-p") == 0) {
            p = (float) atof(args[++cnt]); assert(p > 0 && p <= 2);
            printf("p      = %.1f\n", p);
        }
        else if (strcmp(args[cnt], "-z") == 0) {
            zeta = (float) atof(args[++cnt]); assert(zeta >= -1 && zeta <= 1);
            printf("zeta   = %.1f\n", zeta);
        }
        else if (strcmp(args[cnt], "-c") == 0) {
            c = (float) atof(args[++cnt]); assert(c > 1);
            printf("c      = %.1f\n", c);
        }
        else if (strcmp(args[cnt], "-dt") == 0) {
            strncpy(dtype, args[++cnt], sizeof(dtype));
            printf("dtype  = %s\n", dtype);
        }
        else if (strcmp(args[cnt], "-pf") == 0) {
            strncpy(prefix, args[++cnt], sizeof(prefix));
            printf("prefix = %s\n", prefix);
        }
        else if (strcmp(args[cnt], "-of") == 0) {
            strncpy(folder, args[++cnt], sizeof(folder));
            int len = (int) strlen(folder);
            if (folder[len-1] != '/') { folder[len]='/'; folder[len+1]='\0'; }
            printf("folder = %s\n", folder);
            create_dir(folder);
        }
        else {
            printf("Parameters error!\n"); usage(); exit(1);
        }
        ++cnt;
    }
    printf("\n");

    if (strcmp(dtype, "uint8") == 0) {
        interface<uint8_t>(alg, n, qn, d, leaf, L, M, p, zeta, c, prefix, folder);
    }
    else if (strcmp(dtype, "uint16") == 0) {
        interface<uint16_t>(alg, n, qn, d, leaf, L, M, p, zeta, c, prefix, folder);
    }
    else if (strcmp(dtype, "int32") == 0) {
        interface<int>(alg, n, qn, d, leaf, L, M, p, zeta, c, prefix, folder);
    }
    else if (strcmp(dtype, "float32") == 0) {
        interface<float>(alg, n, qn, d, leaf, L, M, p, zeta, c, prefix, folder);
    }
    else {
        printf("Parameters error!\n"); usage();
    }
    return 0;
}
