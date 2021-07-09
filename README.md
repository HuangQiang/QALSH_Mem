# QALSH_Mem: Memory Version of QALSH

Welcome to the **QALSH_Mem** GitHub!

**QALSH_Mem** is a package for the problem of Nearest Neighbor Search (NNS). Given a set of data points and a query, the problem of NNS aims to find the nearest data point to the query. It has wide applications in many data mining and machine learning tasks.

This package provides the internal memory implementations of two LSH schemes QALSH and QALSH<sup>+</sup> for *c*-Approximate Nearest Neighbor Search (c-ANNS) under *l<sub>p</sub>* norm, where *0 < p ⩽ 2*. The external version of QALSH and QALSH<sup>+</sup> can be found [here](https://github.com/HuangQiang/QALSH).

If you want to get more details of QALSH and QALSH<sup>+</sup>, please refer to our works [Query-Aware Locality-Sensitive Hashing for Approximate Nearest Neighbor Search](https://dl.acm.org/doi/abs/10.14778/2850469.2850470) and [Query-Aware Locality-Sensitive Hashing Scheme for *l<sub>p</sub>* Norm](https://link.springer.com/article/10.1007/s00778-017-0472-7), which have been published in PVLDB 2015 and VLDBJ 2017, respectively.

## Datasets and Queries

We study the performance of QALSH and QALSH<sup>+</sup> over six real-life [datasets](https://drive.google.com/drive/folders/1tKMl0_iLSEeuT1ZJ7s4x1BbLHyX0D5OJ), i.e., Mnist, P53, Trevi, Gist, Sift, and Sift10M. For each dataset, we provide 100 queries (randomly select from its test set or extract from the dataset itself) for evaluations. The statistics of datasets and queries are summarized as follows.

| Datasets | #Data Points | #Queries | Dimensionality | Range of Coordinates | Data Type |
| -------- | ------------ | -------- | -------------- | ----------- | --------- |
| Mnist    | 60,000       | 100      | 50             | [0, 255]    | uint8     |
| P53      | 31,059       | 100      | 5,408          | [0, 10,000] | uint16    |
| Trevi    | 100,800      | 100      | 4,096          | [0, 255]    | uint8     |
| Gist     | 1,000,000    | 100      | 960            | [0, 15000]  | uint16    |
| Sift     | 1,000,000    | 100      | 128            | [0, 255]    | uint8     |
| Sift10M  | 10,000,000   | 100      | 128            | [0, 255]    | uint8     |

Note that all the datasets and queries are in a binary format, which can be considered as an array of `n·d` coordinates, where each coordinate is specified by the data type, e.g., `uint8` or `uint16`. We currently support four data types: `uint8`, `uint16`, `int32`, and `float32`. One can determine the data type of the dataset based on the range of its coordinates. If you want to support more data types, you can update the interface in the `main.cc` and re-compile the package.

## Compilation

The package requires `g++` with `c++11` support. To download and compile the c++ source codes, please run the commands as follows:

```bash
git clone git@github.com/HuangQiang/QALSH_Mem.git
cd QALSH_Mem/methods/
make -j
```

## Usages

Suppose you have cloned the project and you are in the folder `QALSH_Mem/`. We provide bash scripts to run experiments for the six real-life datasets.

### Step 1: Get the Datasets and Queries

Please download the [datasets](https://drive.google.com/drive/folders/1tKMl0_iLSEeuT1ZJ7s4x1BbLHyX0D5OJ) and copy them to the directory `data/`.

For example, when you get `Sift.ds` and `Sift.q`, please move them to the paths `data/Sift/Sift.ds` and `data/Sift/Sift.q`, respectively. We also provide Mnist at this package, you can follow the same pattern to move the datasets to the right place.

### Step 2: Run Experiments

When you run the package, please ensure the paths for the dataset, query set, and truth set are correct. The package will automatically create folder for the output path, so please keep the output path as short as possible. All of the experiments can be run with the following commands:

```bash
cd methods/
bash run_all.sh
```

A gentle reminder is that when running QALSH and QALSH<sup>+</sup>, since they need the ground truth results for evaluation, please run `-alg 0` to get the ground truth results first.

### Step 3. Parameter Settings

Finally, if you would like to use this package for *c*-ANNS over other datasets, you may want to get more information about the parameters and know how to set them effectively.
Based on our experience when we conducted the experiments, we now share some tricks on setting up the parameters, i.e., `lf`, `L`, `M`, `p`, `z`, and `c`. The illustration of the parameters are as follows.

```bash
  -alg    integer    options of algorithms (0 - 3)
  -n      integer    number of data points (cardinality of dataset)
  -qn     integer    number of queries
  -d      integer    dimensionality of dataset and query set
  -lf     integer    leaf size of kd_tree
  -L      integer    number of projections for drusilla_select
  -M      integer    number of candidates  for drusilla_select
  -p      real       l_{p} norm, where 0 < p ⩽ 2
  -z      real       symmetric factor of p-stable distribution (-1 ⩽ z ⩽ 1)
  -c      real       approximation ratio for c-ANNS (c > 1)
  -dt     string     data type (i.e., uint8, uint16, int32, float32)
  -pf     string     the prefix of dataset, query set, and truth set
  -of     string     output folder to store output results
```

#### The settings of `lf`, `L`, and `M`

`lf` is the maximum leaf size of kd-tree. `L` and `M` are two parameters used for Drusilla_Select, where `L` is the number of random projections; `M` is the number of representative data points we select on each random projection.

Let `K` be the number of blocks, and let n<sub>0</sub> be the actual leaf size after kd-tree partitioning. Once `lf` is determined, `K` and **n<sub>0</sub>** can be computed as follows:

- **K = 2<sup>h</sup>**, where `h` is the height of the kd-tree, i.e., **h = ceil(log_2 (n / lf))**;
- **n<sub>0</sub> = floor(n / K)** or **n<sub>0</sub> = ceil(n / K)** (Note: if `n` is not divisible, these two cases can happen.)

When we set up these three parameters `lf`, `L` and `M`, it might be better to satisfy the following three conditions:

- **lf < n**: It is a natural condition that the maximum leaf size should be smaller than the cardinality of dataset;
- **L · M < n<sub>0</sub>**: It is a natural condition to restrict its size **(L · M)** less than **n<sub>0</sub>** when we run Drusilla_Select to select the representative data points on each block;
- **K · L · M ≈ n<sub>0</sub>**: This condition is the main trade-off between efficiency and accuracy to set up these three parameters.
  - On the one hand, if the total number of representative data points **(K · L · M)** is large, we can accurately identify the close blocks to the query, but it may introduce much time for the first-level close block search.
  - On the other hand, if **(K · L · M)** is small, the time for the first-level close block search can be reduced. However, since these blocks may not be really close to the query, the accuracy for the second-level c-ANNS may also be reduced.
  - In our experiments, we find that **selecting the representative data points with cardinality approximate to n<sub>0</sub>** can achieve a good trade-off between accuracy and efficiency.

#### The settings of `p`, `z`, and `c`

`p` and `z` determine the distance metric and the corresponding p-stable distribution. There are four cases as follows.

- *l<sub>2</sub>* distance: set up **p = 2.0** and **z = 0.0**, and apply standard Gaussian distribution.
- *l<sub>1</sub>* distance: set up **p = 1.0** and **z = 0.0** and apply standard Cauchy distribution.
- *l<sub>0.5</sub>* distance: set up **p = 0.5** and **z = 1.0** and apply standard Levy distribution.
- General *l<sub>p</sub>* distance: set up **0 < p ⩽ 2** and **-1 ⩽ z ⩽ 1**.

```c``` is the approximation ratio for *c*-ANNS. We set **c = 2.0** by default. If the dataset is easy, it is also satisfied to set **c = 3.0**.

## Reference

Please use the following BibTex to cite this work if you use **QALSH_Mem** for publications.

```tex
@article{huang2017query,
    title={Query-aware locality-sensitive hashing scheme for $$ l\_p $$ norm}
    author={Huang, Qiang and Feng, Jianlin and Fang, Qiong and Ng, Wilfred and Wang Wei},
    booktitle={The VLDB Journal},
    volumn={26},
    number={5},
    pages={683--708},
    year={2017}
}

@article{huang2015query,
    title={Query-aware locality-sensitive hashing for approximate nearest neighbor search}
    author={Huang, Qiang and Feng, Jianlin and Zhang, Yikai and Fang, Qiong and Ng, Wilfred},
    booktitle={Proceedings of the VLDB Endowment},
    volumn={9},
    number={1},
    pages={1--12},
    year={2015}
}
```

It is welcome to contact me (<huangq@comp.nus.edu.sg>) if you meet any issue. Thank you.
