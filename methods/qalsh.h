#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "def.h"
#include "random.h"
#include "pri_queue.h"
#include "util.h"

namespace nns {

// -----------------------------------------------------------------------------
//  Query-Aware Locality-Sensitive Hashing (QALSH) is used to solve the problem 
//  of c-Approximate Nearest Neighbor Search (c-ANNS).
//  This is an internal memory implementation of the following work:
//
//  Qiang Huang, Jianlin Feng, Yikai Zhang, Qiong Fang, and Wilfred Ng. 
//  Query-aware locality-sensitive hashing for approximate nearest neighbor 
//  search, Proceedings of the VLDB Endowment (PVLDB), 9(1), pages 1–12, 2015.
// -----------------------------------------------------------------------------
template<class DType>
class QALSH {
public:
    int   n_pts_;                   // number of data points
    int   dim_;                     // data dimension
    float p_;                       // l_p distance
    float zeta_;                    // symmetric factor of p-stable distr.
    float c_;                       // approximation ratio
    float w_;                       // bucket width
    int   m_;                       // number of hashtables
    int   l_;                       // collision threshold
    const DType *data_;             // data points
    const int *index_;              // data index
    
    float  *a_;                     // query-aware lsh functions
    Result *tables_;                // hash tables

    // -------------------------------------------------------------------------
    QALSH(                          // constructor
        int   n,                        // number of data points
        int   d,                        // data dimension
        float p,                        // l_p distance
        float zeta,                     // symmetric factor of p-stable distr.
        float c,                        // approximation ratio
        const DType *data,              // data points
        const int *index = NULL);       // data index

    // -------------------------------------------------------------------------
    ~QALSH();                       // destructor

    // -------------------------------------------------------------------------
    void display();                 // display parameters

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {   // get estimated memory usage
        uint64_t ret = 0ULL;
        ret += sizeof(*this);
        ret += sizeof(float)*m_*dim_;    // a_
        ret += sizeof(Result)*m_*n_pts_; // tables_
        return ret;
    }

    // -------------------------------------------------------------------------
    int knn(                        // k-NN search
        int   top_k,                    // top-k value
        const DType *query,             // input query
        MinK_List *list);               // k-NN results (return)

    // -------------------------------------------------------------------------
    int knn2(                       // k-NN search (assit func for QALSH+)
        int   top_k,                    // top-k value
        const DType *query,             // input query
        MinK_List *list);               // k-NN results (return)

protected:
    // -------------------------------------------------------------------------
    void init();                    // initialze basic parameters

    // -------------------------------------------------------------------------
    inline float calc_l0_prob(float x) { return new_levy_prob(x); }

    // -------------------------------------------------------------------------
    inline float calc_l1_prob(float x) { return new_cauchy_prob(x); }

    // -------------------------------------------------------------------------
    inline float calc_l2_prob(float x) { return new_gaussian_prob(x); }

    // -------------------------------------------------------------------------
    inline float calc_hash_value(int tid, const DType *data) { 
        return calc_inner_product<DType>(dim_, &a_[tid*dim_], data);
    }
};

// -----------------------------------------------------------------------------
template<class DType>
QALSH<DType>::QALSH(                // constructor
    int   n,                            // number of data points
    int   d,                            // data dimension
    float p,                            // l_p distance
    float zeta,                         // symmetric factor of p-stable distr.
    float c,                            // approximation ratio
    const DType *data,                  // data points
    const int *index)                   // data index
    : n_pts_(n), dim_(d), p_(p), zeta_(zeta), c_(c), data_(data), index_(index)
{
    // inti basic parameters
    init();

    // bulkloading
    tables_ = new Result[(uint64_t) m_*n_pts_];
    for (int i = 0; i < m_; ++i) {
        Result *table = &tables_[(uint64_t)i*n_pts_];
        for (int j = 0; j < n_pts_; ++j) {
            int id = index_ != NULL ? index_[j] : j;
            table[j].id_  = j;
            table[j].key_ = calc_hash_value(i, &data_[(uint64_t)id*dim_]);
        }
        qsort(table, n_pts_, sizeof(Result), ResultComp);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void QALSH<DType>::init()           // initialize basic parameters
{
    // -------------------------------------------------------------------------
    //  init <w_> <m_> and <l_> (auto tuning-w)
    //  
    //  w0 ----- best w for L_{0.5} norm to minimize m (auto tuning-w)
    //  w1 ----- best w for L_{1.0} norm to minimize m (auto tuning-w)
    //  w2 ----- best w for L_{2.0} norm to minimize m (auto tuning-w)
    //  other w: use linear combination for interpolation
    // -------------------------------------------------------------------------
    float delta = 1.0f / E;
    float beta  = (float) CANDIDATES / (float) n_pts_;

    float w0 = (c_ - 1.0f) / log(sqrt(c_));
    float w1 = 2.0f * sqrt(c_);
    float w2 = sqrt((8.0f * SQR(c_) * log(c_)) / (SQR(c_) - 1.0f));
    float p1 = -1.0f, p2 = -1.0f;

    if (fabs(p_ - 0.5f) < FLOATZERO) {
        w_ = w0;
        p1 = calc_l0_prob(w_ / 2.0f);
        p2 = calc_l0_prob(w_ / (2.0f * c_));
    }
    else if (fabs(p_ - 1.0f) < FLOATZERO) {
        w_ = w1;
        p1 = calc_l1_prob(w_ / 2.0f);
        p2 = calc_l1_prob(w_ / (2.0f * c_));
    }
    else if (fabs(p_ - 2.0f) < FLOATZERO) {
        w_ = w2;
        p1 = calc_l2_prob(w_ / 2.0f);
        p2 = calc_l2_prob(w_ / (2.0f * c_));
    }
    else {
        if (fabs(p_-0.8f) < FLOATZERO) w_ = 2.503f;
        else if (fabs(p_-1.2f) < FLOATZERO) w_ = 3.151f;
        else if (fabs(p_-1.5f) < FLOATZERO) w_ = 3.465f;
        else w_ = (w2 - w1) * p_ + (2.0f * w1 - w2);

        new_stable_prob(p_, zeta_, c_, 1.0f, w_, 1000000, p1, p2);
    }

    float para1 = sqrt(log(2.0f / beta));
    float para2 = sqrt(log(1.0f / delta));
    float para3 = 2.0f * (p1 - p2) * (p1 - p2);
    float eta   = para1 / para2;
    float alpha = (eta * p1 + p2) / (1.0f + eta);

    m_ = (int) ceil((para1 + para2) * (para1 + para2) / para3);
    l_ = (int) ceil(alpha * m_);

    // generate hash functions <a_>
    a_ = new float[m_*dim_];
    for (uint64_t i = 0; i < m_*dim_; ++i) {
        if (fabs(p_-0.5f) < FLOATZERO) a_[i] = levy(1.0f, 0.0f);
        else if (fabs(p_-1.0f) < FLOATZERO) a_[i] = cauchy(1.0f, 0.0f);
        else if (fabs(p_-2.0f) < FLOATZERO) a_[i] = gaussian(0.0f, 1.0f);
        else a_[i] = p_stable(p_, zeta_, 1.0f, 0.0f);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
QALSH<DType>::~QALSH()              // destructor
{
    delete[] a_;
    delete[] tables_;
}

// -----------------------------------------------------------------------------
template<class DType>
void QALSH<DType>::display()        // display parameters
{
    printf("Parameters of QALSH:\n");
    printf("n     = %d\n",   n_pts_);
    printf("d     = %d\n",   dim_);
    printf("p     = %.1f\n", p_);
    printf("zeta  = %.1f\n", zeta_);
    printf("c     = %.1f\n", c_);
    printf("w     = %f\n",   w_);
    printf("m     = %d\n",   m_);
    printf("l     = %d\n",   l_);
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int QALSH<DType>::knn(              // k-nn search
    int   top_k,                        // top-k value
    const DType *query,                 // input query
    MinK_List *list)                    // k-NN results (return)
{
    list->reset();

    // initialize parameters for c-k-ANNS
    int  *freq    = new int[n_pts_];  memset(freq, 0, n_pts_*sizeof(int));
    bool *checked = new bool[n_pts_]; memset(checked, false, n_pts_*sizeof(bool));

    int   *lpos  = new int[m_];
    int   *rpos  = new int[m_];
    float *q_val = new float[m_];
    for (int i = 0; i < m_; ++i) {
        Result tmp;
        tmp.key_ = calc_hash_value(i, query);
        q_val[i] = tmp.key_;

        Result *table = &tables_[(uint64_t)i*n_pts_];
        int pos = std::lower_bound(table, table+n_pts_, tmp, cmp) - table;
        if (pos == 0) { lpos[i] = -1; rpos[i] = pos; } 
        else { lpos[i] = pos; rpos[i] = pos + 1; }
    }

    // c-k-ANNS via dynamic collision counting framework
    bool  *flag  = new bool[m_];           // flag for each hash table
    int   candidates = CANDIDATES+top_k-1; // candidate size
    int   cand_cnt = 0;                    // candidate counter
    float kdist  = MAXREAL;                // k-th NN dist
    float radius = 1.0f;                   // search radius
    float width  = radius * w_ / 2.0f;     // bucket width

    while (true) {
        // step 1: initialize the stop condition for current round
        int num_flag = 0; memset(flag, true, m_*sizeof(bool));

        // step 2: (R,c)-NN search (find frequent data points)
        while (num_flag < m_) {
            for (int j = 0; j < m_; ++j) {
                if (!flag[j]) continue;

                Result *table = &tables_[(uint64_t)j*n_pts_];
                float q_v = q_val[j];
                float ldist = -1.0f, rdist = -1.0f, dist = -1.0f;
                int   cnt = -1, pos = -1, id = -1;

                // step 2.1: scan the left part of hash table
                cnt = 0; pos = lpos[j];
                while (cnt < SCAN_SIZE) {
                    ldist = MAXREAL;
                    if (pos >= 0) ldist = fabs(q_v - table[pos].key_);
                    else break;
                    if (ldist > width) break;

                    id = table[pos].id_;
                    if (++freq[id] >= l_ && !checked[id]) {
                        checked[id] = true;
                        dist = calc_lp_dist<DType>(dim_, p_, kdist, query, 
                            &data_[(uint64_t)id*dim_]);
                        kdist = list->insert(dist, id);
                        if (++cand_cnt >= candidates) break;
                    }
                    --pos; ++cnt;
                }
                if (cand_cnt >= candidates) break;
                lpos[j] = pos;

                // step 2.2: scan the right part of hash table
                cnt = 0; pos = rpos[j];
                while (cnt < SCAN_SIZE) {
                    rdist = MAXREAL;
                    if (pos < n_pts_) rdist = fabs(q_v - table[pos].key_);
                    else break;
                    if (rdist > width) break;

                    id = table[pos].id_;
                    if (++freq[id] >= l_ && !checked[id]) {
                        checked[id] = true;
                        dist = calc_lp_dist<DType>(dim_, p_, kdist, query, 
                            &data_[(uint64_t)id*dim_]);
                        kdist = list->insert(dist, id);
                        if (++cand_cnt >= candidates) break;
                    }
                    ++pos; ++cnt;
                }
                if (cand_cnt >= candidates) break;
                rpos[j] = pos;

                // step 2.3: check whether this width is finished scanned
                if (ldist > width && rdist > width) {
                    flag[j] = false;
                    if (++num_flag >= m_) break;
                }
            }
            if (num_flag >= m_ || cand_cnt >= candidates) break;
        }
        // step 3: stop conditions t1 and t2
        if (kdist < c_*radius && cand_cnt >= top_k) break;
        if (cand_cnt >= candidates) break;

        // step 4: auto-update radius
        radius = radius * c_;
        width  = radius * w_ / 2.0f;
    }
    // release space
    delete[] flag;
    delete[] q_val;
    delete[] rpos;
    delete[] lpos;
    delete[] checked;
    delete[] freq;

    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int QALSH<DType>::knn2(             // k-NN search (assis func for QALSH+)
    int   top_k,                        // top-k value
    const DType *query,                 // input query
    MinK_List *list)                    // k-NN results (return)
{
    // initialize parameters for c-k-ANNS
    int   *freq = new int[n_pts_]; memset(freq, 0, n_pts_*sizeof(int));
    bool  *checked = new bool[n_pts_]; memset(checked, false, n_pts_*sizeof(bool));
    bool  *range_flag = new bool[m_]; memset(range_flag, true,  m_*sizeof(bool));
    int   *lpos  = new int[m_];
    int   *rpos  = new int[m_];
    float *q_val = new float[m_];

    for (int i = 0; i < m_; ++i) {
        Result tmp;
        tmp.key_ = calc_hash_value(i, query);
        q_val[i] = tmp.key_;

        Result *table = &tables_[(uint64_t)i*n_pts_];
        int pos = std::lower_bound(table, table+n_pts_, tmp, cmp) - table;
        if (pos <= 0) { lpos[i] = -1; rpos[i] = pos; }
        else { lpos[i] = pos-1; rpos[i] = pos; }
    }

    // c-k-ANNS via dynamic collision counting framework
    bool *bucket_flag = new bool[m_];
    int  candidates = CANDIDATES+top_k-1; // candidate size
    int  cand_cnt = 0;                    // #candidates computation
    int  num_range = 0;                   // #search range flag
    
    float kdist  = list->max_key();       // k-th NN dist 
    float radius = 1.0f;                  // search radius
    float width  = radius*w_/2.0f;        // bucket width
    float range  = kdist>MAXREAL-1.0f ? MAXREAL : kdist*w_/2.0f; // search rangea

    while (true) {
        // step 1: initialize the stop condition for current round
        int num_buckets = 0; memset(bucket_flag, true, m_*sizeof(bool));

        // step 2: (R,c)-NN search (find frequent data points)
        while (num_buckets < m_ && num_range < m_) {
            for (int j = 0; j < m_; ++j) {
                if (!bucket_flag[j]) continue;

                Result *table = &tables_[(uint64_t)j*n_pts_];
                float q_v = q_val[j];
                float ldist = -1.0f, rdist = -1.0f, dist = -1.0f;
                int   cnt = -1, pos = -1, id = -1;
                
                // step 2.1: scan the left part of hash table
                cnt = 0; pos = lpos[j];
                while (cnt < SCAN_SIZE) {
                    ldist = MAXREAL;
                    if (pos >= 0) ldist = fabs(q_v - table[pos].key_);
                    else break;
                    if (ldist > width || ldist > range) break;

                    id = table[pos].id_;
                    if (++freq[id] >= l_ && !checked[id]) {
                        checked[id] = true;
                        dist = calc_lp_dist<DType>(dim_, p_, kdist, query, 
                            &data_[(uint64_t)index_[id]*dim_]);
                        kdist = list->insert(dist, index_[id]);
                        if (++cand_cnt >= candidates) break;
                    }
                    --pos; ++cnt;
                }
                if (cand_cnt >= candidates) break;
                lpos[j] = pos;

                // step 2.2: scan the right part of hash table
                cnt = 0; pos = rpos[j];
                while (cnt < SCAN_SIZE) {
                    rdist = MAXREAL;
                    if (pos < n_pts_) rdist = fabs(q_v - table[pos].key_);
                    else break;
                    if (rdist > width || rdist > range) break;

                    id = table[pos].id_;
                    if (++freq[id] >= l_ && !checked[id]) {
                        checked[id] = true;
                        dist = calc_lp_dist<DType>(dim_, p_, kdist, query, 
                            &data_[(uint64_t)index_[id]*dim_]);
                        kdist = list->insert(dist, index_[id]);
                        if (++cand_cnt >= candidates) break;
                    }
                    ++pos; ++cnt;
                }
                if (cand_cnt >= candidates) break;
                rpos[j] = pos;

                // step 2.3: check whether this width is finished scanned
                if (ldist > width && rdist > width) {
                    bucket_flag[j] = false;
                    if (++num_buckets > m_) break;
                }
                if (ldist > range && rdist > range) {
                    if (bucket_flag[j]) {
                        bucket_flag[j] = false;
                        if (++num_buckets > m_) break;
                    }
                    if (range_flag[j]) {
                        range_flag[j] = false;
                        if (++num_range > m_) break;
                    }
                }
            }
            if (num_buckets>m_ || num_range>m_ || cand_cnt>=candidates) break;
        }
        // step 3: stop conditions
        if (num_range >= m_ || cand_cnt >= candidates) break;

        // step 4: auto-update <radius>
        radius = radius * c_;
        width  = radius * w_ / 2.0f;
    }
    // release space
    delete[] bucket_flag;
    delete[] q_val;
    delete[] rpos;
    delete[] lpos;
    delete[] range_flag;
    delete[] checked;
    delete[] freq;

    return 0;
}

} // end namespace nns
