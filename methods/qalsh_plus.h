#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "def.h"
#include "pri_queue.h"
#include "util.h"
#include "kd_tree.h"
#include "qalsh.h"

namespace nns {

// -----------------------------------------------------------------------------
//  QALSH_PLUS: an two-level QALSH scheme for high-dimensional c-k-ANN search
//  This is an internal memory implementation of the following work:
//
//  Qiang Huang, Jianlin Feng, Qiong Fang, Wilfred Ng, and Wei Wang. 
//  Query-aware locality-sensitive hashing scheme for l_p norm. The VLDB 
//  Journal, 26(5): 683-708, 2017.
// -----------------------------------------------------------------------------
template<class DType>
class QALSH_PLUS {
public:
    QALSH_PLUS(                     // constructor
        int   n,                        // number of data points
        int   d,                        // data dimension
        int   leaf,                     // leaf size of kd-tree
        int   L,                        // #projection for drusilla-select
        int   M,                        // #candidates for drusilla-select
        float p,                        // l_p distance
        float zeta,                     // symmetric factor of p-stable distr.
        float c,                        // approximation ratio
        const DType *data);             // data points

    // -------------------------------------------------------------------------
    ~QALSH_PLUS();                  // destructor

    // -------------------------------------------------------------------------
    void display();                 // display parameters

    // -------------------------------------------------------------------------
    inline int get_num_blocks() { return n_blocks_; }

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {   // get estimated memory usage
        uint64_t ret = 0ULL;
        ret += sizeof(*this);
        ret += sizeof(int)*n_pts_;  // index_
        ret += sizeof(DType)*n_blocks_*n_samples_*dim_; // sample_data_
        ret += lsh_->get_memory_usage();      // first level lsh 
        for (int i = 0; i < n_blocks_; ++i) { // second level lsh
            ret += blocks_[i]->get_memory_usage();
        }
        return ret;
    }

    // -------------------------------------------------------------------------
    int knn(                        // k-NN search
        int   top_k,                    // top-k value
        int   nb,                       // number of blocks for search
        const DType *query,             // input query
        MinK_List *list);               // k-NN results (return)

protected:
    int   n_pts_;                   // number of data points
    int   dim_;                     // data dimension
    int   n_samples_;               // number of samples for drusilla-select
    const DType *data_;             // original data points
    
    int   n_blocks_;                // number of blocks 
    int   *index_;                  // data index after kd-tree partition
    DType *sample_data_;            // sample data
    QALSH<DType> *lsh_;             // first level lsh index for sample data
    std::vector<QALSH<DType>*> blocks_; // second level lsh index for blocks

    // -------------------------------------------------------------------------
    void kd_tree_partition(         // kd-tree partition
        int leaf,                       // leaf size of kd-tree
        std::vector<int> &block_size,   // block size (return)
        int *index);                    // data index (return)

    // -------------------------------------------------------------------------
    void drusilla_select(           // drusilla select
        int   n,                        // number of data in a block
        int   L,                        // #projection for drusilla-select
        int   M,                        // #candidates for drusilla-select
        const int *index,               // data index for this block
        DType *sample_data);            // sample data (return)

    // -------------------------------------------------------------------------
    void calc_shift_data(           // calculate shift data points
        int   n,                        // number of data points in this block
        const int *index,               // data index in this block
        int   &max_id,                  // data id with max l2-norm (return)
        float &max_norm,                // max l2-norm (return)
        float *norm,                    // l2-norm of shift data (return)
        float *shift_data);             // shift data (return)

    // -------------------------------------------------------------------------
    void shift(                     // shift the original data by centroid
        const DType *data,              // original data point
        const float *centroid,          // centroid
        float &norm,                    // l2-norm of shifted data (return)
        float *shift_data);             // shifted data (return)

    // -------------------------------------------------------------------------
    void select_proj(               // select project vector
        float norm,                     // max l2-norm
        const float *shift_data,        // shift data with max l2-norm
        float *proj);                   // projection vector

    // -------------------------------------------------------------------------
    float calc_distortion(          // calc distortion
        float offset,                   // offset
        const float *proj,              // projection vector
        const float *shift_data);       // input shift data

    // -------------------------------------------------------------------------
    void copy(                      // copy original data to sample data
        int   id,                       // input data id
        DType *sample_data);            // sample data (return)

    // -------------------------------------------------------------------------
    void get_block_order(           // get block order
        int   nb,                       // number of blocks for search
        const DType *query,             // input query points
        std::vector<int> &block_order); // block order (return)
};

// -----------------------------------------------------------------------------
template<class DType>
QALSH_PLUS<DType>::QALSH_PLUS(      // constructor
    int   n,                            // number of data points
    int   d,                            // dimensionality
    int   leaf,                         // leaf size of kd-tree
    int   L,                            // #projection for drusilla-select
    int   M,                            // #candidates for drusilla-select
    float p,                            // l_p distance
    float zeta,                         // symmetric factor of p-stable distr.
    float c,                            // approximation ratio
    const DType *data)                  // data points
    : n_pts_(n), dim_(d), n_samples_(L*M), data_(data)
{
    // use kd-tree partition to get n_blocks_ and index_
    index_ = new int[n];
    std::vector<int> block_size;
    kd_tree_partition(leaf, block_size, index_);

    // init sample_data_ and build qalsh for each block
    int n_sample_pts = n_blocks_*n_samples_;
    sample_data_ = new DType[(uint64_t) n_sample_pts*dim_];

    int start = 0;
    int count = 0;
    for (int i = 0; i < n_blocks_; ++i) {
        int n_blk = block_size[i]; 
        const int *index = (const int*) &index_[start];

        // get sample data from each blcok using drusilla-select
        assert(n_blk > n_samples_);
        drusilla_select(n_blk, L, M, index, &sample_data_[(uint64_t)count*dim_]);
        
        // build qalsh for each blcok
        QALSH<DType> *lsh = new QALSH<DType>(n_blk, d, p, zeta, c, data, index);
        blocks_.push_back(lsh);

        // update parameters
        start += n_blk;
        count += n_samples_;
    }
    assert(start == n && count == n_sample_pts);

    // build qalsh for sample_data
    lsh_ = new QALSH<DType>(n_sample_pts, d, p, zeta, c, 
        (const DType*) sample_data_);
    
    block_size.clear(); block_size.shrink_to_fit();
}

// -----------------------------------------------------------------------------
template<class DType>
void QALSH_PLUS<DType>::kd_tree_partition(// kd-tree partition
    int leaf,                           // leaf size of kd-tree
    std::vector<int> &block_size,       // block size (return)
    int *index)                         // data index (return)
{
    // build a kd-tree for input data with specific leaf size
    KD_Tree<DType> *tree = new KD_Tree<DType>(n_pts_, dim_, leaf, data_);
    
    // init index_ and n_blocks_
    tree->traversal(block_size, index);
    n_blocks_ = (int) block_size.size(); assert(n_blocks_ > 0);
    delete tree;
}

// -----------------------------------------------------------------------------
template<class DType>
void QALSH_PLUS<DType>::drusilla_select(// drusilla select
    int   n,                            // number of data in a block
    int   L,                            // #projection for drusilla-select
    int   M,                            // #candidates for drusilla-select
    const int *index,                   // data index
    DType *sample_data)                 // sample data (return)
{
    // calc shift data
    int   max_id      = -1;
    float max_norm    = MINREAL;
    float *norm       = new float[n];
    float *shift_data = new float[(uint64_t)n*dim_];
    
    calc_shift_data(n, index, max_id, max_norm, norm, shift_data);

    // drusilla select
    float  *proj        = new float[dim_];
    Result *score       = new Result[n];
    bool   *close_angle = new bool[n];
    float  offset       = -1.0f;
    float  distortion   = -1.0f;

    for (int i = 0; i < L; ++i) {
        // select the projection vector with largest norm and normalize it
        select_proj(norm[max_id], &shift_data[(uint64_t)max_id*dim_], proj);
        
        // calculate offsets and distortions
        for (int j = 0; j < n; ++j) {
            close_angle[j] = false;
            score[j].id_ = j;

            if (norm[j] > 0.0f) {
                const float *tmp = &shift_data[(uint64_t)j*dim_];
                offset = calc_inner_product<float>(dim_, (const float*)proj, tmp);
                distortion = calc_distortion(offset, (const float*)proj, tmp);
                score[j].key_ = offset*offset - distortion;

                if (atan(sqrt(distortion) / fabs(offset)) < ANGLE) {
                    close_angle[j] = true;
                }
            }
            else if (fabs(norm[j]) < FLOATZERO) {
                score[j].key_ = MINREAL + 1.0f;
            }
            else {
                score[j].key_ = MINREAL;
            }
        }
        // collect the points that are well-represented by this projection
        qsort(score, n, sizeof(Result), ResultCompDesc);
        for (int j = 0; j < M; ++j) {
            int id = score[j].id_;
            copy(index[id], &sample_data[(uint64_t)(i*M+j)*dim_]);
            norm[id] = -1.0f;
        }
        // find the next data id (max_id) with max l2-norm
        max_id = -1; max_norm = MINREAL;
        for (int j = 0; j < n; ++j) {
            if (norm[j] > 0.0f && close_angle[j]) { norm[j] = 0.0f; }
            if (norm[j] > max_norm) { max_norm = norm[j]; max_id = j; }
        }
    }
    // release space
    delete[] close_angle;
    delete[] score;
    delete[] proj;
    delete[] norm;
    delete[] shift_data;
}

// -----------------------------------------------------------------------------
template<class DType>
void QALSH_PLUS<DType>::calc_shift_data(// calculate shift data points
    int   n,                            // number of data points
    const int *index,                   // data index
    int   &max_id,                      // data id with max l2-norm (return)
    float &max_norm,                    // max l2-norm (return)
    float *norm,                        // l2-norm of shift data (return)
    float *shift_data)                  // shift data (return)
{
    // calculate the centroid of data points 
    float *centroid = new float[dim_]; 
    memset(centroid, 0.0, dim_*sizeof(float));
    for (int i = 0; i < n; ++i) {
        const DType *tmp = &data_[(uint64_t) index[i]*dim_];
        for (int j = 0; j < dim_; ++j) {
            centroid[j] += (float) tmp[j];
        }
    }
    for (int i = 0; i < dim_; ++i) centroid[i] /= n;

    // calc shift data and their l2-norm and find max l2-norm and its id
    max_id = -1; max_norm = MINREAL;
    for (int i = 0; i < n; ++i) {
        shift(&data_[(uint64_t) index[i]*dim_], centroid, norm[i], 
            &shift_data[(uint64_t) i*dim_]);
        if (norm[i] > max_norm) { max_norm = norm[i]; max_id = i; }
    }
    delete[] centroid;
}

// -----------------------------------------------------------------------------
template<class DType>
void QALSH_PLUS<DType>::shift(      // shift the original data by centroid
    const DType *data,                  // original data point
    const float *centroid,              // centroid
    float &norm,                        // l2-norm of shifted data (return)
    float *shift_data)                  // shifted data (return)
{
    norm = 0.0f;
    for (int j = 0; j < dim_; ++j) {
        float tmp = (float) data[j] - centroid[j];
        shift_data[j] = tmp; norm += tmp * tmp;
    }
    norm = sqrt(norm);
}

// -----------------------------------------------------------------------------
template<class DType>
void QALSH_PLUS<DType>::select_proj(// select project vector
    float norm,                         // max l2-norm
    const float *shift_data,            // shift data with max l2-norm
    float *proj)                        // projection vector
{
    for (int j = 0; j < dim_; ++j) {
        proj[j] = shift_data[j] / norm;
    }
}

// -----------------------------------------------------------------------------
template<class DType>
float QALSH_PLUS<DType>::calc_distortion(// calc distortion (l2-sqr)
    float offset,                       // offset
    const float *proj,                  // projection vector
    const float *shift_data)            // input shift data
{
    float distortion = 0.0f;
    for (int j = 0; j < dim_; ++j) {
        float tmp = shift_data[j] - offset * proj[j];
        distortion += tmp * tmp;
    }
    return distortion;
}

// -----------------------------------------------------------------------------
template<class DType>
void QALSH_PLUS<DType>::copy(       // copy original data with id to sample_data
    int   id,                           // input data id
    DType *sample_data)                 // sample data (return)
{
    const DType *data = &data_[(uint64_t) id*dim_];
    for (int i = 0; i < dim_; ++i) {
        sample_data[i] = data[i];
    }
}

// -----------------------------------------------------------------------------
template<class DType>
QALSH_PLUS<DType>::~QALSH_PLUS()    // destructor
{
    for (int i = 0; i < n_blocks_; ++i) {
        delete blocks_[i]; blocks_[i] = NULL;
    }
    blocks_.clear(); blocks_.shrink_to_fit();
    delete lsh_;

    delete[] sample_data_;
    delete[] index_;
}

// -----------------------------------------------------------------------------
template<class DType>
void QALSH_PLUS<DType>::display()   // display parameters
{
    printf("Parameters of QALSH+:\n");
    printf("n         = %d\n", n_pts_);
    printf("d         = %d\n", dim_);
    printf("n_samples = %d\n", n_samples_);
    printf("n_blocks  = %d\n", n_blocks_);
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int QALSH_PLUS<DType>::knn(         // c-k-ANN search
    int   top_k,                        // top-k value
    int   nb,                           // number of blocks for search
    const DType *query,                 // input query points
    MinK_List *list)                    // k-NN results
{
    assert(nb > 0 && nb <= n_blocks_);
    list->reset();

    // use sample data to determine block_order (nb size) for k-NN search
    std::vector<int> block_order;
    get_block_order(nb, query, block_order);
    
    // use `nb` blocks for k-NN search
    for (int bid : block_order) {
        // find candidates by qalsh for this block
        blocks_[bid]->knn2(top_k, query, list);
    }
    block_order.clear(); block_order.shrink_to_fit();
    
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
void QALSH_PLUS<DType>::get_block_order(// get block order
    int   nb,                           // number of blocks for search
    const DType *query,                 // input query points
    std::vector<int> &block_order)      // block order (return)
{
    // get candidates from first level qalsh
    MinK_List *list = new MinK_List(MAXK);
    lsh_->knn(MAXK, query, list);

    // init the counter of each block
    Result *pair = new Result[n_blocks_];
    for (int i = 0; i < n_blocks_; ++i) {
        pair[i].id_  = i;
        pair[i].key_ = 0.0f;
    }

    // select the first <nb> blocks with largest counters
    for (int i = 0; i < list->size(); ++i) {
        int block_id = list->ith_id(i) / n_samples_;
        pair[block_id].key_ += 1.0f;
    }
    qsort(pair, n_blocks_, sizeof(Result), ResultCompDesc);
    
    for (int i = 0; i < nb; ++i) {
        // if (fabs(pair[i].key_) < FLOATZERO) break;
        block_order.push_back(pair[i].id_);
    }
    delete[] pair;
    delete   list;
}

} // end namespace nns
