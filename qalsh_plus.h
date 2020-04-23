#ifndef __QALSH_PLUS_H
#define __QALSH_PLUS_H

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "kd_tree.h"
#include "qalsh.h"

class QALSH;
class MinK_List;

// -----------------------------------------------------------------------------
//  Block: an block which stores hash tables for some of data objects
// -----------------------------------------------------------------------------
struct Block {
	int   n_pts_;
	int   *index_;
	QALSH *lsh_;

	Block() { n_pts_ = -1; index_ = NULL; lsh_ = NULL; }
	~Block() { if (lsh_ != NULL) { delete lsh_; lsh_ = NULL; } }
};

// -----------------------------------------------------------------------------
//  QALSH_PLUS: an two-level LSH scheme for high-dimensional c-k-ANN search
// -----------------------------------------------------------------------------
class QALSH_PLUS {
public:
	QALSH_PLUS(						// constructor
		int   n,						// cardinality
		int   d,						// dimensionality
		int   leaf,						// leaf size of kd-tree
		int   L,						// number of projection
		int   M,						// number of candidates for each proj
		float p,						// l_p distance
		float zeta,						// symmetric factor of p-stable distr.
		float ratio,					// approximation ratio
		const float **data);			// data objects

	// -------------------------------------------------------------------------
	~QALSH_PLUS();					// destructor

	// -------------------------------------------------------------------------
	void display();					// display parameters

	// -------------------------------------------------------------------------
	int knn(						// k-NN seach	
		int   top_k,					// top-k value
		int   nb,						// number of blocks for search
		const float *query,				// input query object
		MinK_List *list);				// k-NN results (return)

	// -------------------------------------------------------------------------
	int   n_pts_;					// cardinality
	int   dim_;						// dimensionality
	int   L_;						// number of projection (drusilla)
	int   M_;						// number of candidates (drusilla)
	float p_;						// l_p distance
	int   num_blocks_;				// number of blocks 
	
	const float **data_;			// data objects
	int   *new_order_id_;			// new order id after kd-tree partition
	float **sample_data_;			// sample data
	QALSH *lsh_;					// index of sample data objects
	std::vector<Block*> blocks_;	// index of blocks

protected:
	// -------------------------------------------------------------------------
	void kd_tree_partition(			// kd-tree partition 
		int leaf,						// leaf size of kd-tree
		std::vector<int> &block_size,	// block size (return)
		int *new_order_id);				// new order id (return)

	// -------------------------------------------------------------------------
	void drusilla_select(			// drusilla select
		int   n,						// number of data objects
		const int *new_order_id,		// new order data id
		int   *sample_id);				// sample data id (return)

	// -------------------------------------------------------------------------
	void calc_shift_data(			// calculate shift data objects
		int   n,						// number of data objects
		const int *new_order_id,		// new order data id
		int   &max_id,					// data id with max l2-norm (return)
		float &max_norm,				// max l2-norm (return)
		float *norm,					// l2-norm of shift data (return)
		float **shift_data); 			// shift data (return)

	// -------------------------------------------------------------------------
	void get_block_order(			// get block order
		int nb,							// number of blocks for search
		MinK_List *cand,				// candidates
		std::vector<int> &block_order);	// block order (return)
};

#endif // QALSH_PLUS
