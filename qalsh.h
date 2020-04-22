#ifndef __QALSH_H
#define __QALSH_H

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "def.h"
#include "util.h"
#include "random.h"
#include "pri_queue.h"

struct Result;
class  MinK_List;

// -----------------------------------------------------------------------------
//  Query-Aware Locality-Sensitive Hashing (QALSH) is used to solve the problem 
//  of c-Approximate Nearest Neighbor (c-ANN) search.
//
//  the idea was introduced by Qiang Huang, Jianlin Feng, Yikai Zhang, Qiong 
//  Fang, and Wilfred Ng in their paper "Query-aware locality-sensitive hashing 
//  for approximate nearest neighbor search", in Proceedings of the VLDB 
//  Endowment (PVLDB), 9(1), pages 1–12, 2015.
// -----------------------------------------------------------------------------
class QALSH {
public:
	// -------------------------------------------------------------------------
	QALSH(							// constructor
		int   n,						// cardinality
		int   d,						// dimensionality
		float p,						// l_p distance
		float zeta,						// a parameter of p-stable distr.
		float ratio);					// approximation ratio

	// -------------------------------------------------------------------------
	QALSH(							// constructor
		int   n,						// cardinality
		int   d,						// dimensionality
		float p,						// l_p distance
		float zeta,						// a parameter of p-stable distr.
		float ratio,					// approximation ratio
		const float **data);			// data objects

	// -------------------------------------------------------------------------
	~QALSH();						// destructor

	// -------------------------------------------------------------------------
	float calc_hash_value(			// calc hash value
		int   tid,						// hash table id
		const float *data);				// one data/query object

	// -------------------------------------------------------------------------
	void display();					// display parameters

	// -------------------------------------------------------------------------
	int knn(						// k-NN search
		int   top_k,					// top-k value
		const float *query,				// input query object
		MinK_List *list);				// k-NN results (return)

	// -------------------------------------------------------------------------
	int knn(						// k-NN search
		int   top_k,					// top-k value
		float R,						// limited search range
		const float *query,				// input query object
		std::vector<int> &cand);		// object id mapping

	// -------------------------------------------------------------------------
	int    n_pts_;					// cardinality
	int    dim_;					// dimensionality
	float  p_;						// l_p distance
	float  zeta_;					// a parameter of p-stable distr.
	float  ratio_;					// approximation ratio
	float  w_;						// bucket width
	int    m_;						// number of hashtables
	int    l_;						// collision threshold

	const  float **data_;			// data objects
	float  **a_;					// hash functions
	Result **tables_;				// hash tables

protected:
	// -------------------------------------------------------------------------
	void init();					// basic initialzation

	// -------------------------------------------------------------------------
	float calc_l0_prob(				// calc <p1> and <p2> for L_{0.5} distance
		float x);						// x = w / (2.0 * r)

	float calc_l1_prob(				// calc <p1> and <p2> for L_{1.0} distance
		float x);						// x = w / (2.0 * r)

	float calc_l2_prob(				// calc <p1> and <p2> for L_{2.0} distance
		float x);						// x = w / (2.0 * r)
};

#endif // __QALSH_H
