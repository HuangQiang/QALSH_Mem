#ifndef __QALSH_H
#define __QALSH_H

#include <vector>
using namespace std;

class  MinK_List;
struct Result;

// -----------------------------------------------------------------------------
//  QALSH: an LSH scheme for for high-dimensional c-k-ANN search
// -----------------------------------------------------------------------------
class QALSH {
public:
	QALSH();						// constructor
	~QALSH();						// destructor

	// -------------------------------------------------------------------------
	int build(						// build index
		int   n,						// cardinality
		int   d,						// dimensionality
		float p,						// l_p distance
		float zeta,						// a parameter of p-stable distr.
		float ratio,					// approximation ratio
		const float **data);			// data objects

	// -------------------------------------------------------------------------
	int knn(						// k-NN search
		int top_k,						// top-k value
		const float *query,				// input query object
		MinK_List *list);				// k-NN results (return)

	// -------------------------------------------------------------------------
	int knn(						// k-NN search
		int top_k,						// top-k value
		float R,						// limited search range
		const float *query,				// input query object
		const vector<int> &object_id,	// object id mapping
		MinK_List *list);				// k-NN results (return)

protected:
	// -------------------------------------------------------------------------
	int    n_pts_;					// cardinality
	int    dim_;					// dimensionality
	float  p_;						// l_p distance
	float  zeta_;					// a parameter of p-stable distr.
	float  appr_ratio_;				// approximation ratio
	const  float **data_;			// data objects

	float  w_;						// bucket width
	float  p1_;						// positive probability
	float  p2_;						// negative probability
	float  alpha_;					// collision threshold percentage
	float  beta_;					// false positive percentage
	float  delta_;					// error probability
	int    m_;						// number of hashtables
	int    l_;						// collision threshold
	float  *a_array_;				// hash functions
	Result **tables_;				// hash tables

	int    *freq_;					// frequency		
	int    *lpos_;					// left  position of hash table
	int    *rpos_;					// right position of hash table
	bool   *checked_;				// whether checked
	bool   *bucket_flag_;			// bucket flag
	bool   *range_flag_;			// range flag
	float  *q_val_;					// hash value of query
	
	// -------------------------------------------------------------------------
	void calc_params();				// calc parameters of qalsh

	// -------------------------------------------------------------------------
	float calc_l0_prob(				// calc <p1> and <p2> for L_{0.5} distance
		float x);						// x = w / (2.0 * r)

	float calc_l1_prob(				// calc <p1> and <p2> for L_{1.0} distance
		float x);						// x = w / (2.0 * r)

	float calc_l2_prob(				// calc <p1> and <p2> for L_{2.0} distance
		float x);						// x = w / (2.0 * r)

	// -------------------------------------------------------------------------
	void gen_hash_func();			// generate hash functions

	// -------------------------------------------------------------------------
	void display();					// display parameters
	
	// -------------------------------------------------------------------------
	void bulkload();				// build hash tables	

	// -------------------------------------------------------------------------
	float calc_hash_value(			// calc hash value
		int table_id,					// hash table id
		const float *point);			// one point

	// -------------------------------------------------------------------------
	int binary_search_pos(			// find position by binary search
		int table_id,					// hash table id
		float value);					// hash value
};

#endif // __QALSH_H
