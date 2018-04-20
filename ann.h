#ifndef __ANN_H
#define __ANN_H

// -----------------------------------------------------------------------------
int ground_truth(					// find ground truth
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimensionality
	float p,							// the p value of Lp norm, p in (0,2]
	const char *data_set,				// address of data  set
	const char *query_set,				// address of query set
	const char *truth_set);				// address of truth set

// -----------------------------------------------------------------------------
int qalsh_plus(						// k-NN search of qalsh+
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimensionality
	int   kd_leaf_size,					// leaf size of kd-tree
	int   L,							// number of projection (drusilla)
	int   M,							// number of candidates (drusilla)
	int   nb,							// number of blocks for search
	float p,							// the p value of Lp norm, p in (0,2]
	float zeta,							// symmetric factor of p-stable distr.
	float ratio,						// approximation ratio
	const char *data_set,				// address of data  set
	const char *query_set,				// address of query set
	const char *truth_set,				// address of truth set
	const char *output_folder);			// output folder

// -----------------------------------------------------------------------------
int qalsh(							// k-NN search of qalsh
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimensionality
	float p,							// the p value of Lp norm, p in (0,2]
	float zeta,							// symmetric factor of p-stable distr.
	float ratio,						// approximation ratio
	const char *data_set,				// address of data  set
	const char *query_set,				// address of query set
	const char *truth_set,				// address of truth set
	const char *output_folder);			// output folder

// -----------------------------------------------------------------------------
int linear_scan(					// k-NN search of linear scan method
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimensionality
	float p,							// the p value of Lp norm, p in (0,2]
	const char *data_set,				// address of data  set
	const char *query_set,				// address of query set
	const char *truth_set,				// address of truth set
	const char *output_folder);			// output folder

#endif // __ANN_H
