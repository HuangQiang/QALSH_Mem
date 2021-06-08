#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>

#include "def.h"
#include "pri_queue.h"

namespace nns {

extern timeval g_start_time;		// global parameter: start time
extern timeval g_end_time;			// global parameter: end time

extern float g_indexing_time;		// global parameter: indexing time
extern float g_estimated_mem;		// global parameter: estimated memory
extern float g_runtime;				// global parameter: running time
extern float g_ratio;				// global parameter: overall ratio
extern float g_recall;				// global parameter: recall

// -----------------------------------------------------------------------------
void create_dir(					// create dir if the path exists
	char *path);						// input path

// -----------------------------------------------------------------------------
int write_ground_truth(				// write ground truth to disk
	int   n,							// number of ground truth results
	int   d,			 				// dimension of ground truth results
	float p,							// l_p distance
	const char *prefix,					// prefix of truth set
	const Result *truth);				// ground truth

// -----------------------------------------------------------------------------
float calc_ratio(					// calc overall ratio [1,\infinity)
	int   k,							// top-k value
	const Result *truth,				// ground truth results 
	MinK_List *list);					// top-k approximate results

// -----------------------------------------------------------------------------
float calc_recall(					// calc recall (percentage)
	int   k,							// top-k value
	const Result *truth,				// ground truth results 
	MinK_List *list);					// results returned by algorithms

// -----------------------------------------------------------------------------
template<class DType>
int read_data(						// read data (binary) from disk
	int   n,							// cardinality
	int   d,			 				// dimensionality
	int   sign,							// 0-data; 1-query; 2-truth
	float p,							// L_p distance
	const char *prefix,					// prefix of data set
	DType *data)						// data (return)
{
	char fname[200]; 
	switch (sign) {
		case 0: sprintf(fname, "%s.ds", prefix); break;
		case 1: sprintf(fname, "%s.q", prefix); break;
		case 2: sprintf(fname, "%s.gt%3.1f", prefix, p); break;
		default: printf("Parameters error!\n"); return 1;
	}
	
	FILE *fp = fopen(fname, "rb");
	if (!fp) { printf("Could not open %s\n", fname); return 1; }

	uint64_t size = (uint64_t) n*d;
	uint32_t max_uint = 4294967295U;
	if (size < max_uint) {
		// if no longer than size_t, read the whole array directly
		fread(data, sizeof(DType), (uint32_t) size, fp);
	}
	else {
		// we store data points in linear order, n*d == d*n (save multi times)
		for (int i = 0; i < d; ++i) {
			fread(&data[(uint64_t)i*n], sizeof(DType), n, fp);
		}
	}
	fclose(fp);
	return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
float calc_inner_product(			// calc inner product
	int   dim,							// dimension
	const float *p1,					// 1st point
	const DType *p2)					// 2nd point
{
	float r = 0.0f;
	for (int i = 0; i < dim; ++i) r += (float) (p1[i] * p2[i]);
	return r;
}

// -----------------------------------------------------------------------------
template<class DType>
float calc_l2_sqr(					// calc l2 square distance
	int   dim,							// dimension
	float threshold,					// threshold
	const DType *p1,					// 1st point
	const DType *p2)					// 2nd point
{
	unsigned d = dim & ~unsigned(7);
	const DType *aa = p1, *end_a = aa + d;
	const DType *bb = p2, *end_b = bb + d;

	// -------------------------------------------------------------------------
	// __builtin_prefetch (const void *addr[, rw[, locality]])
	// addr (required): Represents the address of the memory.
	//
	// rw (optional): A compile-time constant which can take the values:
	//   0 (default): prepare the prefetch for a read
	//   1 : prepare the prefetch for a write to the memory
	// 
	// locality (optional): A compile-time constant integer which can take the
	// following temporal locality (L) values:
	//   0: None, the data can be removed from the cache after the access
	//   1: Low, L3 cache, leave the data in L3 cache after access
	//   2: Moderate, L2 cache, leave the data in L2, L3 cache after access
	//   3 (default): High, L1 cache, leave the data in L1, L2, and L3 cache
	// -------------------------------------------------------------------------
	const int SHIFT = 8 * sizeof(DType);
	__builtin_prefetch(aa, 0, 3);
	__builtin_prefetch(bb, 0, 0);

	float r = 0.0f;
	float r0, r1, r2, r3, r4, r5, r6, r7;

	const DType *a = end_a, *b = end_b;

	r0 = r1 = r2 = r3 = r4 = r5 = r6 = r7 = 0.0f;
	switch (dim & 7) {
		case 7: r6 = (float) SQR(a[6] - b[6]);
		case 6: r5 = (float) SQR(a[5] - b[5]);
		case 5: r4 = (float) SQR(a[4] - b[4]);
		case 4: r3 = (float) SQR(a[3] - b[3]);
		case 3: r2 = (float) SQR(a[2] - b[2]);
		case 2: r1 = (float) SQR(a[1] - b[1]);
		case 1: r0 = (float) SQR(a[0] - b[0]);
	}

	a = aa; b = bb;
	for (; a < end_a; a += 8, b += 8) {
		__builtin_prefetch(a+SHIFT, 0, 3);
		__builtin_prefetch(b+SHIFT, 0, 0);

		r += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
		if (r > threshold) return r;

		r0 = (float) SQR(a[0] - b[0]);
		r1 = (float) SQR(a[1] - b[1]);
		r2 = (float) SQR(a[2] - b[2]);
		r3 = (float) SQR(a[3] - b[3]);
		r4 = (float) SQR(a[4] - b[4]);
		r5 = (float) SQR(a[5] - b[5]);
		r6 = (float) SQR(a[6] - b[6]);
		r7 = (float) SQR(a[7] - b[7]);
	}
	r += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
	
	return r;
}

// -----------------------------------------------------------------------------
template<class DType>
float calc_l1_dist(					// calc Manhattan distance
	int   dim,							// dimension
	float threshold,					// threshold
	const DType *p1,					// 1st point
	const DType *p2)					// 2nd point
{
	unsigned d = dim & ~unsigned(7);
	const DType *aa = p1, *end_a = aa + d;
	const DType *bb = p2, *end_b = bb + d;

	const int SHIFT = 8 * sizeof(DType);
	__builtin_prefetch(aa, 0, 3);
	__builtin_prefetch(bb, 0, 0);

	float r = 0.0f;
	float r0, r1, r2, r3, r4, r5, r6, r7;

	const DType *a = end_a, *b = end_b;

	r0 = r1 = r2 = r3 = r4 = r5 = r6 = r7 = 0.0f;
	switch (dim & 7) {
		case 7: r6 = (float) fabs(a[6] - b[6]);
		case 6: r5 = (float) fabs(a[5] - b[5]);
		case 5: r4 = (float) fabs(a[4] - b[4]);
		case 4: r3 = (float) fabs(a[3] - b[3]);
		case 3: r2 = (float) fabs(a[2] - b[2]);
		case 2: r1 = (float) fabs(a[1] - b[1]);
		case 1: r0 = (float) fabs(a[0] - b[0]);
	}

	a = aa; b = bb;
	for (; a < end_a; a += 8, b += 8) {
		__builtin_prefetch(a+SHIFT, 0, 3);
		__builtin_prefetch(b+SHIFT, 0, 0);

		r += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
		if (r > threshold) return r;

		r0 = (float) fabs(a[0] - b[0]);
		r1 = (float) fabs(a[1] - b[1]);
		r2 = (float) fabs(a[2] - b[2]);
		r3 = (float) fabs(a[3] - b[3]);
		r4 = (float) fabs(a[4] - b[4]);
		r5 = (float) fabs(a[5] - b[5]);
		r6 = (float) fabs(a[6] - b[6]);
		r7 = (float) fabs(a[7] - b[7]);
	}
	r += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
	
	return r;
}

// -----------------------------------------------------------------------------
template<class DType>
float calc_l0_sqrt(					// calc L_{0.5} sqrt distance
	int   dim,							// dimension
	float threshold,					// threshold
	const DType *p1,					// 1st point
	const DType *p2)					// 2nd point
{
	unsigned d = dim & ~unsigned(7);
	const DType *aa = p1, *end_a = aa + d;
	const DType *bb = p2, *end_b = bb + d;

	const int SHIFT = 8 * sizeof(DType);
	__builtin_prefetch(aa, 0, 3);
	__builtin_prefetch(bb, 0, 0);

	float r = 0.0f;
	float r0, r1, r2, r3, r4, r5, r6, r7;

	const DType *a = end_a, *b = end_b;

	r0 = r1 = r2 = r3 = r4 = r5 = r6 = r7 = 0.0f;
	switch (dim & 7) {
		case 7: r6 = sqrt((float) fabs(a[6] - b[6]));
		case 6: r5 = sqrt((float) fabs(a[5] - b[5]));
		case 5: r4 = sqrt((float) fabs(a[4] - b[4]));
		case 4: r3 = sqrt((float) fabs(a[3] - b[3]));
		case 3: r2 = sqrt((float) fabs(a[2] - b[2]));
		case 2: r1 = sqrt((float) fabs(a[1] - b[1]));
		case 1: r0 = sqrt((float) fabs(a[0] - b[0]));
	}

	a = aa; b = bb;
	for (; a < end_a; a += 8, b += 8) {
		__builtin_prefetch(a+SHIFT, 0, 3);
		__builtin_prefetch(b+SHIFT, 0, 0);

		r += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
		if (r > threshold) return r;

		r0 = sqrt((float) fabs(a[0] - b[0]));
		r1 = sqrt((float) fabs(a[1] - b[1]));
		r2 = sqrt((float) fabs(a[2] - b[2]));
		r3 = sqrt((float) fabs(a[3] - b[3]));
		r4 = sqrt((float) fabs(a[4] - b[4]));
		r5 = sqrt((float) fabs(a[5] - b[5]));
		r6 = sqrt((float) fabs(a[6] - b[6]));
		r7 = sqrt((float) fabs(a[7] - b[7]));
	}
	r += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
	
	return r;
}

// -----------------------------------------------------------------------------
template<class DType>
float calc_lp_pow(					// calc L_p pow_p distance
	int   dim,							// dimension
	float p,							// the p value of L_p norm, p in (0,2]
	float threshold,					// threshold
	const DType *p1,					// 1st point
	const DType *p2)					// 2nd point
{
	unsigned d = dim & ~unsigned(7);
	const DType *aa = p1, *end_a = aa + d;
	const DType *bb = p2, *end_b = bb + d;

	const int SHIFT = 8 * sizeof(DType);
	__builtin_prefetch(aa, 0, 3);
	__builtin_prefetch(bb, 0, 0);

	float r = 0.0f;
	float r0, r1, r2, r3, r4, r5, r6, r7;

	const DType *a = end_a, *b = end_b;

	r0 = r1 = r2 = r3 = r4 = r5 = r6 = r7 = 0.0f;
	switch (dim & 7) {
		case 7: r6 = pow((float) fabs(a[6] - b[6]), p);
		case 6: r5 = pow((float) fabs(a[5] - b[5]), p);
		case 5: r4 = pow((float) fabs(a[4] - b[4]), p);
		case 4: r3 = pow((float) fabs(a[3] - b[3]), p);
		case 3: r2 = pow((float) fabs(a[2] - b[2]), p);
		case 2: r1 = pow((float) fabs(a[1] - b[1]), p);
		case 1: r0 = pow((float) fabs(a[0] - b[0]), p);
	}

	a = aa; b = bb;
	for (; a < end_a; a += 8, b += 8) {
		__builtin_prefetch(a+SHIFT, 0, 3);
		__builtin_prefetch(b+SHIFT, 0, 0);

		r += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
		if (r > threshold) return r;

		r0 = pow((float) fabs(a[0] - b[0]), p);
		r1 = pow((float) fabs(a[1] - b[1]), p);
		r2 = pow((float) fabs(a[2] - b[2]), p);
		r3 = pow((float) fabs(a[3] - b[3]), p);
		r4 = pow((float) fabs(a[4] - b[4]), p);
		r5 = pow((float) fabs(a[5] - b[5]), p);
		r6 = pow((float) fabs(a[6] - b[6]), p);
		r7 = pow((float) fabs(a[7] - b[7]), p);
	}
	r += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
	
	return r;
}

// -----------------------------------------------------------------------------
template<class DType>
float calc_lp_dist(					// calc l_p distance
	int   dim,							// dimension
	float p,							// l_p distance, p \in (0,2]
	float threshold,					// threshold
	const DType *p1,					// 1st point
	const DType *p2)					// 2nd point
{
	if (fabs(p - 2.0f) < FLOATZERO) {
		return sqrt(calc_l2_sqr<DType>(dim, SQR(threshold), p1, p2));
	}
	else if (fabs(p - 1.0f) < FLOATZERO) {
		return calc_l1_dist<DType>(dim, threshold, p1, p2);
	}
	else if (fabs(p - 0.5f) < FLOATZERO) {
		float ret = calc_l0_sqrt<DType>(dim, sqrt(threshold), p1, p2);
		return SQR(ret);
	}
	else {
		float ret = calc_lp_pow<DType>(dim, p, pow(threshold, p), p1, p2);
		return pow(ret, 1.0f / p);
	}
}

// -----------------------------------------------------------------------------
template<class DType>
void kNN_search(					// k-NN search
	int   n, 							// cardinality
	int   d, 							// dimensionality
	int   k,							// top-k value
	float p,							// l_p distance, p \in (0,2]
	const DType *data,					// data points
	const DType *query,					// query point
	MinK_List *list)					// top-k results (return)
{
	float dist = -1.0f, kdist = MAXREAL;
	list->reset();
	for (int j = 0; j < n; ++j) {
		dist = calc_lp_dist<DType>(d, p, kdist, &data[(uint64_t)j*d], query);
		kdist = list->insert(dist, j+1);
	}
}

} // end namespace nns


