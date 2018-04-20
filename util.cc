#include "headers.h"


// -----------------------------------------------------------------------------
int ResultComp(						// compare function for qsort (ascending)
	const void *e1,						// 1st element
	const void *e2)						// 2nd element
{
	int ret = 0;
	Result *item1 = (Result*) e1;
	Result *item2 = (Result*) e2;

	if (item1->key_ < item2->key_) {
		ret = -1;
	} 
	else if (item1->key_ > item2->key_) {
		ret = 1;
	} 
	else {
		if (item1->id_ < item2->id_) ret = -1;
		else if (item1->id_ > item2->id_) ret = 1;
	}
	return ret;
}

// -----------------------------------------------------------------------------
int ResultCompDesc(					// compare function for qsort (descending)
	const void *e1,						// 1st element
	const void *e2)						// 2nd element
{
	int ret = 0;
	Result *item1 = (Result*) e1;
	Result *item2 = (Result*) e2;

	if (item1->key_ < item2->key_) {
		ret = 1;
	} 
	else if (item1->key_ > item2->key_) {
		ret = -1;
	} 
	else {
		if (item1->id_ < item2->id_) ret = -1;
		else if (item1->id_ > item2->id_) ret = 1;
	}
	return ret;
}

// -----------------------------------------------------------------------------
void create_dir(					// create dir if the path exists
	char *path)							// input path
{
	int len = (int) strlen(path);
	for (int i = 0; i < len; ++i) {
		if (path[i] == '/') {
			char ch = path[i + 1];
			path[i + 1] = '\0';
									// check whether the directory exists
			int ret = access(path, F_OK);
			if (ret != 0) {			// create the directory
				ret = mkdir(path, 0755);
				if (ret != 0) {
					printf("Could not create directory %s\n", path);
				}
			}
			path[i + 1] = ch;
		}
	}
}

// -----------------------------------------------------------------------------
int read_data(						// read data/query set from disk
	int   n,							// number of data/query objects
	int   d,			 				// dimensionality
	const char *fname,					// address of data/query set
	float **data)						// data/query objects (return)
{
	FILE *fp = fopen(fname, "r");
	if (!fp) {
		printf("Could not open %s.\n", fname);
		return 1;
	}

	int i   = 0;
	int tmp = -1;
	while (!feof(fp) && i < n) {
		fscanf(fp, "%d", &tmp);
		for (int j = 0; j < d; ++j) {
			fscanf(fp, " %f", &data[i][j]);
		}
		fscanf(fp, "\n");

		++i;
	}
	assert(feof(fp) && i == n);
	fclose(fp);

	return 0;
}

// -----------------------------------------------------------------------------
int read_ground_truth(				// read ground truth results from disk
	int qn,								// number of query objects
	const char *fname,					// address of truth set
	float **R)							// ground truth results (return)
{
	FILE *fp = fopen(fname, "r");
	if (!fp) {
		printf("Could not open %s\n", fname);
		return 1;
	}

	int tmp1 = -1;
	int tmp2 = -1;
	fscanf(fp, "%d %d\n", &tmp1, &tmp2);
	assert(tmp1 == qn && tmp2 == MAXK);

	for (int i = 0; i < qn; ++i) {
		fscanf(fp, "%d", &tmp1);
		for (int j = 0; j < MAXK; ++j) {
			fscanf(fp, " %f", &R[i][j]);
		}
		fscanf(fp, "\n");
	}
	fclose(fp);

	return 0;
}

// -----------------------------------------------------------------------------
float calc_lp_dist(					// calc L_{p} norm
	int   dim,							// dimension
	float p,							// the p value of Lp norm, p in (0, 2]
	const float *vec1,					// 1st point
	const float *vec2)					// 2nd point
{
	float diff = 0.0f;
	float ret  = 0.0f;

	// ---------------------------------------------------------------------
	//  calc L_{0.5} norm
	// ---------------------------------------------------------------------
	if (fabs(p - 0.5f) < FLOATZERO) {
		for (int i = 0; i < dim; ++i) {
			diff = fabs(vec1[i] - vec2[i]);
			ret += sqrt(diff);
		}
		return ret * ret;
	}
	// ---------------------------------------------------------------------
	//  calc L_{1.0} norm
	// ---------------------------------------------------------------------
	else if (fabs(p - 1.0f) < FLOATZERO) {
		for (int i = 0; i < dim; ++i) {
			ret += fabs(vec1[i] - vec2[i]);
		}
		return ret;
	}
	// ---------------------------------------------------------------------
	//  calc L_{2.0} norm
	// ---------------------------------------------------------------------
	else if (fabs(p - 2.0f) < FLOATZERO) {
		for (int i = 0; i < dim; ++i) {
			diff = vec1[i] - vec2[i];
			ret += diff * diff;
		}
		return sqrt(ret);
	}
	// ---------------------------------------------------------------------
	//  calc other L_{p} norm (general way)
	// ---------------------------------------------------------------------
	else {
		for (int i = 0; i < dim; ++i) {
			diff = fabs(vec1[i] - vec2[i]);
			ret += pow(diff, p);
		}
		return pow(ret, 1.0f / p);
	}
}
