#include "headers.h"

// -----------------------------------------------------------------------------
int ground_truth(					// find ground truth
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimensionality
	float p,							// the p value of Lp norm, p in (0,2]
	const char *data_set,				// address of data  set
	const char *query_set,				// address of query set
	const char *truth_set)				// address of truth set
{
	timeval start_time, end_time;

	// -------------------------------------------------------------------------
	//  read data set and query set
	// -------------------------------------------------------------------------
	gettimeofday(&start_time, NULL);
	float **data = new float*[n];
	for (int i = 0; i < n; ++i) data[i] = new float[d];
	if (read_data(n, d, data_set, data) == 1) {
		printf("Reading Dataset Error!\n");
		exit(1);
	}

	float **query = new float*[qn];
	for (int i = 0; i < qn; ++i) query[i] = new float[d];
	if (read_data(qn, d, query_set, query) == 1) {
		printf("Reading Query Set Error!\n");
		exit(1);
	}

	gettimeofday(&end_time, NULL);
	float read_file_time = end_time.tv_sec - start_time.tv_sec + 
		(end_time.tv_usec - start_time.tv_usec) / 1000000.0f;
	printf("Read Dataset and Query Set: %f Seconds\n\n", read_file_time);

	// -------------------------------------------------------------------------
	//  find ground truth results (using linear scan method)
	// -------------------------------------------------------------------------
	gettimeofday(&start_time, NULL);
	FILE *fp = fopen(truth_set, "w");
	if (!fp) {
		printf("Could not create %s.\n", truth_set);
		return 1;
	}

	MinK_List *list = new MinK_List(MAXK);
	fprintf(fp, "%d %d\n", qn, MAXK);
	for (int i = 0; i < qn; ++i) {
		list->reset();
		for (int j = 0; j < n; ++j) {
			float dist = calc_lp_dist(d, p, data[j], query[i]);
			list->insert(dist, j);
		}

		fprintf(fp, "%d", i + 1);
		for (int j = 0; j < MAXK; ++j) {
			fprintf(fp, " %f", list->ith_key(j));
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	gettimeofday(&end_time, NULL);
	float truth_time = end_time.tv_sec - start_time.tv_sec + 
		(end_time.tv_usec - start_time.tv_usec) / 1000000.0f;
	printf("Ground Truth: %f Seconds\n\n", truth_time);

	// -------------------------------------------------------------------------
	//  release space
	// -------------------------------------------------------------------------
	for (int i = 0; i < n; ++i) {
		delete[] data[i]; data[i] = NULL;
	}
	delete[] data; data = NULL;
	
	for (int i = 0; i < qn; ++i) {
		delete[] query[i]; query[i] = NULL;
	}
	delete[] query; query = NULL;

	return 0;
}

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
	const char *output_folder)			// output folder
{
	timeval start_time, end_time;

	// -------------------------------------------------------------------------
	//  read data set, query set and truth set
	// -------------------------------------------------------------------------
	gettimeofday(&start_time, NULL);
	float **data = new float*[n];
	for (int i = 0; i < n; ++i) data[i] = new float[d];
	if (read_data(n, d, data_set, data) == 1) {
		printf("Reading Dataset Error!\n");
		exit(1);
	}

	float **query = new float*[qn];
	for (int i = 0; i < qn; ++i) query[i] = new float[d];
	if (read_data(qn, d, query_set, query) == 1) {
		printf("Reading Query Set Error!\n");
		exit(1);
	}

	float **R = new float*[qn];
	for (int i = 0; i < qn; ++i) R[i] = new float[MAXK];
	if (read_ground_truth(qn, truth_set, R) == 1) {
		printf("Reading Truth Set Error!\n");
		exit(1);
	}

	gettimeofday(&end_time, NULL);
	float read_file_time = end_time.tv_sec - start_time.tv_sec + 
		(end_time.tv_usec - start_time.tv_usec) / 1000000.0f;
	printf("Read Data, Query and Ground Truth: %f Seconds\n\n", read_file_time);

	// -------------------------------------------------------------------------
	//  indexing
	// -------------------------------------------------------------------------
	gettimeofday(&start_time, NULL);
	char index_path[200];
	strcpy(index_path, output_folder);
	strcat(index_path, "qalsh_plus_mem/");

	QALSH_Plus *lsh = new QALSH_Plus();
	lsh->build(n, d, kd_leaf_size, L, M, p, zeta, ratio, (const float **) data, 
		index_path);

	gettimeofday(&end_time, NULL);
	float indexing_time = end_time.tv_sec - start_time.tv_sec + 
		(end_time.tv_usec - start_time.tv_usec) / 1000000.0f;
	printf("Indexing Time: %f Seconds\n\n", indexing_time);

	// -------------------------------------------------------------------------
	//  c-k-ANN search
	// -------------------------------------------------------------------------
	char output_set[200];
	sprintf(output_set, "%sqalsh_plus_mem_nb=%d.out", output_folder, nb);

	FILE *fp = fopen(output_set, "w");
	if (!fp) {
		printf("Could not create %s.\n", output_set);
		return 1;
	}

	int kNNs[] = { 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
	int maxRound = 11;
	int top_k = -1;

	float runtime = -1.0f;
	float overall_ratio = -1.0f;

	printf("c-k-ANN Search by QALSH_Plus:\n");
	printf("  Top-k\t\tRatio\t\tTime (ms)\n");
	for (int round = 0; round < maxRound; ++round) {
		gettimeofday(&start_time, NULL);
		top_k = kNNs[round];
		overall_ratio = 0.0f;

		MinK_List *list = new MinK_List(top_k);
		for (int i = 0; i < qn; ++i) {
			list->reset();
			lsh->knn(top_k, nb, query[i], list);

			float ratio = 0.0f;
			for (int j = 0; j < top_k; ++j) {
				ratio += list->ith_key(j) / R[i][j];
			}
			overall_ratio += ratio / top_k;
		}
		delete list; list = NULL;
		gettimeofday(&end_time, NULL);
		runtime = end_time.tv_sec - start_time.tv_sec + (end_time.tv_usec - 
			start_time.tv_usec) / 1000000.0f;

		overall_ratio = overall_ratio / qn;
		runtime = (runtime * 1000.0f) / qn;

		printf("  %3d\t\t%.4f\t\t%.2f\n", top_k, overall_ratio, runtime);
		fprintf(fp, "%d\t%f\t%f\n", top_k, overall_ratio, runtime);
	}
	printf("\n");
	fclose(fp);

	// -------------------------------------------------------------------------
	//  release space
	// -------------------------------------------------------------------------
	delete lsh; lsh = NULL;
	for (int i = 0; i < n; ++i) {
		delete[] data[i]; data[i] = NULL;
	}
	delete[] data; data = NULL;
	
	for (int i = 0; i < qn; ++i) {
		delete[] query[i]; query[i] = NULL;
		delete[] R[i]; R[i] = NULL;
	}
	delete[] query; query = NULL;
	delete[] R; R = NULL;
	
	return 0;
}

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
	const char *output_folder)			// output folder
{
	timeval start_time, end_time;

	// -------------------------------------------------------------------------
	//  read data set, query set and truth set
	// -------------------------------------------------------------------------
	gettimeofday(&start_time, NULL);
	float **data = new float*[n];
	for (int i = 0; i < n; ++i) data[i] = new float[d];
	if (read_data(n, d, data_set, data) == 1) {
		printf("Reading Dataset Error!\n");
		exit(1);
	}

	float **query = new float*[qn];
	for (int i = 0; i < qn; ++i) query[i] = new float[d];
	if (read_data(qn, d, query_set, query) == 1) {
		printf("Reading Query Set Error!\n");
		exit(1);
	}

	float **R = new float*[qn];
	for (int i = 0; i < qn; ++i) R[i] = new float[MAXK];
	if (read_ground_truth(qn, truth_set, R) == 1) {
		printf("Reading Truth Set Error!\n");
		exit(1);
	}

	gettimeofday(&end_time, NULL);
	float read_file_time = end_time.tv_sec - start_time.tv_sec + 
		(end_time.tv_usec - start_time.tv_usec) / 1000000.0f;
	printf("Read Data, Query and Ground Truth: %f Seconds\n\n", read_file_time);
	
	// -------------------------------------------------------------------------
	//  indexing
	// -------------------------------------------------------------------------
	gettimeofday(&start_time, NULL);
	char index_path[200];
	strcpy(index_path, output_folder);
	strcat(index_path, "qalsh_mem/");

	QALSH *lsh = new QALSH();
	lsh->build(n, d, p, zeta, ratio, (const float**) data, index_path);

	gettimeofday(&end_time, NULL);
	float indexing_time = end_time.tv_sec - start_time.tv_sec + 
		(end_time.tv_usec - start_time.tv_usec) / 1000000.0f;
	printf("Indexing Time: %f Seconds\n\n", indexing_time);

	// -------------------------------------------------------------------------
	//  c-k-ANN search
	// -------------------------------------------------------------------------
	char output_set[200];
	sprintf(output_set, "%sqalsh_mem.out", output_folder);

	FILE *fp = fopen(output_set, "w");
	if (!fp) {
		printf("Could not create %s.\n", output_set);
		return 1;
	}

	int kNNs[] = { 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
	int maxRound = 11;
	int top_k = -1;

	float runtime = -1.0f;
	float overall_ratio = -1.0f;

	printf("c-k-ANN Search by QALSH:\n");
	printf("  Top-k\t\tRatio\t\tTime (ms)\n");
	for (int round = 0; round < maxRound; ++round) {
		gettimeofday(&start_time, NULL);
		top_k = kNNs[round];
		overall_ratio = 0.0f;

		MinK_List *list = new MinK_List(top_k);
		for (int i = 0; i < qn; ++i) {
			list->reset();
			lsh->knn(top_k, query[i], list);

			float ratio = 0.0f;
			for (int j = 0; j < top_k; ++j) {
				ratio += list->ith_key(j) / R[i][j];
			}
			overall_ratio += ratio / top_k;
		}
		delete list; list = NULL;
		gettimeofday(&end_time, NULL);
		runtime = end_time.tv_sec - start_time.tv_sec + (end_time.tv_usec - 
			start_time.tv_usec) / 1000000.0f;

		overall_ratio = overall_ratio / qn;
		runtime = (runtime * 1000.0f) / qn;

		printf("  %3d\t\t%.4f\t\t%.2f\n", top_k, overall_ratio, runtime);
		fprintf(fp, "%d\t%f\t%f\n", top_k, overall_ratio, runtime);
	}
	printf("\n");
	fclose(fp);

	// -------------------------------------------------------------------------
	//  release space
	// -------------------------------------------------------------------------
	delete lsh; lsh = NULL;
	for (int i = 0; i < n; ++i) {
		delete[] data[i]; data[i] = NULL;
	}
	delete[] data; data = NULL;
	
	for (int i = 0; i < qn; ++i) {
		delete[] query[i]; query[i] = NULL;
		delete[] R[i]; R[i] = NULL;
	}
	delete[] query; query = NULL;
	delete[] R; R = NULL;

	return 0;
}

// -----------------------------------------------------------------------------
int linear_scan(					// k-NN search of linear scan method
	int   n,							// number of data  objects
	int   qn,							// number of query objects
	int   d,							// dimensionality
	float p,							// the p value of Lp norm, p in (0,2]
	const char *data_set,				// address of data  set
	const char *query_set,				// address of query set
	const char *truth_set,				// address of truth set
	const char *output_folder)			// output folder
{
	timeval start_time, end_time;

	// -------------------------------------------------------------------------
	//  read data set, query set and truth set
	// -------------------------------------------------------------------------
	gettimeofday(&start_time, NULL);
	float **data = new float*[n];
	for (int i = 0; i < n; ++i) data[i] = new float[d];
	if (read_data(n, d, data_set, data) == 1) {
		printf("Reading Dataset Error!\n");
		exit(1);
	}

	float **query = new float*[qn];
	for (int i = 0; i < qn; ++i) query[i] = new float[d];
	if (read_data(qn, d, query_set, query) == 1) {
		printf("Reading Query Set Error!\n");
		exit(1);
	}

	float **R = new float*[qn];
	for (int i = 0; i < qn; ++i) R[i] = new float[MAXK];
	if (read_ground_truth(qn, truth_set, R) == 1) {
		printf("Reading Truth Set Error!\n");
		exit(1);
	}

	gettimeofday(&end_time, NULL);
	float read_file_time = end_time.tv_sec - start_time.tv_sec + 
		(end_time.tv_usec - start_time.tv_usec) / 1000000.0f;
	printf("Read Data, Query and Ground Truth: %f Seconds\n\n", read_file_time);

	// -------------------------------------------------------------------------
	//  k-NN search
	// -------------------------------------------------------------------------
	char output_set[200];
	sprintf(output_set, "%slinear_mem.out", output_folder);

	FILE *fp = fopen(output_set, "w");
	if (!fp) {
		printf("Could not create %s.\n", output_set);
		return 1;
	}

	int kNNs[] = { 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
	int maxRound = 11;
	int top_k = -1;

	float runtime = -1.0f;
	float overall_ratio = -1.0f;

	printf("k-NN Search by Linear Scan:\n");
	printf("  Top-k\t\tRatio\t\tTime (ms)\n");
	for (int round = 0; round < maxRound; ++round) {
		gettimeofday(&start_time, NULL);
		top_k = kNNs[round];
		overall_ratio = 0.0f;

		MinK_List *list = new MinK_List(top_k);
		for (int i = 0; i < qn; ++i) {
			list->reset();
			for (int j = 0; j < n; ++j) {
				float dist = calc_lp_dist(d, p, data[j], query[i]);
				list->insert(dist, j);
			}

			float ratio = 0.0f;
			for (int j = 0; j < top_k; ++j) {
				ratio += list->ith_key(j) / R[i][j];
			}
			overall_ratio += ratio / top_k;
		}
		delete list; list = NULL;
		gettimeofday(&end_time, NULL);
		runtime = end_time.tv_sec - start_time.tv_sec + (end_time.tv_usec - 
			start_time.tv_usec) / 1000000.0f;

		overall_ratio = overall_ratio / qn;
		runtime = (runtime * 1000.0f) / qn;

		printf("  %3d\t\t%.4f\t\t%.2f\n", top_k, overall_ratio, runtime);
		fprintf(fp, "%d\t%f\t%f\n", top_k, overall_ratio, runtime);
	}
	printf("\n");
	fclose(fp);

	// -------------------------------------------------------------------------
	//  release space
	// -------------------------------------------------------------------------
	for (int i = 0; i < n; ++i) {
		delete[] data[i]; data[i] = NULL;
	}
	delete[] data; data = NULL;
	
	for (int i = 0; i < qn; ++i) {
		delete[] query[i]; query[i] = NULL;
		delete[] R[i]; R[i] = NULL;
	}
	delete[] query; query = NULL;
	delete[] R; R = NULL;
	
	return 0;
}