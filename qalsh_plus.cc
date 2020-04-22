#include "qalsh_plus.h"

// -----------------------------------------------------------------------------
QALSH_PLUS::QALSH_PLUS(				// constructor
	int   n,							// cardinality
	int   d,							// dimensionality
	int   leaf,							// leaf size of kd-tree
	int   L,							// number of projection
	int   M,							// number of candidates for each proj
	float p,							// l_p distance
	float zeta,							// symmetric factor of p-stable distr.
	float ratio,						// approximation ratio
	const float **data)					// data objects
{
	// -------------------------------------------------------------------------
	//  init parameters
	// -------------------------------------------------------------------------
	n_pts_ = n;
	dim_   = d;
	L_     = L;
	M_     = M;
	p_     = p;
	data_  = data;

	// -------------------------------------------------------------------------
	//  kd-tree partition
	// -------------------------------------------------------------------------
	g_memory += SIZEINT * n;
	new_order_id_ = new int[n];
	std::vector<int> block_size;
	kd_tree_partition(leaf, block_size, new_order_id_);

	num_blocks_ = (int) block_size.size();

	// -------------------------------------------------------------------------
	//  get sample_id and build qalsh for each block
	// -------------------------------------------------------------------------
	int sample_size  = L_ * M_;
	int sample_n_pts = num_blocks_ * sample_size;
	int *sample_id   = new int[sample_n_pts];

	int start = 0;
	int count = 0;
	for (int i = 0; i < num_blocks_; ++i) {
		// ---------------------------------------------------------------------
		//  select sample data from each blcok 
		// ---------------------------------------------------------------------
		int block_n = block_size[i]; assert(block_n > sample_size);
		drusilla_select(block_n, new_order_id_ + start, sample_id + count);

		// ---------------------------------------------------------------------
		//  build qalsh for each blcok 
		// ---------------------------------------------------------------------
		Block *block = new Block();
		block->n_pts_ = block_n;
		block->index_ = new_order_id_ + start;
		block->lsh_   = new QALSH(block_n, d, p, zeta, ratio);

		int m = block->lsh_->m_;
		for (int j = 0; j < block_n; ++j) {
			int id = block->index_[j];
			for (int u = 0; u < m; ++u) {
				float val = block->lsh_->calc_hash_value(u, data[id]);
				block->lsh_->tables_[u][j].id_  = j;
				block->lsh_->tables_[u][j].key_ = val;
			}
		}
		for (int j = 0; j < m; ++j) {
			qsort(block->lsh_->tables_[j], block_n, sizeof(Result), ResultComp);
		}
		blocks_.push_back(block);

		// ---------------------------------------------------------------------
		//  update parameters
		// ---------------------------------------------------------------------
		start += block_n;
		count += sample_size;
	}
	assert(start == n);
	assert(count == sample_n_pts);

	// -------------------------------------------------------------------------
	//  build qalsh for sample data
	// -------------------------------------------------------------------------
	g_memory += SIZEFLOAT * sample_n_pts * d;
	sample_data_ = new float*[sample_n_pts];
	for (int i = 0; i < sample_n_pts; ++i) {
		sample_data_[i] = new float[d];

		const float *one_data = data[sample_id[i]];
		for (int j = 0; j < d; ++j) {
			sample_data_[i][j] = one_data[j];
		}
	}
	lsh_ = new QALSH(sample_n_pts, d, p, zeta, ratio, (const float**) sample_data_);

	// -------------------------------------------------------------------------
	//  release space
	// -------------------------------------------------------------------------
	delete[] sample_id; sample_id = NULL;
}

// -----------------------------------------------------------------------------
void QALSH_PLUS::kd_tree_partition(	// kd-tree partition
	int leaf,							// leaf size of kd-tree
	std::vector<int> &block_size,		// block size (return)
	int *new_order_id)					// new order id (return)
{
	KD_Tree *tree = new KD_Tree(n_pts_, dim_, leaf, data_);
	tree->traversal(block_size, new_order_id);

	delete tree; tree = NULL;
}

// -----------------------------------------------------------------------------
void QALSH_PLUS::drusilla_select(	// drusilla select
	int   n,							// number of objects
	const int *new_order_id,			// new order data id
	int   *sample_id)					// sample data id (return)
{
	// -------------------------------------------------------------------------
	//  calc shift data
	// -------------------------------------------------------------------------
	int   max_id   = -1;
	float max_norm = MINREAL;
	float *norm = new float[n];
	float **shift_data = new float*[n];
	for (int i = 0; i < n; ++i) shift_data[i] = new float[dim_];
	
	calc_shift_data(n, new_order_id, max_id, max_norm, norm, shift_data);

	// -------------------------------------------------------------------------
	//  drusilla select
	// -------------------------------------------------------------------------
	float  *proj  = new float[dim_];
	Result *score = new Result[n];
	bool   *close_angle = new bool[n];

	for (int i = 0; i < L_; ++i) {
		// ---------------------------------------------------------------------
		//  select the projection vector with largest norm and normalize it
		// ---------------------------------------------------------------------
		for (int j = 0; j < dim_; ++j) {
			proj[j] = shift_data[max_id][j] / norm[max_id];
		}

		// ---------------------------------------------------------------------
		//  calculate offsets and distortions
		// ---------------------------------------------------------------------
		for (int j = 0; j < n; ++j) {
			score[j].id_ = j;
			close_angle[j] = false;

			if (norm[j] > 0.0f) {
				float offset = calc_inner_product(dim_, shift_data[j], proj);

				float distortion = 0.0f;
				for (int u = 0; u < dim_; ++u) {
					distortion += SQR(shift_data[j][u] - offset * proj[u]);
				}

				score[j].key_ = offset * offset - distortion;
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

		// ---------------------------------------------------------------------
		//  collect the objects that are well-represented by this projection
		// ---------------------------------------------------------------------
		qsort(score, n, sizeof(Result), ResultCompDesc);
		for (int j = 0; j < M_; ++j) {
			int id = score[j].id_;
			sample_id[i * M_ + j] = new_order_id[id];
			norm[id] = -1.0f;
		}

		// ---------------------------------------------------------------------
		//  find the next largest norm and the corresponding object
		// ---------------------------------------------------------------------
		max_id   = -1;
		max_norm = MINREAL;
		for (int j = 0; j < n; ++j) {
			if (norm[j] > 0.0f && close_angle[j]) { norm[j] = 0.0f; }
			if (norm[j] > max_norm) { max_norm = norm[j]; max_id = j; }
		}
	}
	// -------------------------------------------------------------------------
	//  release space
	// -------------------------------------------------------------------------
	delete[] norm;        norm        = NULL;
	delete[] close_angle; close_angle = NULL;
	delete[] proj;        proj        = NULL;
	delete[] score;       score       = NULL;

	for (int i = 0; i < n; ++i) {
		delete[] shift_data[i]; shift_data[i] = NULL;
	}
	delete[] shift_data; shift_data = NULL;
}

// -----------------------------------------------------------------------------
void QALSH_PLUS::calc_shift_data(	// calculate shift data objects
	int   n,							// number of data objects
	const int *new_order_id,			// new order data id
	int   &max_id,						// data id with max l2-norm (return)
	float &max_norm,					// max l2-norm (return)
	float *norm,						// l2-norm of shift data (return)
	float **shift_data) 				// shift data (return)
{
	// -------------------------------------------------------------------------
	//  calculate the centroid of data objects
	// -------------------------------------------------------------------------
	float *centroid = new float[dim_];
	memset(centroid, 0.0f, dim_ * SIZEFLOAT);
	for (int i = 0; i < n; ++i) {
		int id = new_order_id[i];
		for (int j = 0; j < dim_; ++j) {
			centroid[j] += data_[id][j];
		}
	}
	for (int i = 0; i < dim_; ++i) centroid[i] /= n;

	// -------------------------------------------------------------------------
	//  calc shift data and their l2-norm and find max l2-norm and its id
	// -------------------------------------------------------------------------
	max_id   = -1;
	max_norm = MINREAL;

	for (int i = 0; i < n; ++i) {
		int id = new_order_id[i];

		norm[i] = 0.0f;
		for (int j = 0; j < dim_; ++j) {
			float tmp = data_[id][j] - centroid[j];
			shift_data[i][j] = tmp;
			norm[i] += SQR(tmp);
		}
		norm[i] = sqrt(norm[i]);

		if (norm[i] > max_norm) { max_norm = norm[i]; max_id = i; }
	}
	delete[] centroid; centroid = NULL;
}

// -----------------------------------------------------------------------------
QALSH_PLUS::~QALSH_PLUS()			// destructor
{
	delete lsh_; lsh_ = NULL;

	int sample_n_pts = M_ * L_ * num_blocks_;
	for (int i = 0; i < sample_n_pts; ++i) {
		delete[] sample_data_[i]; sample_data_[i] = NULL;
	}
	delete[] sample_data_; sample_data_ = NULL;
	g_memory -= SIZEFLOAT * sample_n_pts * dim_;

	delete[] new_order_id_; new_order_id_ = NULL;
	g_memory -= SIZEINT * n_pts_;

	for (int i = 0; i < num_blocks_; ++i) {
		delete blocks_[i]; blocks_[i] = NULL;
	}
	blocks_.clear(); blocks_.shrink_to_fit();
}

// -----------------------------------------------------------------------------
void QALSH_PLUS::display()			// display parameters
{
	printf("Parameters of QALSH+:\n");
	printf("    n          = %d\n", n_pts_);
	printf("    d          = %d\n", dim_);
	printf("    L          = %d\n", L_);
	printf("    M          = %d\n", M_);
	printf("    p          = %.1f\n", p_);
	printf("    num_blocks = %d\n", num_blocks_);
	printf("\n");
}

// -----------------------------------------------------------------------------
int QALSH_PLUS::knn(				// c-k-ANN search
	int   top_k,						// top-k value
	int   nb,							// number of blocks for search
	const float *query,					// input query objects
	MinK_List *list)					// k-NN results
{
	assert(nb > 0 && nb <= num_blocks_);

	// -------------------------------------------------------------------------
	//  use sample data to determine block_order for k-NN search
	// -------------------------------------------------------------------------
	std::vector<int> block_order;
	MinK_List *cand_list = new MinK_List(MAXK);
	lsh_->knn(MAXK, query, cand_list);
	
	get_block_order(nb, cand_list, block_order);
	
	// -------------------------------------------------------------------------
	//  use <nb> blocks for k-NN search
	// -------------------------------------------------------------------------
	std::vector<int> cand;
	float radius = MAXREAL;
	for (size_t i = 0; i < block_order.size(); ++i) {
		int   bid = block_order[i];
		Block *block = blocks_[bid];

		// find candidates by qalsh for this block
		cand.clear();
		block->lsh_->knn(top_k, radius, query, cand);

		// check candidates
		for (size_t j = 0; j < cand.size(); ++j) {
			int   id   = block->index_[cand[j]];
			float dist = calc_lp_dist(dim_, p_, radius, data_[id], query);
			radius = list->insert(dist, id);
		}
	}
	return 0;
}

// -----------------------------------------------------------------------------
void QALSH_PLUS::get_block_order(	// get block order
	int nb,								// number of blocks for search
	MinK_List *cand,					// candidates
	std::vector<int> &block_order)		// block order (return)
{
	// -------------------------------------------------------------------------
	//  init the counter of each block
	// -------------------------------------------------------------------------
	Result *pair = new Result[num_blocks_];
	for (int i = 0; i < num_blocks_; ++i) {
		pair[i].id_  = i;
		pair[i].key_ = 0.0f;
	}

	// -------------------------------------------------------------------------
	//  select the first <nb> blocks with largest counters
	// -------------------------------------------------------------------------
	int sample_size  = L_ * M_;
	for (int i = 0; i < cand->size(); ++i) {
		int block_id = cand->ith_id(i) / sample_size;
		pair[block_id].key_ += 1.0f;
	}

	qsort(pair, num_blocks_, sizeof(Result), ResultCompDesc);
	for (int i = 0; i < nb; ++i) {
		// if (fabs(pair[i].key_) < FLOATZERO) break;
		block_order.push_back(pair[i].id_);
	}
	delete[] pair; pair = NULL;
}
