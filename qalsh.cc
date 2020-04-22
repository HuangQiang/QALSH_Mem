#include "qalsh.h"

// -----------------------------------------------------------------------------
QALSH::QALSH(						// constructor
	int   n,							// cardinality
	int   d,							// dimensionality
	float p,							// l_p distance
	float zeta,							// symmetric factor of p-stable distr.
	float ratio)						// approximation ratio
{
	// -------------------------------------------------------------------------
	//  init parameters
	// -------------------------------------------------------------------------
	n_pts_ = n;
	dim_   = d;
	p_     = p;
	zeta_  = zeta;
	ratio_ = ratio;
	data_  = NULL;

	init();
}

// -----------------------------------------------------------------------------
QALSH::QALSH(						// constructor
	int   n,							// cardinality
	int   d,							// dimensionality
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
	p_     = p;
	zeta_  = zeta;
	ratio_ = ratio;
	data_  = data;

	init();

	// -------------------------------------------------------------------------
	//  bulkloading
	// -------------------------------------------------------------------------
	for (int i = 0; i < m_; ++i) {
		for (int j = 0; j < n_pts_; ++j) {
			tables_[i][j].id_  = j;
			tables_[i][j].key_ = calc_hash_value(i, data[j]);
		}
		qsort(tables_[i], n_pts_, sizeof(Result), ResultComp);
	}
}

// -----------------------------------------------------------------------------
void QALSH::init()					// basic initialization
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

	float w0 = (ratio_ - 1.0f) / log(sqrt(ratio_));
	float w1 = 2.0f * sqrt(ratio_);
	float w2 = sqrt((8.0f * SQR(ratio_) * log(ratio_)) / (SQR(ratio_) - 1.0f));
	float p1 = -1.0f, p2 = -1.0f;

	if (fabs(p_ - 0.5f) < FLOATZERO) {
		w_ = w0;
		p1 = calc_l0_prob(w_ / 2.0f);
		p2 = calc_l0_prob(w_ / (2.0f * ratio_));
	}
	else if (fabs(p_ - 1.0f) < FLOATZERO) {
		w_ = w1;
		p1 = calc_l1_prob(w_ / 2.0f);
		p2 = calc_l1_prob(w_ / (2.0f * ratio_));
	}
	else if (fabs(p_ - 2.0f) < FLOATZERO) {
		w_ = w2;
		p1 = calc_l2_prob(w_ / 2.0f);
		p2 = calc_l2_prob(w_ / (2.0f * ratio_));
	}
	else {
		if (fabs(p_-0.8f) < FLOATZERO) w_ = 2.503f;
		else if (fabs(p_-1.2f) < FLOATZERO) w_ = 3.151f;
		else if (fabs(p_-1.5f) < FLOATZERO) w_ = 3.465f;
		else w_ = (w2 - w1) * p_ + (2.0f * w1 - w2);

		new_stable_prob(p_, zeta_, ratio_, 1.0f, w_, 1000000, p1, p2);
	}

	float para1 = sqrt(log(2.0f / beta));
	float para2 = sqrt(log(1.0f / delta));
	float para3 = 2.0f * (p1 - p2) * (p1 - p2);
	float eta   = para1 / para2;
	float alpha = (eta * p1 + p2) / (1.0f + eta);

	m_ = (int) ceil((para1 + para2) * (para1 + para2) / para3);
	l_ = (int) ceil(alpha * m_);

	// -------------------------------------------------------------------------
	//  generate hash functions <a_>
	// -------------------------------------------------------------------------
	g_memory += SIZEFLOAT * m_ * dim_;
	a_ = new float*[m_];
	for (int i = 0; i < m_; ++i) {
		a_[i] = new float[dim_];
		for (int j = 0; j < dim_; ++j) {
			if (fabs(p_-0.5f) < FLOATZERO) a_[i][j] = levy(1.0f, 0.0f);
			else if (fabs(p_-1.0f) < FLOATZERO) a_[i][j]= cauchy(1.0f, 0.0f);
			else if (fabs(p_-2.0f) < FLOATZERO) a_[i][j] = gaussian(0.0f, 1.0f);
			else a_[i][j] = p_stable(p_, zeta_, 1.0f, 0.0f);
		}
	}

	// -------------------------------------------------------------------------
	//  allocate space for hash tables <tables_>
	// -------------------------------------------------------------------------
	g_memory += sizeof(Result) * m_ * n_pts_;
	tables_ = new Result*[m_];
	for (int i = 0; i < m_; ++i) tables_[i] = new Result[n_pts_];
}

// -----------------------------------------------------------------------------
inline float QALSH::calc_l0_prob(	// calc prob of L1/2 dist
	float x)							// x = w / (2.0 * r)
{
	return new_levy_prob(x);
}

// -----------------------------------------------------------------------------
inline float QALSH::calc_l1_prob(	// calc prob of L1 dist
	float x)							// x = w / (2.0 * r)
{
	return new_cauchy_prob(x);
}

// -----------------------------------------------------------------------------
inline float QALSH::calc_l2_prob(	// calc prob of L2 dist
	float x)							// x = w / (2.0 * r)
{
	return new_gaussian_prob(x);
}

// -----------------------------------------------------------------------------
float QALSH::calc_hash_value( 		// calc hash value
	int   tid,							// hash table id
	const float *data)					// one data/query object
{
	return calc_inner_product(dim_, a_[tid], data);
}

// -----------------------------------------------------------------------------
QALSH::~QALSH()						// destructor
{
	for (int i = 0; i < m_; ++i) {
		delete[] a_[i]; a_[i] = NULL;
		delete[] tables_[i]; tables_[i] = NULL;
	}
	delete[] a_; a_ = NULL;
	delete[] tables_; tables_ = NULL;

	g_memory -= SIZEFLOAT * m_ * dim_;
	g_memory -= sizeof(Result) * m_ * n_pts_;
}

// -----------------------------------------------------------------------------
void QALSH::display()				// display parameters
{
	printf("Parameters of QALSH:\n");
	printf("    n     = %d\n",   n_pts_);
	printf("    d     = %d\n",   dim_);
	printf("    p     = %.1f\n", p_);
	printf("    zeta  = %.1f\n", zeta_);
	printf("    ratio = %.1f\n", ratio_);
	printf("    w     = %f\n",   w_);
	printf("    m     = %d\n",   m_);
	printf("    l     = %d\n",   l_);
	printf("\n");
}

// -----------------------------------------------------------------------------
int QALSH::knn(						// k-nn search
	int   top_k,						// top-k value
	const float *query,					// input query object
	MinK_List *list)					// k-NN results (return)
{
	int   *freq    = new int[n_pts_];
	int   *lpos    = new int[m_];
	int   *rpos    = new int[m_];
	bool  *checked = new bool[n_pts_];
	bool  *flag    = new bool[m_];
	float *q_val   = new float[m_];

	// -------------------------------------------------------------------------
	//  init parameters
	// -------------------------------------------------------------------------
	memset(freq,    0,     n_pts_ * SIZEINT);
	memset(checked, false, n_pts_ * SIZEBOOL);

	Result tmp;
	Result *table = NULL;
	for (int i = 0; i < m_; ++i) {
		tmp.key_ = calc_hash_value(i, query);
		q_val[i] = tmp.key_;

		table = tables_[i];
		int pos = std::lower_bound(table, table+n_pts_, tmp, cmp) - table;
		if (pos == 0) { lpos[i] = -1;  rpos[i] = pos; } 
		else { lpos[i] = pos; rpos[i] = pos + 1; }
	}

	// -------------------------------------------------------------------------
	//  c-k-ANN search
	// -------------------------------------------------------------------------
	int candidates = CANDIDATES + top_k - 1; // candidate size
	int cand_cnt   = 0;				// candidate counter
	
	float kdist  = MAXREAL;
	float radius = 1.0f;			// search radius
	float width  = radius * w_ / 2.0f; // bucket width

	while (true) {
		// ---------------------------------------------------------------------
		//  step 1: initialize the stop condition for current round
		// ---------------------------------------------------------------------
		int num_flag = 0;
		memset(flag, true, m_ * SIZEBOOL);

		// ---------------------------------------------------------------------
		//  step 2: (R,c)-NN search
		// ---------------------------------------------------------------------
		while (num_flag < m_) {
			for (int j = 0; j < m_; ++j) {
				if (!flag[j]) continue;

				table = tables_[j];
				float q_v = q_val[j], ldist = -1.0f, rdist = -1.0f;
				// -------------------------------------------------------------
				//  step 2.1: scan the left part of hash table
				// -------------------------------------------------------------
				int cnt = 0;
				int pos = lpos[j];
				while (cnt < SCAN_SIZE) {
					ldist = MAXREAL;
					if (pos >= 0) {
						ldist = fabs(q_v - table[pos].key_);
					}
					else break;
					if (ldist > width) break;

					int id = table[pos].id_;
					if (++freq[id] >= l_ && !checked[id]) {
						checked[id] = true;
						float dist = calc_lp_dist(dim_, p_, kdist, data_[id], query);
						kdist = list->insert(dist, id);

						if (++cand_cnt >= candidates) break;
					}
					--pos; ++cnt;
				}
				if (cand_cnt >= candidates) break;
				lpos[j] = pos;

				// -------------------------------------------------------------
				//  step 2.2: scan the right part of hash table
				// -------------------------------------------------------------
				cnt = 0;
				pos = rpos[j];
				while (cnt < SCAN_SIZE) {
					rdist = MAXREAL;
					if (pos < n_pts_) {
						rdist = fabs(q_v - table[pos].key_);
					}
					else break;
					if (rdist > width) break;

					int id = table[pos].id_;
					if (++freq[id] >= l_ && !checked[id]) {
						checked[id] = true;
						float dist = calc_lp_dist(dim_, p_, kdist, data_[id], query);
						kdist = list->insert(dist, id);

						if (++cand_cnt >= candidates) break;
					}
					++pos; ++cnt;
				}
				if (cand_cnt >= candidates) break;
				rpos[j] = pos;

				// -------------------------------------------------------------
				//  step 2.3: check whether this width is finished scanned
				// -------------------------------------------------------------
				if (ldist > width && rdist > width) {
					flag[j] = false;
					if (++num_flag >= m_) break;
				}
			}
			if (num_flag >= m_ || cand_cnt >= candidates) break;
		}
		// ---------------------------------------------------------------------
		//  step 3: stop conditions t1 and t2
		// ---------------------------------------------------------------------
		if (kdist < ratio_ * radius && cand_cnt >= top_k) break;
		if (cand_cnt >= candidates) break;

		// ---------------------------------------------------------------------
		//  step 4: auto-update radius
		// ---------------------------------------------------------------------
		radius = ratio_ * radius;
		width  = radius * w_ / 2.0f;
	}
	// -------------------------------------------------------------------------
	//  release space
	// -------------------------------------------------------------------------
	delete[] freq;    freq    = NULL;
	delete[] lpos;    lpos    = NULL;
	delete[] rpos;    rpos    = NULL;
	delete[] checked; checked = NULL;
	delete[] flag;    flag    = NULL;
	delete[] q_val;   q_val   = NULL;

	return 0;
}

// -----------------------------------------------------------------------------
int QALSH::knn(						// k-NN search
	int   top_k,						// top-k value
	float R,							// limited search range
	const float *query,					// input query object
	std::vector<int> &cand)				// NN candidates (return)
{
	int   *freq        = new int[n_pts_];
	int   *lpos        = new int[m_];
	int   *rpos        = new int[m_];
	bool  *checked     = new bool[n_pts_];
	bool  *bucket_flag = new bool[m_];
	bool  *range_flag  = new bool[m_];
	float *q_val       = new float[m_];

	// -------------------------------------------------------------------------
	//  init parameters
	// -------------------------------------------------------------------------
	memset(freq,       0,     n_pts_ * SIZEINT);
	memset(checked,    false, n_pts_ * SIZEBOOL);
	memset(range_flag, true,  m_ * SIZEBOOL);

	Result tmp;
	Result *table = NULL;
	for (int i = 0; i < m_; ++i) {
		tmp.key_ = calc_hash_value(i, query);
		q_val[i] = tmp.key_;

		table = tables_[i];
		int pos = std::lower_bound(table, table+n_pts_, tmp, cmp) - table;
		if (pos <= 0) { lpos[i] = -1; rpos[i] = pos; }
		else { lpos[i] = pos - 1; rpos[i] = pos; }
	}

	// -------------------------------------------------------------------------
	//  k-nn search via dynamic collision counting
	// -------------------------------------------------------------------------
	int   candidates = CANDIDATES + top_k - 1; // candidate size
	int   cand_cnt   = 0;			// number of candidates computation	
	int   num_range  = 0;			// number of search range flag

	float radius = 1.0f;			// search radius
	float width  = radius * w_ / 2.0f;  // bucket width
	float range  = R > MAXREAL-1.0f ? MAXREAL : R * w_ / 2.0f; // search range

	while (true) {
		// ---------------------------------------------------------------------
		//  step 1: initialize the stop condition for current round
		// ---------------------------------------------------------------------
		int num_bucket = 0;
		memset(bucket_flag, true, m_ * SIZEBOOL);

		// ---------------------------------------------------------------------
		//  step 2: (R,c)-NN search
		// ---------------------------------------------------------------------
		while (num_bucket < m_ && num_range < m_) {
			for (int j = 0; j < m_; ++j) {
				if (!bucket_flag[j]) continue;

				table = tables_[j];
				float q_v = q_val[j], ldist = -1.0f, rdist = -1.0f;
				// -------------------------------------------------------------
				//  step 2.1: scan the left part of hash table
				// -------------------------------------------------------------
				int cnt = 0;
				int pos = lpos[j];
				while (cnt < SCAN_SIZE) {
					ldist = MAXREAL;
					if (pos >= 0) {
						ldist = fabs(q_v - table[pos].key_);
					}
					else break;
					if (ldist > width || ldist > range) break;

					int id = table[pos].id_;
					if (++freq[id] >= l_ && !checked[id]) {
						checked[id] = true;
						cand.push_back(id);

						if (++cand_cnt >= candidates) break;
					}
					--pos; ++cnt;
				}
				if (cand_cnt >= candidates) break;
				lpos[j] = pos;

				// -------------------------------------------------------------
				//  step 2.2: scan the right part of hash table
				// -------------------------------------------------------------
				cnt = 0;
				pos = rpos[j];
				while (cnt < SCAN_SIZE) {
					rdist = MAXREAL;
					if (pos < n_pts_) {
						rdist = fabs(q_v - table[pos].key_);
					}
					else break;
					if (rdist > width || rdist > range) break;

					int id = table[pos].id_;
					if (++freq[id] >= l_ && !checked[id]) {
						checked[id] = true;
						cand.push_back(id);

						if (++cand_cnt >= candidates) break;
					}
					++pos; ++cnt;
				}
				if (cand_cnt >= candidates) break;
				rpos[j] = pos;

				// -------------------------------------------------------------
				//  step 2.3: check whether this width is finished scanned
				// -------------------------------------------------------------
				if (ldist > width && rdist > width) {
					bucket_flag[j] = false;
					if (++num_bucket > m_) break;
				}
				if (ldist > range && rdist > range) {
					if (bucket_flag[j]) {
						bucket_flag[j] = false;
						if (++num_bucket > m_) break;
					}
					if (range_flag[j]) {
						range_flag[j] = false;
						if (++num_range > m_) break;
					}
				}
			}
			if (num_bucket > m_ || num_range > m_ || cand_cnt >= candidates) break;
		}
		// ---------------------------------------------------------------------
		//  step 3: stop conditions
		// ---------------------------------------------------------------------
		if (num_range >= m_ || cand_cnt >= candidates) break;

		// ---------------------------------------------------------------------
		//  step 4: auto-update <radius>
		// ---------------------------------------------------------------------
		radius = radius * ratio_;
		width  = radius * w_ / 2.0f;
	}
	// -------------------------------------------------------------------------
	//  release space
	// -------------------------------------------------------------------------
	delete[] freq;        freq        = NULL;
	delete[] lpos;        lpos        = NULL;
	delete[] rpos;        rpos        = NULL;
	delete[] checked;     checked     = NULL;
	delete[] bucket_flag; bucket_flag = NULL;
	delete[] range_flag;  range_flag  = NULL;
	delete[] q_val;       q_val       = NULL;

	return 0;
}
