#pragma once

#include <vector>

namespace nns {

// -----------------------------------------------------------------------------
//  Macros
// -----------------------------------------------------------------------------
#define MIN(a, b)      (((a) < (b)) ? (a) : (b))
#define MAX(a, b)      (((a) > (b)) ? (a) : (b))
#define SQR(x)         ((x) * (x))
#define SUM(x, y)      ((x) + (y))
#define DIFF(x, y)     ((y) - (x))
#define SWAP(x, y)     { int tmp=x; x=y; y=tmp; }

// -----------------------------------------------------------------------------
//  Constants
// -----------------------------------------------------------------------------
const float MAXREAL    = 3.402823466e+38F;
const float MINREAL    = -MAXREAL;
const int   MAXINT     = 2147483647;
const int   MININT     = -MAXINT;

const float E          = 2.7182818F;
const float PI         = 3.141592654F;
const float FLOATZERO  = 1e-6F;
const float ANGLE      = PI / 8.0f;
const int   SCAN_SIZE  = 128;
const int   CANDIDATES = 100;

const std::vector<int> TOPKs = { 1, 2, 5, 10, 20, 50, 100 };
const int MAXK = TOPKs.back(); 

} // end namespace nns
