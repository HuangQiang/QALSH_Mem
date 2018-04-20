#ifndef __DEF_H
#define __DEF_H

// -----------------------------------------------------------------------------
//  Macros
// -----------------------------------------------------------------------------
#define MIN(a, b)	(((a) < (b)) ? (a) : (b))
#define MAX(a, b)	(((a) > (b)) ? (a) : (b))
#define POW(x)		((x) * (x))
#define SUM(x, y)	((x) + (y))
#define DIFF(x, y)	((y) - (x))
#define SWAP(x, y)	{int tmp=x; x=y; y=tmp;}

#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2

// -----------------------------------------------------------------------------
//  Constants
// -----------------------------------------------------------------------------
const float MAXREAL  = 3.402823466e+38F;
const float MINREAL  = -MAXREAL;
const int   MAXINT   = 2147483647;
const int   MININT   = -MAXINT;

const int SIZEBOOL   = (int) sizeof(bool);
const int SIZEINT    = (int) sizeof(int);
const int SIZECHAR   = (int) sizeof(char);
const int SIZEFLOAT  = (int) sizeof(float);

const float E  = 2.7182818F;
const float PI = 3.141592654F;
const float FLOATZERO = 1e-6F;
const float ANGLE = PI / 8.0f;

const int MAXK = 100;				// max k-all-pairs and k-closest-pairs
const int SCAN_SIZE = 128;			// size of one scan
const int CANDIDATES = 100;			// number of candidates for QALSH
const int MAX_BLOCK_NUM = 10000;    // maximum number of objects in a block

#endif // __DEF_H
