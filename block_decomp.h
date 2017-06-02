#include "reshape.h"
/** @file block_decomp.h
 */

/** A @ref DimDecomp where the data is distributed in blocks.
 */
struct block_decomp{
  struct dim_decomp parent; //!< super-type
  int blocksize; //!< maximum size of data block.
};
typedef struct block_decomp *BlockDecomp;

int block_rank_offset(DimDecomp d, int global);
int block_type_offset(DimDecomp d, int global);

/** make a @ref DimDecomp where the data is distributed in blocks. The blocksize * is (length+proc-1)/proc.
 * @param len number of global coordinates.
 * @param proc number of processors to distribute over.
 */
DimDecomp makeBlockDecomp(int len, int proc);
/** Make a BlockDecomp where the blocksize is a multiple of a known factor.
 *
 * @param len length of axis
 * @param factor required factor of block size (must also factor len)
 * @param procs number of procs
 *
 */
DimDecomp makeBlockDecompWithFactor(int len,int factor,int procs);
