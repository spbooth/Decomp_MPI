#include "reshape.h"
/** @file interleave_decomp.h
 */

/** A @ref DimDecomp where the data is distributed as interleaved
 * sub-sequences. Each interleaved sequence maps to the same processor
 *
 * x = y + A.z
 *
 * z is local.
 */
struct interleave_decomp{
  struct dim_decomp parent; //!< super-type
  DimDecomp nested;  //!< decomposition of non-local y sub-dimension
  int A; //!< number of sequences
  int B; //!< length of each sequences.
};
typedef struct interleave_decomp *InterleaveDecomp;

int interleave_rank_offset(DimDecomp d, int global);
int interleave_type_offset(DimDecomp d, int global);
void print_interleave(FILE *out,DimDecomp d);
DimDecomp clone_interleave(DimDecomp d);

/** make a @ref DimDecomp where the data is distributed in blocks. The blocksize * is (length+proc-1)/proc.
 * @param B number of interleaves sequences
 * @param P length of each sequence
 * @param proc number of processors to distribute over.
 */
DimDecomp makeInterleaveDecomp(DimDecomp nested, int B);
