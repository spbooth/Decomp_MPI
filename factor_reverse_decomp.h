#include "reshape.h"
/** @file factor_reverse_decomp.h
 */

/** A  @ref DimDecomp which reverses the significance
 * of two factors of the coordinate.
 * ie It extracts Q interleaved sequences of length P
 * and turns them into Q contiguous blocks of length P
 */
struct factor_reverse_decomp{
  struct dim_decomp parent;
  int A;
  int B;
  DimDecomp nested; 
};
typedef struct factor_reverse_decomp *FactorReverseDecomp;

int factor_reverse_rank_offset(DimDecomp d, int global);
int factor_reverse_type_offset(DimDecomp d, int global);
DimDecomp clone_factor_reverse(DimDecomp d);

DimDecomp makeFactorReverseDecomp(int A,int B,DimDecomp nested);

AxisDecomp makeFactorReverseAxisDecomp(int A, int B, AxisDecomp orig);
