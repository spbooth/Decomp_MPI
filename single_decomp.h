/** @file single_decomp.h
 */
#include "reshape.h"


/** A decomposition where all the data is on a single rank.
 */
struct single_decomp{
  struct dim_decomp parent; //!< super-type
  int rank; //!< rank where data lives.
};
typedef struct single_decomp *SingleDecomp;

int single_rank_offset(DimDecomp d, int global);
int single_type_offset(DimDecomp d, int global);
DimDecomp single_clone(DimDecomp d);

/** make a @ref DimDecomp where all the data lives on a single rank.
 * @param coord_size number of global coordinates
 * @param rank The processor rank (in this dimension) where all the data lives.
 * @param size The number of processors in this dimension.

 */
DimDecomp makeSingleDecomp(int coord_size, int rank,int size);
