/** @file local_decomp.h
 */
#include "reshape.h"
/** A @ref DimDecomp that represents a non distributed data axis
 */
struct local_decomp{
  struct dim_decomp parent;
};
typedef struct local_decomp *LocalDecomp;

int local_rank_offset(DimDecomp d, int global);
int local_type_offset(DimDecomp d, int global);
DimDecomp local_clone(DimDecomp d);

/** make a DimDecomp for a local data axis.
 * @param len number of global goordinates
 */
DimDecomp makeLocalDecomp(int len);
