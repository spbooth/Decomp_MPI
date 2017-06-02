#include "reshape.h"

struct rotate_decomp{
  struct dim_decomp parent;
  int rotate;
  DimDecomp nested; 
};
typedef struct rotate_decomp *RotateDecomp;

int rotate_rank_offset(DimDecomp d, int global);
int rotate_type_offset(DimDecomp d, int global);
DimDecomp clone_rotate(DimDecomp d);

DimDecomp makeRotatedDecomp(int rotate,DimDecomp nested);
