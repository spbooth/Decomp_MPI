#include <stdlib.h>
#include <string.h>
#include "single_decomp.h"


int single_rank_offset(DimDecomp d, int global){
  SingleDecomp sd = (SingleDecomp)d;
  return sd->rank;
}
int single_type_offset(DimDecomp d,int global){
  return global;
}
DimDecomp single_clone(DimDecomp d){
	DimDecomp result = calloc(sizeof(struct single_decomp),1);
	return memmove(result,d,sizeof(struct single_decomp));
}
DimDecomp makeSingleDecomp(int coord_size,int rank,int proc_size){
  SingleDecomp result = (SingleDecomp) calloc(sizeof(struct single_decomp),1);

  result->parent.coord_size=coord_size;
  result->parent.proc_size=proc_size;
  result->parent.rank_offset=single_rank_offset;
  result->parent.type_offset=single_type_offset;
  result->parent.destroy=NULL;
  result->parent.cloner=single_clone;
  result->parent.type="single";
  result->rank=rank;

  return (DimDecomp) result;
}
