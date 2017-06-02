#include <stdlib.h>
#include <string.h>
#include "local_decomp.h"


int local_rank_offset(DimDecomp d, int global){
  return 0;
}
int local_type_offset(DimDecomp d,int global){
  return global;
}
DimDecomp local_clone(DimDecomp d){
	LocalDecomp result = (LocalDecomp) calloc(sizeof(struct local_decomp),1);
	return memmove(result,d,sizeof(struct local_decomp));
}
DimDecomp makeLocalDecomp(int len){
  LocalDecomp result = (LocalDecomp) calloc(sizeof(struct local_decomp),1);

  result->parent.coord_size=len;
  result->parent.proc_size=1;
  result->parent.rank_offset=local_rank_offset;
  result->parent.type_offset=local_type_offset;
  result->parent.type="local";
  result->parent.destroy=NULL;
  result->parent.cloner=local_clone;

  return (DimDecomp) result;
}
