#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "factor_reverse_decomp.h"

int factor_reverse(FactorReverseDecomp rd,int global);

int factor_reverse_rank_offset(DimDecomp d, int global){
  FactorReverseDecomp rd = (FactorReverseDecomp)d;
  return rd->nested->rank_offset(rd->nested,factor_reverse(rd,global));
}
int factor_reverse_type_offset(DimDecomp d,int global){
  FactorReverseDecomp rd = (FactorReverseDecomp)d;
  return rd->nested->type_offset(rd->nested,factor_reverse(rd,global));
}
void factor_reverse_destroy(DimDecomp d){
  FactorReverseDecomp rd = (FactorReverseDecomp)d;
  freeDimDecomp(rd->nested);
}
DimDecomp clone_factor_reverse(DimDecomp d){
	FactorReverseDecomp id=(FactorReverseDecomp)d;
	FactorReverseDecomp result = calloc(sizeof(struct factor_reverse_decomp),1);
	memmove(result,id,sizeof(struct factor_reverse_decomp));
	result->nested=cloneDimDecomp(id->nested);
	return (DimDecomp) result;
}
int factor_reverse(FactorReverseDecomp rd,int global){
  // x = y + A.z
  int z = global / rd->A;
  int y = global % rd->A;
  int k = z + rd->B*y;
  //fprintf(stderr,"rotate %d->%d\n",global,i);
  return k;
}

DimDecomp makeFactorReverseDecomp(int A, int B, DimDecomp nested){
  FactorReverseDecomp result = (FactorReverseDecomp) calloc(sizeof(struct factor_reverse_decomp),1);

  result->parent.coord_size=nested->coord_size;
  result->parent.proc_size=nested->proc_size;
  result->parent.rank_offset=factor_reverse_rank_offset;
  result->parent.type_offset=factor_reverse_type_offset;
  result->parent.type="factor_reverse";
  result->parent.destroy=factor_reverse_destroy;
  result->parent.cloner=clone_factor_reverse;
  result->A=A;
  result->B=B;
  result->nested=cloneDimDecomp(nested);
  if( A*B != nested->coord_size ){
    fprintf(stderr,"FactorReverse: factors do not equal size\n");
    return NULL;
  }
  return (DimDecomp) result;
}

AxisDecomp makeFactorReverseAxisDecomp(int A, int B, AxisDecomp orig){
	return makeAxisDecomp(orig->comm,makeFactorReverseDecomp(A,B,orig->decomp),orig->memory_stride,orig->rank_stride);
}
