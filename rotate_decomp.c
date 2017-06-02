#include <stdlib.h>
#include <stdio.h>
#include "rotate_decomp.h"

int rotate(RotateDecomp rd,int global);

int rotate_rank_offset(DimDecomp d, int global){
  RotateDecomp rd = (RotateDecomp)d;
  return rd->nested->rank_offset(rd->nested,rotate(rd,global));
}
int rotate_type_offset(DimDecomp d,int global){
  RotateDecomp rd = (RotateDecomp)d;
  return rd->nested->type_offset(rd->nested,rotate(rd,global));
}
void rotate_destroy(DimDecomp d){
  RotateDecomp rd = (RotateDecomp)d;
  freeDimDecomp(rd->nested);
}

DimDecomp clone_rotate(DimDecomp d){
	RotateDecomp id=(RotateDecomp)d;
	RotateDecomp result = calloc(sizeof(struct rotate_decomp),1);
	memmove(result,id,sizeof(struct rotate_decomp));
	result->nested=cloneDimDecomp(id->nested);
	return result;
}

int rotate(RotateDecomp rd,int global){
  int len = rd->parent.coord_size;
  int i= (global +len + rd->rotate )%len;
  //fprintf(stderr,"rotate %d->%d\n",global,i);
  return i;
}


DimDecomp makeRotatedDecomp(int rotate, DimDecomp nested){
  RotateDecomp result = (RotateDecomp) calloc(sizeof(struct rotate_decomp),1);

  result->parent.coord_size=nested->coord_size;
  result->parent.proc_size=nested->proc_size;
  result->parent.rank_offset=rotate_rank_offset;
  result->parent.type_offset=rotate_type_offset;
  result->parent.type="rotate";
  result->parent.destroy=rotate_destroy;
  result->parent.cloner=clone_rotate;
  result->rotate=rotate;
  result->nested=cloneDimDecomp(nested);

  return (DimDecomp) result;
}
