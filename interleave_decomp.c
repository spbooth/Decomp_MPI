#include <stdlib.h>
#include <string.h>
#include "interleave_decomp.h"


int interleave_rank_offset(DimDecomp d, int global){
  InterleaveDecomp id=(InterleaveDecomp)d;
  int y = global % id->A;
  int z = global / id->A;

  return id->nested->rank_offset(id->nested,y);
}
int interleave_type_offset(DimDecomp d,int global){
  InterleaveDecomp id=(InterleaveDecomp)d;
  int y = global % id->A;
  int z = global / id->A;

  // sequences assigned to processors cyclically
  return z + id->B*( id->nested->type_offset(id->nested,y));
}

void interleave_destory(DimDecomp d){
	 InterleaveDecomp id=(InterleaveDecomp)d;
	 if( id->nested->destroy != NULL ){
		 id->nested->destroy(id->nested);
	 }
}
void print_interleave(FILE *out,DimDecomp d){
	InterleaveDecomp id=(InterleaveDecomp)d;
	fprintf(out," A=%d B=%d nested:",id->A,id->B);
	describeDimDecomp(out,id->nested);
}
DimDecomp clone_interleave(DimDecomp d){
	InterleaveDecomp id=(InterleaveDecomp)d;
	InterleaveDecomp result = calloc(sizeof(struct interleave_decomp),1);
	memmove(result,id,sizeof(struct interleave_decomp));
	result->nested=cloneDimDecomp(id->nested);
	return (DimDecomp) result;
}

DimDecomp makeInterleaveDecomp(DimDecomp nested, int B){
  InterleaveDecomp result = (InterleaveDecomp) calloc(sizeof(struct interleave_decomp),1);
  result->B = B;
  result->A = nested->coord_size;
  int len = result->A * result->B;
  result->parent.coord_size=len;
  result->parent.proc_size=nested->proc_size;
  result->parent.rank_offset=interleave_rank_offset;
  result->parent.type_offset=interleave_type_offset;
  result->parent.destroy=interleave_destory;
  result->parent.printer=print_interleave;
  result->parent.cloner=clone_interleave;
  result->parent.type="interleave";
  result->nested=cloneDimDecomp(nested);
  return (DimDecomp) result;
}
