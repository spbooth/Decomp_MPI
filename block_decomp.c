#include <stdlib.h>
#include <string.h>
#include "block_decomp.h"


int block_rank_offset(DimDecomp d, int global){
  BlockDecomp bd=(BlockDecomp)d;
  return global/bd->blocksize;
}
int block_type_offset(DimDecomp d,int global){
  BlockDecomp bd=(BlockDecomp)d;
  return global%bd->blocksize;
}
void printBlock(FILE *out, DimDecomp d){
	BlockDecomp bd=(BlockDecomp)d;
	fprintf(out," blocksize=%d",bd->blocksize);
}
DimDecomp cloneBlockDecomp(DimDecomp d){
	BlockDecomp bd=(BlockDecomp)d;
	BlockDecomp result = calloc(sizeof(struct block_decomp),1);
	return memmove(result,bd,sizeof(struct block_decomp));
}
DimDecomp makeBlockDecomp(int len,int procs){
	return makeBlockDecompWithFactor(len,1,procs);
}
/** Make a BlockDecomp
 *
 * @param len length of axis
 * @param factor required factor of block size (must also factor len)
 * @param procs number of procs
 *
 */
DimDecomp makeBlockDecompWithFactor(int len,int factor,int procs){


	BlockDecomp result = (BlockDecomp) calloc(sizeof(struct block_decomp),1);
	if( len % factor != 0){
		return NULL;
	}
	result->parent.coord_size=len;
	result->parent.proc_size=procs;
	result->parent.rank_offset=block_rank_offset;
	result->parent.type_offset=block_type_offset;
	result->parent.printer=printBlock;
	result->parent.destroy=NULL;
	result->parent.cloner=cloneBlockDecomp;
	result->parent.type="block";
	result->blocksize=factor*(((len/factor)+procs-1)/procs);

	return (DimDecomp) result;
}
