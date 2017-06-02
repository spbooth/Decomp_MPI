#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include "reshape.h"


void dim_scale(int dim,int *global_pos,Decomp decomp, _Complex double factor, _Complex double **factors, _Complex double *data, int offset, int sign){
	int pos;
	//printf("In dim_scale dim=%d\n",dim);
	DimDecomp dim_decomp = decomp->dims[dim]->decomp;
	int count = decomp->dims[dim]->max_local+1;
	int memory_stride =decomp->dims[dim]->memory_stride;
	for( pos = 0 ; pos < count ; pos++){
			//printf("scale dim=%d global=%d local=%d\n",dim,global,pos);
			int new_offset = offset + (pos * memory_stride);
			_Complex double new_factor;
			if( factors == NULL || factors[dim] == NULL){
				//printf("supressed factor dim=%d pos=%d\n",dim,pos);
				new_factor=factor;
			}else{
				new_factor = factor * factors[dim][pos];
				//printf("factor update (%e,%e) * (%e,%e) -> (%e,%e)\n",
				//		creal(factor),cimag(factor),
				//		creal(factors[dim][pos]),cimag(factors[dim][pos]),
				//		creal(new_factor),cimag(new_factor));
			}
			// A zero denotes an unused index
			if( new_factor !=  0.0){

				if( dim == (decomp->ndim -1)){
					if( sign == 1){
						new_factor = conj(new_factor);
					}
					_Complex double old_value = data[new_offset];
					_Complex double new_value = new_factor * old_value;
					//int j;
					//for(j=0;j<decomp->ndim;j++){
					//	printf("[%d]",global_pos[j]);
					//}
					//printf("scale offset=%d factor=(%e,%e) (%e,%e)->(%e,%e)\n",new_offset,creal(new_factor),cimag(new_factor),creal(old_value),cimag(old_value),creal(new_value),cimag(new_value));
					data[new_offset] = new_value;
				}else{
					// recurse
					dim_scale(dim+1,global_pos,decomp,new_factor,factors,data, new_offset,sign);
				}
			}
	}

}
/** apply a dimension by dimension scaling
 *
 */
void decomp_scale(Decomp decomp, _Complex double **factors, _Complex double *data, int sign ){
	//printf("In decomp_scale\n");
	_Complex double factor = 1.0 + 0.0*I;

	int *global_pos;

	global_pos= malloc(decomp->ndim * sizeof(int));
    //int i;
    //for(i=0;i<decomp->ndim;i++){
    //	global_pos[i]=0;
    //}
	dim_scale(0,global_pos,decomp,factor, factors,data,0,sign);

	free(global_pos);
}
