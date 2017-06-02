#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "processor_grid.h"
#include "logger.h"

GridResult makeGridResult(int ndim){
	GridResult result =  calloc(sizeof(struct grid_result),1);
	result->ndim=ndim;
	result->length=calloc(sizeof(int),ndim);
	result->load=0.0;
	return result;
}
void setScore(int ndim, GridResult result, int ntasks[ndim] , int grid[ndim] ){

}
void freeGridResult(GridResult result){
	bzero(result->length,result->ndim*sizeof(int));
	free(result->length);
	bzero(result,sizeof(struct grid_result));
	free(result);
}

void searchGrid(MPI_Comm comm, GridResult best,  int ndim, int mydim,int grid[ndim], int tasks[ndim],int max_proc, int proc, int load, int min, int max ){
	int newproc=proc;
	int i;
	logger(comm,"searching dim=%d out of %d max_proc=%d newproc=%d tasks=%d load=%d min=%d max=%d\n",mydim,ndim,max_proc,newproc,tasks[mydim],load,min,max);
	for(grid[mydim]=1;grid[mydim]<=max_proc && grid[mydim]<=tasks[mydim]&& newproc <=max_proc;grid[mydim]++){
		for(i=0;i<=mydim;i++){
			logger(comm,"grid[%d]=%d\n",i,grid[i]);
		}
		newproc = proc * grid[mydim];
		if( newproc <= max_proc){
			int mymin = tasks[mydim]/grid[mydim];
			int mymax = (tasks[mydim]+grid[mydim]-1)/grid[mydim];
			int newload = load*mymax;
			if( min < mymin){
				mymin=min;
			}
			if( max > mymax){
				mymax = max;
			}
			if(mydim < (ndim - 1)){
				searchGrid(comm,best,ndim,mydim+1,grid,tasks,max_proc,newproc,newload,mymin,mymax);
			}else{
				double ratio = ((double)mymax)/((double)mymin);
				logger(comm,"newload=%d best->load=%d ratio=%e\n",newload,best->load,ratio);
				if( newload <= best->load || best->load == 0.0){

					if(best->load == 0.0 || newload < best->load ||  ratio < best->asym){
						logger(comm,"found better\n");
						// better result
						best->procs=newproc;
						best->load=newload;
						best->asym=ratio;
						int i;
						for(i=0;i<ndim;i++){
							best->length[i]=grid[i];
						}

					}
				}
			}
		}

	}
}

/** Makes a processor grid for dividing the array of parallel tasks
 * fairly evenly but constrained to a maximum number of processors
 *
 */
int * makeProcessorGrid(MPI_Comm comm,int ndim, int tasks[ndim]){
	GridResult gr = makeGridResult(ndim);
	int *grid = calloc(sizeof(int),ndim);
	gr->procs=1;
	gr->load=1;
	gr->asym=10000.0;
	int npe;
	MPI_Comm_size(comm,&npe);
	int i;
	int max;
	max=1;
	for(i=0;i<ndim;i++){
		gr->length[i]=1;
		gr->load *= tasks[i];
		if(tasks[i]>max){
			max = tasks[i];
		}
	}
	searchGrid(comm,gr,ndim,0,grid,tasks,npe,1,1,max,0);
	for(i=0;i<ndim;i++){
		grid[i]=gr->length[i];
	}
	freeGridResult(gr);
	return grid;
}

