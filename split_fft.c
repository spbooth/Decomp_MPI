#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include "split_fft.h"
#include "interleave_decomp.h"
#include "block_decomp.h"
#include "factor_reverse_decomp.h"
#include "processor_grid.h"
#include "logger.h"

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif
#ifdef USE_TIMER
#include "timer.h"
#define TIMER timer
#else
#define TIMER MPI_Wtime
#endif

#ifndef SEARCH_REPEAT
#define SEARCH_REPEAT 1
#endif
#ifndef SEARCH_TRIAL
#define SEARCH_TRIAL 1 
#endif
#ifndef SEARCH_PASS
#define SEARCH_PASS 1
#endif
/** Make a Decomp for
 *
 */
Decomp makeInterleavedDecomp(int ndim,MPI_Comm comm, int A[ndim], int B[ndim],int *grid){
	DimDecomp *axis = (DimDecomp *) malloc(ndim * sizeof(DimDecomp));
	int nproc,i;
	MPI_Comm_size(comm,&nproc);
	logger(comm,"in makeInterLeavedDecomp\n");

	logger(comm,"made grid\n");
	for(i=0;i<ndim;i++){
		logger(comm,"A[%d]=%d grid=%d\n",i,A[i],grid[i]);
	}

	for(i=0;i<ndim;i++){
		DimDecomp blocked_decomp = makeBlockDecomp(A[i], grid[i]);
		axis[i] = makeInterleaveDecomp(blocked_decomp, B[i]);
	}
	logger(comm,"made DimDecomps\n");

	Decomp result =  makePackedDecomp(comm,ndim,axis);
	logger(comm,"made PackedDimDecomps\n");

	//free(grid);
	free(axis);
	logger(comm,"done free\n");

	return result;
}

Decomp makeBlockedDecomp(int ndim, MPI_Comm comm, int A[ndim], int B[ndim], int *grid){
    logger(comm,"In makeBlockedDecomp\n");
	DimDecomp *axis = (DimDecomp *) malloc(ndim * sizeof(DimDecomp));
		int nproc,i;
		MPI_Comm_size(comm,&nproc);
		for(i=0;i<ndim;i++){
			logger(comm,"A[%d]=%d B[%d]=%d grid[%d]=%d\n",i,A[i],i,B[i],i,grid[i]);
			axis[i] = makeBlockDecompWithFactor(A[i]*B[i],A[i],grid[i]);
		}
		Decomp result =  makePackedDecomp(comm,ndim,axis);

		//free(grid);
		free(axis);
		return result;
}
void testFull(int flags, int use_first, int use_second, int ndim, int A[ndim], int B[ndim], int L[ndim], int grid_a[ndim], int grid_b[ndim], SplitPlan *result, double *best_time, Decomp start, fftw_complex *dat,int length, fftw_complex workspace[length]){
  logger(start->comm,"making auto-tune trial plan\n");
  SplitPlan old_plan = *result;


  SplitPlan new_plan = buildParallelPlan(flags,use_first,use_second,start,ndim,A,B,L,grid_a,grid_b,dat,length,workspace);
  if( new_plan != NULL){
    double tstart, tstop;
    double best=1e20;
    int trial, repeat;
    MPI_Barrier(start->comm);
    for( trial=0 ; trial < SEARCH_TRIAL ; trial++){
      tstart=TIMER();
      for( repeat=0 ; repeat < SEARCH_REPEAT; repeat++){
	runSplitPlan(new_plan,FFTW_FORWARD);
      }
      tstop=TIMER();
      tstop = tstop -tstart;
      MPI_Allreduce(&tstop,&tstart,1,MPI_DOUBLE,MPI_MAX,start->comm);
      if( tstart > 0.0 && tstart < best){
	best = tstart;
      }
    }
    best = best / (double)SEARCH_REPEAT;
    //if( me == 0 ){
    //	 int i;
    //	 for(i=0;i<start->ndim;i++){
    //		 printf("A[%d]=%d ",i,new_plan->A_factors[i]);
    //	 }
    //	 //printf("time=%e\n",best);
    //}
    if( best < *best_time){
      //printf("new best\n");
      
      logger(start->comm,"found better plan %e\n",best);
      *best_time = best;
      *result = new_plan;
      if( old_plan != NULL ){
#ifndef LEAK
	logger(start->comm,"delete previous best\n");
	freeSplitPlan(old_plan);
#endif
      }
    }else{
#ifndef LEAK
      logger(start->comm,"delete trial plan\n");
      freeSplitPlan(new_plan);
#endif
    }
  }
}

void searchGridPlanned(int flags, int use_first, int use_second, int ndim, int A[ndim], int B[ndim], int L[ndim],int dim_a, int grid_a[ndim], int dim_b,int grid_b[ndim], SplitPlan *result, double *best_time, Decomp start, fftw_complex *dat,int length, fftw_complex workspace[length]){
  int nproc;
  MPI_Comm_size(start->comm,&nproc);
  if( dim_a < ndim ){
    int prev = 1;
    int i;
    for( i=0; i< dim_a; i++){
      prev *= grid_a[i];
    }
    for( grid_a[dim_a] = 1 ; (prev * grid_a[dim_a]) <= nproc && grid_a[dim_a] <= A[dim_a]; grid_a[dim_a]++){
      logger(start->comm,"searchGridPlanned dim_a= %d a-loop=%d prev=%d A=%d\n",dim_a,grid_a[dim_a],prev,A[dim_a]);
      searchGridPlanned(flags,use_first,use_second,ndim,A,B,L,dim_a+1,grid_a,dim_b,grid_b,result,best_time,start,dat,length,workspace);
    }
    return;  
  }else if( dim_b < ndim && (! use_first)){
    int prev = 1;
    int i;
    for( i=0; i< dim_b; i++){
      prev *= grid_b[i];
    }
    for( grid_b[dim_b] = 1 ; (prev * grid_b[dim_b]) <= nproc && grid_b[dim_b] <= B[dim_b]; grid_b[dim_b]++){
      searchGridPlanned(flags,use_first,use_second,ndim,A,B,L,dim_a,grid_a,dim_b+1,grid_b,result,best_time,start,dat,length,workspace);
    }
    return;  
  }else{
    testFull(flags, use_first,use_second,ndim,A,B,L,grid_a,grid_b,result,best_time,start,dat,length,workspace);
  }


}
void testFactors(int ndim, int *initial_factor, SplitPlan *result, double *best_time, Decomp start, fftw_complex *dat,int length, fftw_complex workspace[length]){
  //				 if( me == 0 ){
  //	 int i;
  //	 printf("makeParalellPlan ");
  //	 for(i=0;i<start->ndim;i++){
  //		 printf("B[%d]=%d ",i,B[i]);
  //	 }
  //	 printf("\n");
  //}
  int i;
  int use_first =1;
  int use_second=1; // are we going to do all FFTs in second step
  int flags = PREFER_STAGE1 ;

  int *A = (int *) malloc(ndim*sizeof(int));
  int *B = (int *) malloc(ndim*sizeof(int));
  int *L = (int *) malloc(ndim*sizeof(int));
  for(i=0;i<ndim;i++){
    B[i]=initial_factor[i];
    use_second = use_second && B[i] == 1;
    L[i]=sizeDimDecomp(start->dims[i]->decomp);
    A[i] = L[i]/B[i];
    if( A[i]*B[i] != L[i] ){
      raise_error(start->comm,"Invalid factor");
    }
    use_first = use_first && A[i] == 1;
    logger(start->comm,"A[%d]=%d B[%d]=%d L[%d]=%d\n",i,A[i],i,B[i],i,L[i]);
  }
  if( use_second ){
    logger(start->comm,"reject: single proc FFT on second pass\n");
    // same as use first use that in preference
    free(A);
    free(B);
    free(L);
    return;
  }

#ifdef GUESS_GRID  
  logger(start->comm,"make interleave decomp\n");
  int *grid_a=makeProcessorGrid(start->comm,ndim,A);
  int *grid_b=makeProcessorGrid(start->comm,ndim,B);
  testFull(flags, use_first,use_second,ndim,A,B,L,grid_a,grid_b,result,best_time,start,dat,length,workspace);
#else
  int *grid_a = calloc(sizeof(int),ndim);
  int *grid_b = calloc(sizeof(int),ndim);
  searchGridPlanned(flags, use_first,use_second,ndim,A,B,L,0,grid_a,0,grid_b,result,best_time,start,dat,length,workspace);

#endif
  free(grid_a);
  free(grid_b);
  free(A);
  free(B);
  free(L);
}
void searchPlan(int ndim, int dim, int *B, SplitPlan *result, double *best_time, Decomp start, fftw_complex *dat,int length, fftw_complex workspace[length]){
	int me;
	MPI_Comm_rank(start->comm,&me);
	int L = start->dims[dim]->decomp->coord_size;
	logger(start->comm,"dim=%d L=%d\n",dim,L);
	for(B[dim]=1;B[dim]<=L ; B[dim]++){
		logger(start->comm,"B[%d]=%d\n",dim,B[dim]);
		 if( 0 == (L % B[dim])  ){
			 if( dim < ndim-1 ){
				 logger(start->comm,"Go for recurse\n");

				 searchPlan(ndim,dim+1,B,result,best_time,start,dat,length,workspace);
			 }else{
				 testFactors(ndim,B,result,best_time,start,dat,length,workspace);
			 }
		 }
	}
}

SplitPlan autotuneParallelPlan(Decomp start, fftw_complex *dat,int length, fftw_complex workspace[length]){
	SplitPlan result = NULL;
	int pass;
	double best_time=1e+20;
	int ndim=start->ndim;
	int *factor = calloc(sizeof(int),ndim);
	// scan whole space multiple times in case of long term variability
	for(pass=0; pass< SEARCH_PASS ; pass++){
	  printf("search pass %d\n",pass);
		searchPlan(ndim,0,factor,&result,&best_time,start,dat,length,workspace);
	}
	bzero(factor,ndim*sizeof(int));
	free(factor);
	return result;
}

SplitPlan buildParallelPlan(int flags, int use_first, int use_second,Decomp start, int ndim,int  A[ndim], int B[ndim], int L[ndim],int grid_a[ndim], int grid_b[ndim],fftw_complex *dat,int length, fftw_complex workspace[length]){


  if( (flags & FORCE_USE_FIRST) && ! use_first ){
    return NULL;
  }

	double tstart,tstop;
	tstart=TIMER();
	SplitPlan result = (SplitPlan) calloc(sizeof(struct split_fft_plan),1);
	if( result == NULL){
		return result;
	}
	result->comm=start->comm;
	result->src=dat;
	result->work=workspace;
	int nproc;
	int me;

	MPI_Comm_size(start->comm,&nproc);
	MPI_Comm_rank(start->comm,&me);
	int i, qq, pp;
  logger(start->comm,"buildParallelPlan %d use_first=%d use_second=%d ",flags,use_first,use_second);
  for(i=0;i<ndim;i++){
    fragment_logger("dim=%d A=%d B=%d L=%d grid_a=%d grid_b=%d ",i,A[i],B[i],L[i],grid_a[i],grid_b[i]);

  }
  fragment_logger("\n");

	logger(start->comm,"make interleave decomp\n");

	// first decomposition has long distance factor local
	result->A = makeInterleavedDecomp(ndim,start->comm,A,B,grid_a);
#ifdef DEBUG
	printf("%d: decomp A\n",me);describeDeomp(stdout,result->A);
#endif
	logger(start->comm,"made interleave decomp\n");
	result->extent=extent(result->A);
	logger(start->comm,"A extent= %d\n",result->extent);


        if ( ! use_first ){
	  logger(start->comm,"make blocked decomp\n");
	  // second decomp is blocked in multiple of short-distance factor
	  result->B = makeBlockedDecomp(ndim,start->comm,A,B,grid_b);

	  int b_extent = extent(result->B);
	  logger(start->comm,"required b extent=%d\n",b_extent);
#ifdef DEBUG
	  printf("%d: decomp B\n",me);describeDeomp(stdout,result->B);
#endif
	  if( b_extent > result->extent){
	    result->extent=b_extent;
	  }
	}
	if( result->extent * 2 > length){
		logger(start->comm,"Not enough workspace need=%d have=%d\n",(result->extent*2),length);
		freeSplitPlan(result);
		return NULL;
	}
	if( (flags & PREFER_LARGE_FFT) && ! use_first){
		// If there is a common factor in the blocking redundancy
		// for the two decompositions then one of the FFT sizes can
		// be made larger reducing the redundancy of the other.
		for(i=0;i<ndim;i++){
			unsigned int a_redundancy, b_redundancy;
			InterleaveDecomp ad = (InterleaveDecomp) result->A->dims[i]->decomp;
			a_redundancy = ((BlockDecomp)ad->nested)->blocksize;
			if( a_redundancy > L[i]){
				raise_error(start->comm,"Bad a_redundancy %d\n",a_redundancy);
			}
			BlockDecomp bd = (BlockDecomp) result->B->dims[i]->decomp;
			b_redundancy = bd->blocksize / A[i];
			if( b_redundancy > L[i]){
				raise_error(start->comm,"Bad b_redundancy[%d]=%d\n",i,b_redundancy);
			}
			unsigned int g = gcd(a_redundancy, b_redundancy);
			logger(start->comm,"a_red[%d]=%d b_red[%d]=%d gcd=%d\n",i,a_redundancy,i,b_redundancy,g);
			if (g > 1) {
				logger(start->comm,"Over decimation of dimension=%d\n",i);
				freeSplitPlan(result);
				return NULL;
			}
		}

	}
	MPI_Datatype base;

	logger(start->comm,"make complex base type\n");
	// make fftw_complex datatype
	MPI_Type_contiguous(2,MPI_DOUBLE,&base);
	logger(start->comm,"make plan A\n");
	result->start_to_A = makeReshapePlan(start->comm,base,start,result->A);
	result->A_to_start = makeReshapePlan(start->comm,base,result->A,start);

	logger(start->comm,"use_first=%d use_second=%d\n",use_first,use_second);
	// We only need twiddle and second reshape if using 2 stages
	// the The A and B decomp should be equivalent in the 2 cases so
	// use the A reshape to set up.
	if( ! (use_first || use_second) ){
		int non_trivial=0;
		result->twiddle = (_Complex double **)calloc(sizeof(_Complex double *),ndim);

		logger(start->comm,"twiddle pointer\n");
		for(i=0;i<ndim;i++){
			int max=0;
			int global;
			DimDecomp decomp = result->A->dims[i]->decomp;
			for(global=0;global<L[i];global++){
				if( isCoordLocal(result->A,i,global)){

					int local = decomp->type_offset(decomp,global);
					logger(start->comm,"dim=%d global=%d local=%d\n",i,global,local);
					if( local >= max){
						if( max == 0 ){
							logger(start->comm,"allocate dim=%d size=%d \n",i,(local+1)*sizeof(_Complex double));
							result->twiddle[i] = calloc(sizeof(_Complex double),(local+1));
						}else{
							logger(start->comm,"re-allocate dim=%d size=%d\n",i,(local+1)*sizeof(_Complex double));
							result->twiddle[i] = realloc(result->twiddle[i],(local+1) *  sizeof(_Complex double));
						}
						logger(start->comm,"twiddle[%d] = %08x\n",i,result->twiddle[i]);
						max=local+1;
					}
					int y = global%A[i];
					int l = global/A[i];
					//printf("%d: global=%d y=%d l=%d\n",me,global,y,l);
					double angle = -(2.0 * M_PI) * ((double)(y * l) /(double)L[i]) ;
					double real = cos(angle);
					double imag = sin(angle);
					// get multiples of pi/2 exactly right
					int remainder = (y*l)%L[i];
					if( remainder == 0 ){
						real = 1.0;
						imag = 0.0;
					}else if( 4*remainder == L[i]){
						real = 0.0;
						imag = -1.0;
					}else if( 2 * remainder == L[i]){
						real = -1.0;
						imag= 0.0;
					}else if( 4 *  remainder == 3 * L[i] ){
						real = -0.0;
						imag= 1.0;
					}


					if( real != 1.0 && imag != 0.0){
						non_trivial = 1;
					}
					_Complex double f = real + I * imag;

					//printf("%d: global=%d y=%d l=%d theta=%e re=%e im=%e\n",me,global,y,l,angle,real,imag);
					result->twiddle[i][local] = f;
					//}else{
					//printf("%d: dim=%d global=%d non-local\n",me,i,global);
				}
			}

		}
		if( non_trivial == 0 ){
			// only trivial factors so supress twiddle factors
			logger(start->comm,"supressing twiddle factors\n");
			for(i=0;i<ndim;i++){
				if( result->twiddle[i] != NULL){
					free(result->twiddle[i]);
					result->twiddle[i]=NULL;
				}
			}
			free(result->twiddle);
			result->twiddle=NULL;
		}
		logger(start->comm,"make plan A->B\n");
		result->A_to_B = makeReshapePlan(start->comm,base,result->A,result->B);
		result->B_to_A = makeReshapePlan(start->comm,base,result->B,result->A);
	} // end ! use_first || use_second
	// Make the initial fft plan
	result->plan_a=NULL;
	if( me < result->A->proc_used && ! use_second){
		fftw_iodim *a_trans = calloc(sizeof(fftw_iodim),ndim);
		fftw_iodim *a_loop = calloc(sizeof(fftw_iodim),ndim);
		int ntrans=0;
		int nloop=0;
		for(i=0;i<ndim;i++){
			if( B[i] > 1){
				a_trans[ntrans].n=B[i];
				a_trans[ntrans].is=result->A->dims[i]->memory_stride;
				a_trans[ntrans].os=result->A->dims[i]->memory_stride;
				logger(start->comm,"stage-1 dim=%d fft-dim=%d size=%d stride=%d\n",i,ntrans,a_trans[ntrans].n,a_trans[ntrans].is);
				ntrans++;
			}
			int count = result->A->dims[i]->n_local / B[i];


			if( count > 1){
				a_loop[nloop].n = count;
				a_loop[nloop].is=result->A->dims[i]->memory_stride*B[i];
				a_loop[nloop].os=result->A->dims[i]->memory_stride*B[i];
				logger(start->comm,"stage-1 dim=%d loop-dim=%d size=%d stride=%d\n",i,nloop,a_loop[nloop].n,a_trans[nloop].is);
				nloop++;
			}

		}
		if( nloop == 0 ){
		  logger(start->comm,"adding single count loop\n");
			a_loop[nloop].n=1;
			a_loop[nloop].is=0;
			a_loop[nloop].os=0;
			nloop++;
		}
		logger(start->comm,"make FFTW plan_a\n");
		if( ntrans != 0 ){
			result->plan_a      =fftw_plan_guru_dft(ntrans,a_trans,nloop,a_loop,workspace+result->extent,workspace,FFTW_FORWARD,FFTW_MEASURE);
			logger(start->comm,"made forward plan_a\n");
#ifdef MAKE_REVERSE
			// reverse sense of strides
			for(i=0;i<ntrans;i++){
				int tmp = a_trans[i].is;
				a_trans[i].is=a_trans[i].os;
				a_trans[i].os=tmp;
			}
			for(i=0;i<nloop;i++){
				int tmp = a_loop[i].is;
				a_loop[i].is=a_loop[i].os;
				a_loop[i].os=tmp;
			}
			result->plan_a_back =fftw_plan_guru_dft(ntrans,a_trans,nloop,a_loop,workspace,workspace+result->extent,FFTW_BACKWARD,FFTW_MEASURE);
			logger(start->comm,"made reverse plan_a\n");
#endif
		}else{
			logger(start->comm,"No plan_a\n");
		}
		logger(start->comm,"Go for free\n");
		bzero(a_trans,ndim*sizeof(fftw_iodim));
		free(a_trans);
		bzero(a_loop,ndim*sizeof(fftw_iodim));
		free(a_loop);
	}else{
		logger(start->comm,"not part of first FFT %d >= %d\n",me,result->A->proc_used);
	}
        logger(start->comm,"made first FFT plan\n");
	// Make the second fft plan
	result->plan_b=NULL;
	if(! use_first &&  me < result->B->proc_used ){
		fftw_iodim *b_trans = calloc(sizeof(fftw_iodim),ndim);
		fftw_iodim *b_loop = calloc(sizeof(fftw_iodim),ndim);
		int ntrans=0;
		int nloop=0;
		for(i=0;i<ndim;i++){
			if( A[i] > 1 ){
				b_trans[ntrans].n=A[i];
				b_trans[ntrans].is=result->B->dims[i]->memory_stride;
				b_trans[ntrans].os=result->B->dims[i]->memory_stride;
				logger(start->comm,"stage-2 dim=%d fft-dim=%d size=%d stride=%d\n",i,ntrans,b_trans[ntrans].n,b_trans[ntrans].is);
				ntrans++;
			}
			int count = result->B->dims[i]->n_local / A[i];
			if( count > 1 ){
				b_loop[nloop].n = count;
				b_loop[nloop].is=result->B->dims[i]->memory_stride*A[i];
				b_loop[nloop].os=result->B->dims[i]->memory_stride*A[i];
				logger(start->comm,"stage-2 dim=%d loop-dim=%d size=%d stride=%d\n",i,nloop,b_loop[nloop].n,b_trans[nloop].is);
				nloop++;
			}
		}
		if( nloop == 0 ){
			b_loop[nloop].n=1;
			b_loop[nloop].is=0;
			b_loop[nloop].os=0;
			nloop++;
		}
		//printf("%d: make FFTW plan_b\n",me);
		if( ntrans != 0 ){
			result->plan_b      = fftw_plan_guru_dft(ntrans,b_trans,nloop,b_loop,workspace+result->extent, workspace,FFTW_FORWARD,FFTW_MEASURE);
			// reverse sense of strides
			for(i=0;i<ntrans;i++){
				int tmp = b_trans[i].is;
				b_trans[i].is=b_trans[i].os;
				b_trans[i].os=tmp;
			}
			for(i=0;i<nloop;i++){
				int tmp = b_loop[i].is;
				b_loop[i].is=b_loop[i].os;
				b_loop[i].os=tmp;
			}
			result->plan_b_back = fftw_plan_guru_dft(ntrans,b_trans,nloop,b_loop,workspace,workspace+result->extent,FFTW_BACKWARD,FFTW_MEASURE);
		}else{
			logger(start->comm,"no plan_b\n");
		}
		bzero(b_trans,ndim*sizeof(fftw_iodim));
		free(b_trans);
		bzero(b_loop,ndim*sizeof(fftw_iodim));
		free(b_loop);
	}else{
	  if( ! use_first ){
		logger(start->comm,"not part of second FFT %d >= %d\n",me,result->B->proc_used);
	  }
	}
        logger(start->comm,"made second FFT plan\n");
	// make the flip transformation
        if( ! use_first ){
	  AxisDecomp *flipped = calloc(sizeof(AxisDecomp),ndim);
	  for(i=0;i<ndim;i++){
	    flipped[i] = makeFactorReverseAxisDecomp(A[i],B[i],result->B->dims[i]);
	  }
	  result->B_flipped = makeDecomp(ndim,flipped);
	  free(flipped);
        }
        logger(start->comm,"made flipped\n");

	result->A_factors=calloc(sizeof(int),ndim);
        for(i=0;i<ndim;i++){
	  result->A_factors[i]=A[i];
	}
        logger(start->comm,"saved A\n");

	result->times=calloc(sizeof(double),NUM_TIMERS);
	tstop=TIMER();
	logger(start->comm,"plan time = %e\n",tstop-tstart);
	return result;
}
SplitPlan makeParallelPlan(Decomp start,int *initial_factor, fftw_complex *dat,int length, fftw_complex workspace[length]){
	return proposeParallelPlan(NO_FLAGS,start,initial_factor,dat,length,workspace);
}
SplitPlan makeSplitPlan(int slab,Decomp start,fftw_complex *dat,int length, fftw_complex workspace[length]){
	int ndim = start->ndim;
	int i;
		int *B = (int *) calloc(sizeof(int),ndim);

		for(i=0;i<ndim;i++){
			int AA, BB, LL;
			int qq,pp;
			AA=1;
			LL=BB=sizeDimDecomp(start->dims[i]->decomp);

			if( slab == 1 ){
				// have all dimensions except the last local
				if( i == (ndim-1)){
					BB=1;
					AA=LL;
				}
			}else{
				// now come up with two roughly equal factors
				// with A less than or equal to B
				for(qq=1;qq<LL;qq++){
					if( LL % qq == 0){
						pp = LL/qq;
						if( qq > AA && qq <= pp){
							AA=qq;
							BB=pp;
						}
					}
				}
			}
			// Might be a good idea to swap order of P/Q every other dim
			// to possibly mix large/small
			B[i]=BB;
			logger(start->comm,"A[%d]=%d B[%d]=%d L[%d]=%d\n",i,AA,i,BB,i,LL);
		}
		SplitPlan result = makeParallelPlan(start,B,dat,length,workspace);
		free(B);
		if( result == NULL ){
			raise_error(start->comm,"Cannot generate plan");
		}
		return result;
}
Decomp getTransformed(SplitPlan plan){
	//printf("get flipped\n");
	// print
	//decomp_scale(plan->B_flipped,NULL,plan->work);

	return plan->B_flipped;
}
void runSplitPlan(SplitPlan plan, int direction){
	double tstart, tstop;
	double allstart,allstop;
	int me;

	MPI_Comm_rank(plan->comm,&me);
	if( direction == FFTW_FORWARD){
		logger(plan->comm,"reshape A\n");
		allstart=tstart=TIMER();
		runReshapePlan(plan->comm,plan->start_to_A,plan->src,plan->work+plan->extent);
		tstop=TIMER();
		plan->times[RESHAPE_1]+=tstop-tstart;
		if( plan->plan_a != NULL ){
			// first FFT from end of workspace to lower slot
			logger(plan->comm,"Execute plan_a\n");
			tstart=TIMER();
			fftw_execute(plan->plan_a);
			tstop=TIMER();
			plan->times[FFT_1]+=tstop-tstart;
		}else{
			logger(plan->comm,"No plan_a\n");
		}
		if( plan->twiddle != NULL){
            logger(plan->comm,"Do twiddle\n");
			tstart=TIMER();
			// null twiddle vector means only trivial factors.
			decomp_scale(plan->A,plan->twiddle,plan->work,-1);
			tstop=TIMER();
			plan->times[TWIDDLE]+=tstop-tstart;
		}else{
			logger(plan->comm,"No twiddle\n");
		}
		if( plan->A_to_B != NULL){
			logger(plan->comm,"reshape B\n");
			tstart=TIMER();
			runReshapePlan(plan->comm,plan->A_to_B,plan->work,plan->work+plan->extent);
			tstop=TIMER();
			plan->times[RESHAPE_2]+=tstop-tstart;
		}else{
			logger(plan->comm,"No reshape A->B\n");
		}
		// second FFT from upper to lower again
			// print
			//decomp_scale(plan->B,NULL,plan->work+plan->extent);


		//printf("%d: Execute plan_b\n",me);

		if( plan->plan_b != NULL){
			tstart=TIMER();
			logger(plan->comm,"Execute plan_b\n");
			fftw_execute(plan->plan_b);
			tstop=TIMER();
			plan->times[FFT_2]+=tstop-tstart;
		}else{
			logger(plan->comm,"no plan_b !!!!\n");
		}
		allstop=TIMER();
		plan->times[ALL]+=allstop-allstart;
			// print
			//decomp_scale(plan->B,NULL,plan->work);
	}else if( direction == FFTW_BACKWARD){
		logger(plan->comm,"reshape A\n");
		allstart=tstart=TIMER();

		if( plan->plan_b_back != NULL){
			tstart=TIMER();
			logger(plan->comm,"Execute plan_b_back\n");
			fftw_execute(plan->plan_b_back);
			tstop=TIMER();
			plan->times[FFT_2]+=tstop-tstart;
		}else{
			logger(plan->comm,"no plan_b_back !!!!\n");
		}
		if( plan->B_to_A != NULL){
			logger(plan->comm,"reshape B reverse\n");
			tstart=TIMER();
			runReshapePlan(plan->comm,plan->B_to_A,plan->work+plan->extent,plan->work);
			tstop=TIMER();
			plan->times[RESHAPE_2]+=tstop-tstart;
		}else{
			logger(plan->comm,"No reshape A->B\n");
		}
		if( plan->twiddle != NULL){
			logger(plan->comm,"Do twiddle reverse\n");
			tstart=TIMER();
			// null twiddle vector means only trivial factors.
			decomp_scale(plan->A,plan->twiddle,plan->work,1);
			tstop=TIMER();
			plan->times[TWIDDLE]+=tstop-tstart;
		}else{
			logger(plan->comm,"No twiddle reverse\n");
		}

		runReshapePlan(plan->comm,plan->A_to_start,plan->work+plan->extent,plan->src);
		tstop=TIMER();
		plan->times[RESHAPE_1]+=tstop-tstart;
		if( plan->plan_a_back != NULL ){
			// first FFT from end of workspace to lower slot
			logger(plan->comm,"Execute plan_a_back\n");
			tstart=TIMER();
			fftw_execute(plan->plan_a_back);
			tstop=TIMER();
			plan->times[FFT_1]+=tstop-tstart;
		}else{
			logger(plan->comm,"No plan_a_back\n");
		}


		// second FFT from upper to lower again
			// print
			//decomp_scale(plan->B,NULL,plan->work+plan->extent);


		//printf("%d: Execute plan_b\n",me);


		allstop=TIMER();
		plan->times[ALL]+=allstop-allstart;
			// print
			//decomp_scale(plan->B,NULL,plan->work);
	}
}


SplitPlan proposeParallelPlan(int flags,Decomp start,int *initial_factor, fftw_complex *dat,int length, fftw_complex workspace[length]){

	int ndim = start->ndim;
	int use_first =1;
	int use_second=1; // are we going to do all FFTs in second step
        int i;
	int *A = (int *) malloc(ndim*sizeof(int));
	int *B = (int *) malloc(ndim*sizeof(int));
	int *L = (int *) malloc(ndim*sizeof(int));
	for(i=0;i<ndim;i++){
		B[i]=initial_factor[i];
		use_second = use_second && B[i] == 1;
		L[i]=sizeDimDecomp(start->dims[i]->decomp);
		A[i] = L[i]/B[i];
		if( A[i]*B[i] != L[i] ){
			raise_error(start->comm,"Invalid factor");
		}
		use_first = use_first && A[i] == 1;
		logger(start->comm,"A[%d]=%d B[%d]=%d L[%d]=%d\n",i,A[i],i,B[i],i,L[i]);
	}
	if( use_second && (flags & PREFER_STAGE1)){
			logger(start->comm,"reject: single proc FFT on second pass\n");
		// same as use first use that in preference
			free(A);
			free(B);
			free(L);
			return NULL;
	}
	int *grid_a=makeProcessorGrid(start->comm,ndim,A);
	int *grid_b=makeProcessorGrid(start->comm,ndim,B);
	SplitPlan *result = NULL;

        result = buildParallelPlan(flags,use_first,use_second,start,ndim,A,B,L,grid_a,grid_b,dat,length,workspace);
        free(grid_a);
        free(grid_b);
        free(A);
        free(B);
        free(L);

        return result;
        
}
void freeSplitPlan(SplitPlan plan){
	int ndim;
	int i;
	if( plan->A_factors != NULL){
		free(plan->A_factors);
	}
	if( plan->A != NULL){
		ndim=plan->A->ndim;
		logger(plan->comm,"free A\n");
		freeDecomp(plan->A);
	}
	if( plan->B != NULL){
		logger(plan->comm,"free B\n");
		freeDecomp(plan->B);
	}
	if( plan->B_flipped != NULL){
		logger(plan->comm,"free flipped\n");
		freeDecomp(plan->B_flipped);
	}
	if( plan->start_to_A != NULL ){
		logger(plan->comm,"free start_to_A\n");
		freeReshapePlan(plan->start_to_A);
	}
	if( plan->A_to_start != NULL ){
		logger(plan->comm,"free A_to_start\n");
		freeReshapePlan(plan->A_to_start);
	}
	if( plan->A_to_B != NULL ){
		logger(plan->comm,"free A_to_B\n");
		freeReshapePlan(plan->A_to_B);
	}
	if( plan->B_to_A != NULL ){
		logger(plan->comm,"free B_to_A\n");
		freeReshapePlan(plan->B_to_A);
	}
	if( plan->twiddle != NULL){
		for(i=0 ; i<ndim; i++){
			if( plan->twiddle[i] != NULL){
				logger(plan->comm,"free twiddle[%d]\n",i);
				free(plan->twiddle[i]);
				plan->twiddle[i]=NULL;
			}
		}
		logger(plan->comm,"free twiddle\n");
		free(plan->twiddle);
		plan->twiddle=NULL;
	}
	if( plan->plan_a != NULL){
		logger(plan->comm,"destroy plan_a\n");
		fftw_destroy_plan(plan->plan_a);
	}
	if( plan->plan_a_back != NULL){
		logger(plan->comm,"destroy plan_a_back\n");
		fftw_destroy_plan(plan->plan_a_back);
	}
	if( plan->plan_b != NULL){
		logger(plan->comm,"destroy plan_b\n");
		fftw_destroy_plan(plan->plan_b);
	}
	if( plan->plan_b_back != NULL){
		logger(plan->comm,"destroy plan_b_back\n");
		fftw_destroy_plan(plan->plan_b_back);
	}
	if( plan->times != NULL ){
		logger(plan->comm,"free times\n");
		free(plan->times);
		plan->times=NULL;
	}
	free(plan);
}



void dim_visit(int dim,int *global_pos,Decomp decomp,data_visitor vis , _Complex double *data, int offset){
	int pos;
	//printf("In dim_scale dim=%d\n",dim);
	DimDecomp dim_decomp = decomp->dims[dim]->decomp;
	int count = decomp->dims[dim]->max_local+1;
	int memory_stride =decomp->dims[dim]->memory_stride;
	for( pos = 0 ; pos < count ; pos++){
		//printf("scale dim=%d global=%d local=%d\n",dim,global,pos);
		int new_offset = offset + (pos * memory_stride);


		if( dim == (decomp->ndim -1)){
			vis(decomp->ndim,global_pos,data+new_offset);
		}else{
			// recurse
			dim_visit(dim+1,global_pos,decomp,vis,data, new_offset);
		}
	}
}

void visitData(Decomp decomp, data_visitor vis, _Complex double *dat){
	int *global_pos;

	global_pos= calloc(sizeof(int),decomp->ndim);
	dim_visit(0,global_pos,decomp,vis,dat,0);

	free(global_pos);
}
unsigned int gcd(unsigned int u, unsigned int v)
{
  int shift;

  /* GCD(0,v) == v; GCD(u,0) == u, GCD(0,0) == 0 */
  if (u == 0) return v;
  if (v == 0) return u;

  /* Let shift := lg K, where K is the greatest power of 2
        dividing both u and v. */
  for (shift = 0; ((u | v) & 1) == 0; ++shift) {
         u >>= 1;
         v >>= 1;
  }

  while ((u & 1) == 0)
    u >>= 1;

  /* From here on, u is always odd. */
  do {
       /* remove all factors of 2 in v -- they are not common */
       /*   note: v is not zero, so while will terminate */
       while ((v & 1) == 0)  /* Loop X */
           v >>= 1;

       /* Now u and v are both odd. Swap if necessary so u <= v,
          then set v = v - u (which is even). For bignums, the
          swapping is just pointer movement, and the subtraction
          can be done in-place. */
       if (u > v) {
         unsigned int t = v; v = u; u = t;}  // Swap u and v.
       v = v - u;                       // Here v >= u.
     } while (v != 0);

  /* restore common factors of 2 */
  return u << shift;
}
