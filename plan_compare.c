#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <complex.h>
#include <fftw3.h>
#include "reshape.h"
#include "local_decomp.h"
#include "block_decomp.h"
#include "split_fft.h"
#include "processor_grid.h"
#include "logger.h"
#ifdef USE_TIMER
#include "timer.h"
#define TIMER timer
#else
#define TIMER MPI_Wtime
#endif

#ifndef REPEAT
#define REPEAT 10
#endif
#ifndef TRIAL
#define TRIAL 10 
#endif

#ifdef DEBUG
#include <mcheck.h>
#endif

void my_check_error(int code, char *string)
{
  char exp[MPI_MAX_ERROR_STRING];
  int len;



  if( code != MPI_SUCCESS )
  {
     MPI_Error_string(code,exp,&len);
     fprintf(stderr,"MPI error [%s] %d %s\n",exp,code,string);
     exit(1);
  }
}
#ifdef USE_IO
void writeFile(int size,MPI_Comm comm,MPI_Datatype type,char *name, Decomp d, fftw_complex *dat){
  MPI_File file;
  MPI_Datatype filetype;
  MPI_Offset disp=0;
  int i,j,k;
  int ierror;
  double tstop,tstart;

  tstart=MPI_Wtime();
  logger(comm,"file open\n");
  ierror=MPI_File_open(comm,name,MPI_MODE_WRONLY|MPI_MODE_CREATE,MPI_INFO_NULL,&file);
  my_check_error(ierror,"File open");
  logger(comm,"make io type\n");
  makeIOType(&filetype,type,d);

  logger(comm,"set view\n");
  ierror=MPI_File_set_view(file,disp, type,filetype,"native",MPI_INFO_NULL);
  my_check_error(ierror,"File_set_view");
  for(k=0;k<size;k++){
	  if( isLocal(d->dims[0],k)){
		  for(j=0;j<size;j++){
			  if( isLocal(d->dims[1],j)){
				  for(i=0;i<size;i++){
					  if( isLocal(d->dims[2],i)){
						  int coord[3];
						  coord[0]=k;
						  coord[1]=j;
						  coord[2]=i;
						  int pos = getOffset(d,coord);
						  logger(comm,"write [%d,%d,%d]->%d\n",k,j,i,pos);
						  ierror=MPI_File_write(file,dat+pos,1,type,MPI_STATUS_IGNORE);
						  my_check_error(ierror,"File write");

					  }
				  }
			  }
		  }
	  }
  }
  logger(comm,"close\n");
  MPI_File_close(&file);
  MPI_Barrier(comm);
  tstop=MPI_Wtime();
  printf("write-time=%e\n",tstop-tstart);
}
#endif
void setArray(int size,int rank,int len,Decomp d, fftw_complex *dat){
  double tstop,tstart;
  int i,j,k;
  int ierror;

  tstart=MPI_Wtime();
  for(k=0;k<size;k++){
	  if( isLocal(d->dims[0],k)){
		  for(j=0;j<size;j++){
			  if( isLocal(d->dims[1],j)){
				  for(i=0;i<size;i++){
					  if( isLocal(d->dims[2],i)){
						  int coord[3];
						  coord[0]=k;
						  coord[1]=j;
						  coord[2]=i;
						  int pos = getOffset(d,coord);
						  if( pos < 0 || pos >= len){
							  fprintf(stderr,"Bad index %d",pos);
						  }else{
#ifdef POINT
							  int val=0;
							  if(i==1&& j== 0 && k== 0){
								  // single point
								  val=1;
							  }
#else
							  int val = i + size *( j + size *k);
#endif
							  dat[pos] = ((double) val) + 0.0 * I;
						  }
					  }
				  }
			  }
		  }
	  }
  }
  tstop=MPI_Wtime();
  printf("setArray time=%e\n",tstop-tstart);
}
int checkArray(int size,int len,Decomp d, fftw_complex *dat){
  int i,j,k;
  int ierror=0;
  double tstop,tstart;

  tstart=MPI_Wtime();
  for(k=0;k<size;k++){
	  if( isLocal(d->dims[0],k)){
		  for(j=0;j<size;j++){
			  if( isLocal(d->dims[1],j)){
				  for(i=0;i<size;i++){
					  if( isLocal(d->dims[2],i)){
						  int coord[3];
						  coord[0]=k;
						  coord[1]=j;
						  coord[2]=i;
						  int pos = getOffset(d,coord);
						  if( pos < 0 || pos >= len){
							  fprintf(stderr,"Bad index pos=%d",pos);
						  }else{
#ifdef POINT
							  int val=0;
							  if(i==1 && j== 0 && k==0){
								  val=1;
							  }
#else
							  int val = i + size *( j + size *k);
#endif
							  if( dat[pos] !=  ((double)val) + 0.0*I ){
								  fprintf(stderr, "Incorrect array value [%d,%d,%d] got (%g,%g) expected (%g,%g)\n",k,j,i,creal(dat[pos]),cimag(dat[pos]),(double)val,0.0);
								  ierror++;
							  }
						  }
					  }
				  }
			  }
		  }
	  }
  }
  tstop=MPI_Wtime();
  printf("checkArray time=%e\n",tstop-tstart);
  return ierror;
}
void zapArray(int len, fftw_complex *p){
	double tstop, tstart;
	tstart=MPI_Wtime();
	int i;
	for(i=0;i<len;i++){
		p[i]=-99.0 + -33.0 *I;
	}
	tstop=MPI_Wtime();
	printf("zap time=%e\n",tstop-tstart);
}

#ifdef USE_IO
void readFile(int size,MPI_Comm comm,MPI_Datatype type,char *name, Decomp d, fftw_complex *dat){
  MPI_File file;
  MPI_Datatype filetype;
  MPI_Offset disp=0;
  int i,j,k;
  int ierror;

  ierror=MPI_File_open(comm,name,MPI_MODE_RDONLY,MPI_INFO_NULL,&file);
  my_check_error(ierror,"File open");
  makeIOType(&filetype,type,d);

  ierror=MPI_File_set_view(file,disp, type,filetype,"native",MPI_INFO_NULL);
  my_check_error(ierror,"File_set_view");
  for(k=0;k<size;k++){
	  if( isLocal(d->dims[0],k)){
		  for(j=0;j<size;j++){
			  if( isLocal(d->dims[1],j)){
				  for(i=0;i<size;i++){
					  if( isLocal(d->dims[2],i)){
						  int coord[3];
						  coord[0]=k;
						  coord[1]=j;
						  coord[2]=i;
						  int pos = getOffset(d,coord);
						  ierror=MPI_File_read(file,dat+pos,1,type,MPI_STATUS_IGNORE);
						  my_check_error(ierror,"File read");

					  }
				  }
			  }
		  }
	  }
  }
  MPI_File_close(&file);


}
#endif
#define FUDGE 4
int main(int argc, char **argv){
#ifdef DEBUG
	  mtrace();
#endif
	 int error;
	  int npe, me, i,j;

	  int L[3];
	  char name[80];
	  fftw_complex *in;
	  fftw_complex *out;
	  double tstart, tstop;
	  int size=16;
	  if( argc >= 2 ){
		  size=atoi(argv[1]);
	  }

	  if( size < 1 ){
		  fprintf(stderr,"Bad size %d\n",size);
		  exit(1);
	  }
	  int global_vol = size*size*size;
#ifdef THREADED
	  int provided, threads_ok;
	  error = MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
	  threads_ok = provided >= MPI_THREAD_FUNNELED;
	  if (threads_ok) threads_ok = fftw_init_threads();
	  int nthreads=0;
	  if( argc >= 3 ){
		  nthreads=atoi(argv[2]);
	  }
	  fftw_plan_with_nthreads(nthreads);
#else
	  error = MPI_Init(&argc,&argv);
#endif

	  error = MPI_Comm_size(MPI_COMM_WORLD, &npe);

	  error = MPI_Comm_rank(MPI_COMM_WORLD, &me);
#ifdef REDIRECT
	  sprintf(name,"search_output%d.%d.%d.txt",me,npe,size);
	  freopen(name,"w",stdout);
#endif
	  setlinebuf(stdout);
	  printf("hello world from %d of %d\n",me,npe);
	  MPI_Datatype base;

	  // make fftw_complex datatype
	  MPI_Type_contiguous(2,MPI_DOUBLE,&base);
	  MPI_Type_commit(&base);

	  L[0]=size;
	  L[1]=size;
	  L[2]=size;
	  ptrdiff_t n0,n0_start;

	  ptrdiff_t len = (size*size*size); // generous

	  // old version compat
	  //in = fftw_alloc_complex(len);
	  //out = fftw_alloc_complex(len);
	  in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex)*len);
	  if( in == NULL){
		  perror("malloc failed for initial");
		  exit(2);
	  }
	  zapArray(len,in);
	  ptrdiff_t wrk_len = sizeof(fftw_complex)*len*2;
	  printf("allocate of size %d\n",wrk_len);
	  out = (fftw_complex *) fftw_malloc(wrk_len);

	  if( out == NULL ){
		  perror("malloc failed for workspace");
		  exit(2);
	  }
	  zapArray(len*2,out);
      printf("done zap\n");
	  int *grid = makeProcessorGrid(MPI_COMM_WORLD,3,L);

	  int ndim=3;
	  printf("%d: initial grid=[%d,%d,%d]\n",me,grid[0],grid[1],grid[2]);
	  DimDecomp dims[3];
	  for(i=0;i<3;i++){
		  dims[i] = makeBlockDecomp(size,grid[i]);
	  }
      Decomp start = makePackedDecomp(MPI_COMM_WORLD,3,dims);

      printf("%d: made initial decomp\n",me);


      int slab_B[ndim];
      slab_B[0]=1;
      slab_B[1]=size;
      slab_B[2]=size;
      SplitPlan slab_plan = makeParallelPlan(start, slab_B,in,2*len*FUDGE,out);
      if( slab_plan == NULL ){
         raise_error(MPI_COMM_WORLD,"Cannot make slab plan\n");
      }
      SplitPlan auto_plan = autotuneParallelPlan(start,in,2*len*FUDGE,out);
     if( auto_plan == NULL ){
         raise_error(MPI_COMM_WORLD,"Cannot make auto_plan\n");
      }
      printf("Auto plan is: A[0]=%d A[1]=%d A[2]=%d\n",auto_plan->A_factors[0],auto_plan->A_factors[1],auto_plan->A_factors[2]);

      // Time the 2 options.
      int trial;

      double worst_auto=0.0, worst_slab=0.0;
      double best_auto=1000000000.0, best_slab=1000000000.0;

      for(trial=0;trial<TRIAL;trial++){
    	  MPI_Barrier(MPI_COMM_WORLD);
    	  tstart = TIMER();
    	  for(i=0;i<REPEAT;i++){
    		  runSplitPlan(slab_plan,FFTW_FORWARD);
    	  }
    	  tstop = TIMER();
    	  double elapsed;
    	  tstop=tstop-tstart;
    	  MPI_Allreduce(&tstop,&elapsed,1,MPI_DOUBLE,MPI_MAX,start->comm);
    	  if( elapsed > worst_slab ){
    		  worst_slab=elapsed;
    	  }
    	  if( elapsed < best_slab ){
    		  best_slab=elapsed;
    	  }
    	  MPI_Barrier(MPI_COMM_WORLD);
    	  tstart = TIMER();
    	  for(i=0;i<REPEAT;i++){
    		  runSplitPlan(auto_plan,FFTW_FORWARD);
    	  }
    	  tstop = TIMER();
    	  tstop=tstop-tstart;
    	  MPI_Allreduce(&tstop,&elapsed,1,MPI_DOUBLE,MPI_MAX,start->comm);
    	  if( elapsed > worst_auto ){
    		  worst_auto=elapsed;
    	  }
    	  if( elapsed < best_auto ){
    		  best_auto=elapsed;
    	  }
      }


      if( me == 0 ){
    	  snprintf(name,80,"compare_results_%d.txt",size);
    	  FILE *res = fopen(name,"a+");
    	  fprintf(res,"npe %d initial-grid %d %d %d slab-A %d %d %d slab-grid-a %d %d %d slab-grid-b %d %d %d slab time %.2e %.2e auto-A %d %d %d auto-grid-A %d %d %d auto-grid-B %d %d %d auto-time %.2e %.2e\n",npe,
    			  grid[0],grid[1],grid[2],
    			  slab_plan->A_factors[0],slab_plan->A_factors[1],slab_plan->A_factors[2],
				  slab_plan->A->dims[0]->decomp->proc_size,
				  slab_plan->A->dims[1]->decomp->proc_size,
				  slab_plan->A->dims[2]->decomp->proc_size,
				  slab_plan->B->dims[0]->decomp->proc_size,
				  slab_plan->B->dims[1]->decomp->proc_size,
				  slab_plan->B->dims[2]->decomp->proc_size,
    			  worst_slab/((double)REPEAT),best_slab/((double)REPEAT),
				  auto_plan->A_factors[0],auto_plan->A_factors[1],auto_plan->A_factors[2],
				  auto_plan->A->dims[0]->decomp->proc_size,
				  auto_plan->A->dims[1]->decomp->proc_size,
				  auto_plan->A->dims[2]->decomp->proc_size,
		  auto_plan->B ? auto_plan->B->dims[0]->decomp->proc_size : 0,
		  auto_plan->B ? auto_plan->B->dims[1]->decomp->proc_size : 0,
		  auto_plan->B ? auto_plan->B->dims[2]->decomp->proc_size : 0,
				  worst_auto/((double)REPEAT),best_auto/((double)REPEAT)
    	  );
    	  fclose(res);
      }
#ifndef NO_SAVE
      sprintf(name,"parallel_split_%d-%d-%d_%dl_%dp.dat",auto_plan->A_factors[0],auto_plan->A_factors[1],auto_plan->A_factors[2],size,npe);
      writeFile(size,MPI_COMM_WORLD,base,name,getTransformed(auto_plan),out);
#endif
      freeSplitPlan(slab_plan);
      freeSplitPlan(auto_plan);
	  printf("%d: go for finalize\n",me);
	  error = MPI_Finalize();

	 /*  printf("hello world\n"); */

	   exit(0);
}
