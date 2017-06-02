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
int checkArray(int size,int len,Decomp d, fftw_complex *dat,double norm){
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
							  if( (norm*dat[pos]) !=  ((double)val) + 0.0*I ){
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

	 int error;
	  int npe, me, i,j;
	  int provided, threads_ok;
	  int L[3];
	  char name[80];
	  fftw_complex *in;
	  fftw_complex *out;
	  double tstart, tstop;
	  int slab=0;
	  int size;
#ifdef THREADED
	  error = MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
	  threads_ok = provided >= MPI_THREAD_FUNNELED;
#else
          error = MPI_Init(&argc,&argv);
          threads_ok = 0;
#endif
	  if( argc >= 2 ){
	    size=atoi(argv[1]);
	  }
#ifdef SLAB
	  slab=1;
#endif

	  if (threads_ok) threads_ok = fftw_init_threads();

	  error = MPI_Comm_size(MPI_COMM_WORLD, &npe);

	  error = MPI_Comm_rank(MPI_COMM_WORLD, &me);
#ifdef REDIRECT
	  sprintf(name,"split_output.%d",me);
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

	  ptrdiff_t len = 2*(size*size*size)/npe;

	  // old version compat
	  //in = fftw_alloc_complex(len);
	  //out = fftw_alloc_complex(len);
	  in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex)*len);
	  zapArray(len,in);
	  out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex)*len*2*FUDGE);
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
	  SplitPlan plan = makeSplitPlan(slab,start,in,2*len*FUDGE,out);
#ifdef USE_IO
	  sprintf(name,"initial_%dl.dat",size);
	  readFile(size,MPI_COMM_WORLD,base,name,start,in);
#else
	  printf("%d: Go for set array\n",me);
	  setArray(size,me,len,start,in);
	  printf("%d: go for check\n",me);
	  checkArray(size,len,start,in,1.0);
#endif
#ifndef NO_SAVE
	  sprintf(name,"initial_split_%dl_%dp.dat",size,npe);
	  printf("%d: write to file %s\n",me,name);
	  writeFile(size,MPI_COMM_WORLD,base,name,start,in);
      printf("%d: done write\n",me);
#endif
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      tstart = MPI_Wtime();
	  runSplitPlan(plan,FFTW_FORWARD);
	  tstop=MPI_Wtime();
	  printf("%d: done plan\n",me);
	  printf("%d: elapsed time=%e\n",me,tstop-tstart);
	  Decomp end = getTransformed(plan);



#ifdef SLAB
	  sprintf(name,"slab_timings_%dl.txt",size);
#else
	  sprintf(name,"split_timings_%dl.txt",size);
#endif
	  if( me == 0 ){
		  FILE *timer = fopen(name,"a+");
		  fprintf(timer,"%d %d %e %e %e %e %e %e\n",size,npe,
				  plan->times[ALL],
				  plan->times[RESHAPE_1],
				  plan->times[FFT_1],
				  plan->times[TWIDDLE],
				  plan->times[RESHAPE_2],
				  plan->times[FFT_2]);
		  fclose(timer);
	  }
#ifndef NO_SAVE
	  sprintf(name,"parallel_split_%dl_%dp.dat",size,npe);
	  writeFile(size,MPI_COMM_WORLD,base,name,end,out);
#endif

	  runSplitPlan(plan,FFTW_BACKWARD);
	  printf("%d: go for check\n",me);
	  checkArray(size,len,start,in,(1.0/(size*size*size)));
	  freeSplitPlan(plan);


	  printf("%d: go for finalize\n",me);
	  error = MPI_Finalize();

	 /*  printf("hello world\n"); */

	   exit(0);
}
