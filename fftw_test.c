#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <fftw3.h>
#ifndef USE_RESHAPE
#include <fftw3-mpi.h>
#endif
#include "reshape.h"
#include "local_decomp.h"
#include "block_decomp.h"
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
  MPI_Aint lb,extent,true_lb,true_extent;
  int type_size;
  int me;
  MPI_Comm_rank(comm,&me);

  ierror=MPI_File_open(comm,name,MPI_MODE_WRONLY|MPI_MODE_CREATE,MPI_INFO_NULL,&file);
  my_check_error(ierror,"File open");
  makeIOType(&filetype,type,d);

  MPI_Type_get_extent(filetype,&lb, &extent);
  MPI_Type_get_true_extent(filetype,&true_lb,&true_extent);
  MPI_Type_size(filetype,&type_size);
  printf("%d: lb=%d extent=%d true_lb=%d true_extent=%d max=%d size=%d\n",me,lb,extent,true_lb,true_extent,true_lb+true_extent,type_size);

  ierror=MPI_File_set_view(file,disp, type,filetype,"native",MPI_INFO_NULL);
  my_check_error(ierror,"File_set_view");
  int nwrite=0;
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
						  //printf("write [%d,%d,%d]->%d\n",k,j,i,pos);
						  ierror=MPI_File_write(file,dat+pos,1,type,MPI_STATUS_IGNORE);
						  nwrite++;
						  my_check_error(ierror,"File write");

					  }
				  }
			  }
		  }
	  }
  }
  MPI_File_close(&file);
  printf("%d: nwrite=%d\n",me,nwrite);
}
#endif
void setArray(int size,int rank,int len,Decomp d, fftw_complex *dat){
  int i,j,k;
  int ierror;

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
							  if(i==1 && j== 0 && k== 0){
								  // single point
								  val=1;
							  }
#else
							  int val = i + size *( j + size *k);
#endif
							  dat[pos][0] = (double) val;
							  dat[pos][1] = 0.0;
						  }
					  }
				  }
			  }
		  }
	  }
  }
}
int checkArray(int size,int len,Decomp d, fftw_complex *dat){
  int i,j,k;
  int ierror=0;

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
						  if( ((int)dat[pos][0]) !=  val ||
						      ((int)dat[pos][1]) != 0 ){
							  fprintf(stderr, "Incorrect array value [%d,%d,%d] (%g,%g)\n",k,j,i,dat[pos][0],dat[pos][1]);
							  ierror++;
						  }
						  }
					  }
				  }
			  }
		  }
	  }
  }
  return ierror;
}
void zapArray(int len, fftw_complex *p){
	int i;
	for(i=0;i<len;i++){
		p[i][0]=-99.0;
		p[i][1]=-33.0;
	}
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
int main(int argc, char **argv){

	 int error;
	  int npe, me, i,j;
	  int provided, threads_ok;
	  double tstop, tstart;
	  ptrdiff_t L[3];
	  ptrdiff_t block;
	  char name[80];
	  fftw_complex *in;
	  fftw_complex *out;
	  int size;
#ifdef USE_THREADS
	  error = MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
	  threads_ok = provided >= MPI_THREAD_FUNNELED;
#else
          error = MPI_Init(&argc,&argv);
          threads_ok = 0;
#endif
	  if( argc >= 2 ){
	    size=atoi(argv[1]);
	  }
	  if (threads_ok) threads_ok = fftw_init_threads();
#ifndef USE_RESHAPE
	  fftw_mpi_init();
#endif
	  error = MPI_Comm_size(MPI_COMM_WORLD, &npe);

	  error = MPI_Comm_rank(MPI_COMM_WORLD, &me);

	  sprintf(name,"output.%d",me);
	  freopen(name,"w",stdout);
	  setlinebuf(stdout);

	  printf("hello world from %d of %d\n",me,npe);
	  MPI_Datatype base;

	  // make fftw_complex datatype
	  MPI_Type_contiguous(2,MPI_DOUBLE,&base);
	  MPI_Type_commit(&base);
	  //if (threads_ok) fftw_plan_with_nthreads(nthreads);
	  block = (size +(npe-1))/npe;
	  L[0]=size;
	  L[1]=size;
	  L[2]=size;
	  ptrdiff_t n0,n0_start;
#ifdef USE_RESHAPE
	  ptrdiff_t len = (size*size*block);
	  n0=block;
#else
	  ptrdiff_t len =fftw_mpi_local_size(3, L, MPI_COMM_WORLD,
              &n0, &n0_start);
#endif

	  // old version compat
	  //in = fftw_alloc_complex(len);
	  //out = fftw_alloc_complex(len);
	  in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex)*len);
	  zapArray(len,in);
	  out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex)*len);
	  zapArray(len,in);

	  AxisDecomp start_decomp[3];
	  start_decomp[0] = makeAxisDecomp(MPI_COMM_WORLD,makeBlockDecomp(size,npe),size*size,1);
	  start_decomp[1] = makeAxisDecomp(MPI_COMM_WORLD,makeLocalDecomp(size),size,1);
	  start_decomp[2] = makeAxisDecomp(MPI_COMM_WORLD,makeLocalDecomp(size),1,1);
	  Decomp start = makeDecomp(3,start_decomp);

	  fftw_iodim stage1[3];
	  stage1[0].n=n0;
	  stage1[0].is=size*size;
	  stage1[0].os=size*size;
          stage1[1].n=size;
	  stage1[1].is=size;
	  stage1[1].os=size;
	  stage1[2].n=size;
	  stage1[2].is=1;
	  stage1[2].os=1;

	  AxisDecomp end_decomp[3];
	  end_decomp[0] = makeAxisDecomp(MPI_COMM_WORLD,makeLocalDecomp(size),size,1);
	  end_decomp[1] = makeAxisDecomp(MPI_COMM_WORLD,makeBlockDecomp(size,npe),size*size,1);
	  end_decomp[2] = makeAxisDecomp(MPI_COMM_WORLD,makeLocalDecomp(size),1,1);
	  Decomp end = makeDecomp(3,end_decomp);

	  fftw_iodim stage2[3];
	  stage2[0].n=size;
	  stage2[0].is=size;
	  stage2[0].os=size;
	  stage2[1].n=n0;
	  stage2[1].is=size*size;
	  stage2[1].os=size*size;
	  stage2[2].n=size;
	  stage2[2].is=1;
	  stage2[2].os=1;



#ifdef USE_RESHAPE
	  // fft in dim 1,2 result in out
      fftw_plan stage1_plan = fftw_plan_guru_dft(2,stage1+1,1,stage1,in,out,FFTW_FORWARD,FFTW_MEASURE);
      fftw_plan stage1_back_plan = fftw_plan_guru_dft(2,stage1+1,1,stage1,out,in,FFTW_BACKWARD,FFTW_MEASURE);
	  // fft in dim 0 result in out
      fftw_plan stage2_plan = fftw_plan_guru_dft(1,stage2,2,stage2+1,in,out,FFTW_FORWARD,FFTW_MEASURE);
      fftw_plan stage2_back_plan = fftw_plan_guru_dft(1,stage2,2,stage2+1,out,in,FFTW_BACKWARD,FFTW_MEASURE);

	  ReshapePlan comm_plan = makeReshapePlan(MPI_COMM_WORLD,base,start,end);
	  ReshapePlan comm_back_plan = makeReshapePlan(MPI_COMM_WORLD,base,end,start);

	  if( checkReshapePlanBounds(comm_plan,sizeof(fftw_complex),len,len)){
	    MPI_Abort(MPI_COMM_WORLD,1);
	  }

#else

	  fftw_plan plan = fftw_mpi_plan_many_dft(3, L,
              1, block, block,
              in, out,
              MPI_COMM_WORLD, FFTW_FORWARD,FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT);	 
          fftw_plan back_plan = fftw_mpi_plan_many_dft(3, L,
              1, block, block,
	      out,in,
              MPI_COMM_WORLD, FFTW_BACKWARD,FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN);
#endif

#ifdef USE_IO
	  sprintf(name,"initial_%dl.dat",size);
	  readFile(size,MPI_COMM_WORLD,base,name,start,in);
	  sprintf(name,"initial_parallel_%dl_%dp.dat",size,npe);
	  writeFile(size,MPI_COMM_WORLD,base,name,start,in);
#else
	  setArray(size,me,len,start,in);
	  checkArray(size,len,start,in);
#endif
	  tstart=MPI_Wtime();
#ifdef USE_RESHAPE
	  fftw_execute(stage1_plan);                        // in->out
	  runReshapePlan(MPI_COMM_WORLD,comm_plan,out,in);  // out->in
	  fftw_execute(stage2_plan);                        // in->out
      tstop=MPI_Wtime();


#ifdef USE_IO
	  sprintf(name,"parallel_reshape_%dl_%dp.dat",size,npe);
	  writeFile(size,MPI_COMM_WORLD,base,name,end,out);
#endif
#else
	  fftw_mpi_execute_dft(plan,in,out);
	  tstop=MPI_Wtime();
#ifdef USE_IO
	  sprintf(name,"parallel_fftw_%dl_%dp.dat",size,npe);
	  writeFile(size,MPI_COMM_WORLD,base,name,end,out);
#endif
	  fftw_mpi_cleanup();
#endif
      printf("elapsed time = %e\n",tstop-tstart);

	  error = MPI_Finalize();

	 /*  printf("hello world\n"); */

	   exit(0);
}
