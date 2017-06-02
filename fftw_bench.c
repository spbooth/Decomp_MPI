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
  

  ierror=MPI_File_open(comm,name,MPI_MODE_WRONLY|MPI_MODE_CREATE,MPI_INFO_NULL,&file);
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
						  //printf("write [%d,%d,%d]->%d\n",k,j,i,pos);
						  ierror=MPI_File_write(file,dat+pos,1,type,MPI_STATUS_IGNORE);
						  my_check_error(ierror,"File write");

					  }
				  }
			  }
		  }
	  }
  }
  MPI_File_close(&file);


}
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
void initArray(int size,Decomp d, fftw_complex *dat){
	int i,j,k;


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
							if( k==0 && j == 0 && i == 0 ){
								dat[pos][0]=1.0;
								dat[pos][1]=0.0;
							}else{
								dat[pos][0]=0.0;
								dat[pos][1]=0.0;
							}

						}
					}
				}
			}
		}
	}



}
#define REPEAT 10
#define TRIAL 5 

int main(int argc, char **argv){

	 int error;
	  int npe, me, i,j;
	  int provided, threads_ok;
	  int trial;
	  int nthreads=0;
	  int size;
	  char name[80];
	  FILE * res;
	  ptrdiff_t L[3];
	  ptrdiff_t block;
	  fftw_complex *in;
	  fftw_complex *out;
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
#ifdef THREADED
	  if (threads_ok) threads_ok = fftw_init_threads();
#ifndef USE_RESHAPE
	  fftw_mpi_init();
#endif
	  if( argc >= 3 ){
	    nthreads=atoi(argv[2]);
	  }
	  fftw_plan_with_nthreads(nthreads);
#endif



	  error = MPI_Comm_size(MPI_COMM_WORLD, &npe);

	  error = MPI_Comm_rank(MPI_COMM_WORLD, &me);
	  printf("hello world from %d of %d\n",me,npe);
	  if( me == 0 ){
#ifdef USE_RESHAPE
		sprintf(name,"fftw_reshape_%dt_%dp_%dl.out",nthreads,npe,size);
#else
		sprintf(name,"fftw_mpi_%dt_%dp_%dl.out",nthreads,npe,size);
#endif
		res = fopen(name,"w");
		setlinebuf(res);
	  }


	  MPI_Datatype base;

	  // make fftw_complex datatype
	  MPI_Type_contiguous(2,MPI_DOUBLE,&base);

	  //if (threads_ok) fftw_plan_with_nthreads(nthreads);
	  block = (size +(npe-1))/npe;
	  L[0]=size;
	  L[1]=size;
	  L[2]=size;
	  int Ltot= L[0]*L[1]*L[2];
	  int Ptot= npe;
	  int Lloc=Ltot/Ptot;
	  double scale=1.0/((double)Ltot);

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
	  out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex)*len);


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
	  initArray(size,start,in);

	  double tstart,tstop;
	  double worst=0.0;
	  double best=1000000000.0;
	  MPI_Barrier(MPI_COMM_WORLD);
	  for(trial=0;trial<TRIAL;trial++){
	    tstart = MPI_Wtime();
	    for(i=0;i<REPEAT;i++){
#ifdef USE_RESHAPE
	      fftw_execute(stage1_plan);                           //in->out
	      runReshapePlan(MPI_COMM_WORLD,comm_plan,out,in);     //out->in
	      fftw_execute(stage2_plan);                          //in->out

	      fftw_execute(stage2_back_plan);                     //out->in
	      runReshapePlan(MPI_COMM_WORLD,comm_back_plan,out,in); //in->out
	      fftw_execute(stage1_back_plan);                       //out->in
#else
	      fftw_mpi_execute_dft(plan,in,out);               //in->out
	      fftw_mpi_execute_dft(back_plan,out,in);          //out->in
#endif
	      for(j=0; j< len ; j++){
		in[j][0] *= scale;
		in[j][1] *= scale;
	      }
	    }
	    tstop = MPI_Wtime();
	    double elapsed=tstop-tstart;
	    if( elapsed > worst ){
	      worst=elapsed;
	    }
	    if( elapsed < best ){
	      best=elapsed;
	    }
	  }
	  if( me == 0 ){
	    fprintf(res,"%d %d %d %lf %lf\n",L[0],L[1],L[2],best/(double)REPEAT,worst/(double)REPEAT);
	    fflush(res);
	  }


#ifndef USE_RESHAPE
	  fftw_mpi_cleanup();
#endif
	  error = MPI_Finalize();

	 /*  printf("hello world\n"); */

	   exit(0);
}
