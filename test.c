#include <stdio.h>
#include<stdlib.h>
#include <mpi.h>
#include "reshape.h"
#include "single_decomp.h"
#include "local_decomp.h"
#include "block_decomp.h"
#include "rotate_decomp.h"
#define SIZE 10

/**  routine to print out the local portion of a distributed array.
 * and compare with expected values.
 * note this loops over global coordinares in order so coordinate rotations
 * are not always immediatly obvious
 */
int printSlab(Decomp d, int *dat){
  int i,j;
  int errors=0;
  for(i=0;i<SIZE;i++){
    if( isLocal(d->dims[1],i)){
      for(j=0;j<SIZE;j++){
	if( isLocal(d->dims[0],j)){
	    int coord[2];
	    coord[0]=j;
	    coord[1]=i;
	    int pos = getOffset(d,coord);
	    int expect = j+(SIZE*i);
	    int value = dat[pos];
	    char *tag=" ";
	    if( expect != value ){
	      tag="*";
	      errors++;
	    }
	    printf("[%d,%d]{%d}=%d%s ",j,i,pos,value,tag);
	}
      }
      printf("\n");
    }

  }
  return errors;

}

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
void writeFile(MPI_Comm comm,char *name, Decomp d, int *dat){
  MPI_File file;
  MPI_Datatype filetype;
  MPI_Offset disp=0;
  int i,j;
  int ierror;

  ierror=MPI_File_open(comm,name,MPI_MODE_WRONLY|MPI_MODE_CREATE,MPI_INFO_NULL,&file);
  my_check_error(ierror,"File open");
  makeIOType(&filetype,MPI_INT,d);

  ierror=MPI_File_set_view(file,disp, MPI_INT,filetype,"native",MPI_INFO_NULL);
  my_check_error(ierror,"File_set_view");
  for(j=0;j<SIZE;j++){
    if( isLocal(d->dims[1],j)){
      for(i=0;i<SIZE;i++){
	if( isLocal(d->dims[0],i)){
	    int coord[2];
	    coord[0]=i;
	    coord[1]=j;
	    int pos = getOffset(d,coord);
	    ierror=MPI_File_write(file,dat+pos,1,MPI_INT,MPI_STATUS_IGNORE);
	    my_check_error(ierror,"File write");

	}
      }
    }
  }
  MPI_File_close(&file);


}
#endif
void main(int argc , char **argv){


  int error;
  int npe, me, i,j;

  error = MPI_Init(&argc,&argv);
 
  error = MPI_Comm_size(MPI_COMM_WORLD, &npe);
 
  error = MPI_Comm_rank(MPI_COMM_WORLD, &me);
 
  for(i=0;i<npe;i++){
    if( me == i ){
      printf("hello world from %d of %d\n",me,npe);
    }
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  int *orig=NULL;
  int *dest=NULL;
  int *d2=NULL;
  // Start with a  2D array that only exists on processor rank 0
  //
  //
  AxisDecomp orig_axis_decomp[2];
  orig_axis_decomp[0] = makeAxisDecomp(MPI_COMM_WORLD,makeSingleDecomp(SIZE,0,1),1,1);
  orig_axis_decomp[1] = makeAxisDecomp(MPI_COMM_WORLD,makeSingleDecomp(SIZE,0,npe),SIZE,1);
  Decomp orig_decomp = makeDecomp(2,orig_axis_decomp);

  if( me == 0 ){
    orig=(int *) malloc( SIZE*SIZE*sizeof(int));
    for(i=0;i<(SIZE*SIZE);i++){
      orig[i] = i;
    }
#ifdef USE_IO
    writeFile(MPI_COMM_SELF,"orig.dat",orig_decomp,orig);
#endif
  }
  dest = (int *) malloc( SIZE *((SIZE+npe-1)/npe)*sizeof(int));
  for(i=0;i<(SIZE*((SIZE+npe-1)/npe));i++){
    dest[i] = - me;
  }
  d2 = (int *) malloc( SIZE *((SIZE+npe-1)/npe)*sizeof(int));
  for(i=0;i<(SIZE*((SIZE+npe-1)/npe));i++){
    d2[i] = - me;
  }


  // Second decomposition.
  // This is a local in dim-0 but block in dim-1. The coordinates are also rotated cyclically by (2,1)
  AxisDecomp new_axis_decomp[2];
  new_axis_decomp[0] = makeAxisDecomp(MPI_COMM_WORLD,makeRotatedDecomp(2,makeLocalDecomp(SIZE)),1,1);
  new_axis_decomp[1] = makeAxisDecomp(MPI_COMM_WORLD,makeRotatedDecomp(1,makeBlockDecomp(SIZE,npe)),SIZE,1);
  Decomp new_decomp = makeDecomp(2,new_axis_decomp);

  // debugging check to make sure ranks are consistent
  int my_orig_rank = myRank(orig_decomp);
  if( my_orig_rank != me ){
    fprintf(stderr,"Original ranks do not match %d!=%d\n",my_orig_rank,me);
    MPI_Abort(MPI_COMM_WORLD,33);
  }
  int my_new_rank = myRank(new_decomp);
  if( my_new_rank != me ){
    fprintf(stderr,"New ranks do not match %d!=%d\n",my_new_rank,me);
    MPI_Abort(MPI_COMM_WORLD,33);
  }
  printf("Ranks %d:%d:%d\n",me,my_orig_rank,my_new_rank);


  // make a plan to switch between the 2 distributions
  ReshapePlan plan = makeReshapePlan(MPI_COMM_WORLD,MPI_INT,orig_decomp,new_decomp);

  // run the plan.
  runReshapePlan(MPI_COMM_WORLD,plan,orig,dest);

  int errors,sum_error;
  for(i=0;i<npe;i++){
    if( me == i ){
      printf("hello world from %d of %d\n",me,npe);
      errors=printSlab(new_decomp,dest);
      printf("---------------\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Allreduce(&errors,&sum_error,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if( sum_error > 0 ){
    MPI_Abort(MPI_COMM_WORLD,99);
  }
#ifdef USE_IO
  writeFile(MPI_COMM_WORLD,"new.dat",new_decomp,dest);
#endif
  // if we are a multiple of 2 cpus do some more tests
  if( npe > 1 && npe%2==0){

    // a 2,npe/2 block decomposition rotated by (3,3)
    AxisDecomp d2_axis_decomp[2];
    d2_axis_decomp[0] = makeAxisDecomp(MPI_COMM_WORLD,makeRotatedDecomp(3,makeBlockDecomp(SIZE,2)),1,1);
    d2_axis_decomp[1] = makeAxisDecomp(MPI_COMM_WORLD,makeRotatedDecomp(3,makeBlockDecomp(SIZE,npe/2)),SIZE/2,2);
    Decomp d2_decomp = makeDecomp(2,d2_axis_decomp);
    ReshapePlan plan2a = makeReshapePlan(MPI_COMM_WORLD,MPI_INT,orig_decomp,d2_decomp);
    runReshapePlan(MPI_COMM_WORLD,plan2a,orig,d2);
    for(i=0;i<npe;i++){
      if( me == i ){
	errors=printSlab(d2_decomp,d2);
	printf("---------------\n");
	fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Allreduce(&errors,&sum_error,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    if( sum_error > 0 ){
      MPI_Abort(MPI_COMM_WORLD,99);
    }
#ifdef USE_IO
    writeFile(MPI_COMM_WORLD,"d2.dat",d2_decomp,d2);
#endif
    for(i=0;i<(SIZE*((SIZE+npe-1)/npe));i++){
      d2[i] = - (me*1000);
    }
    ReshapePlan plan2b = makeReshapePlan(MPI_COMM_WORLD,MPI_INT,new_decomp,d2_decomp);
    runReshapePlan(MPI_COMM_WORLD,plan2b,dest,d2);
    for(i=0;i<npe;i++){
      if( me == i ){
	errors=printSlab(d2_decomp,d2);
	printf("---------------\n");
	fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Allreduce(&errors,&sum_error,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    if( sum_error > 0 ){
      MPI_Abort(MPI_COMM_WORLD,99);
    }
#ifdef USE_IO
    writeFile(MPI_COMM_WORLD,"d2b.dat",d2_decomp,d2);
#endif

  }

  error = MPI_Finalize();

/*  printf("hello world\n"); */
 
  exit(0);

}
