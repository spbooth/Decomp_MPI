/** Test case to demonstrate a possible bug in MPI-IO
 *
 */
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>


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

int makeIOType(int size,MPI_Datatype *result,MPI_Aint extent,MPI_Datatype src,int stride){
  int me, npe;
  int counts[2];
  int offsets[2];
  MPI_Datatype extended_type;
  MPI_Datatype tmp, tmp2;
  MPI_Comm_rank(MPI_COMM_WORLD,&me);
  MPI_Comm_size(MPI_COMM_WORLD,&npe);
  //first the full dim
  MPI_Type_create_resized(src,0,extent*stride,&extended_type);

  counts[0]=size;
  offsets[0]=0;
  MPI_Type_indexed(1,counts,offsets,extended_type,&tmp);
  MPI_Type_free(&extended_type);
  stride = stride*size;

  // now the half dim
  MPI_Type_create_resized(tmp,0,extent*stride,&extended_type);
  counts[0]=size/npe;
  offsets[0]=(size/npe)*me;
  MPI_Type_indexed(1,counts,offsets,extended_type,&tmp);
  MPI_Type_free(&extended_type);
  stride= stride*size;

  // now the final dim
  MPI_Type_create_resized(tmp,0,extent*stride,&extended_type);
   counts[0]=size;
   offsets[0]=0;
   MPI_Type_indexed(1,counts,offsets,extended_type,&tmp);
   MPI_Type_free(&extended_type);
   stride = stride*size;

   MPI_Type_create_resized(tmp,0,extent*stride,result);
   MPI_Type_commit(result);
}
void writeFile(int size,MPI_Comm comm,MPI_Datatype type,char *name){
  MPI_File file;
  MPI_Datatype filetype;
  MPI_Offset disp=0;
  _Complex double dat;
  int i,j,k;
  int ierror;
  MPI_Aint lb,extent,true_lb,true_extent;
  int type_size;
  int me, npe;
  MPI_Comm_rank(comm,&me);
  MPI_Comm_size(comm,&npe);
  MPI_Type_get_extent(type,&lb, &extent);
  ierror=MPI_File_open(comm,name,MPI_MODE_WRONLY|MPI_MODE_CREATE,MPI_INFO_NULL,&file);
  my_check_error(ierror,"File open");
  makeIOType(size,&filetype,extent,type,1);

  MPI_Type_get_extent(filetype,&lb, &extent);
  MPI_Type_get_true_extent(filetype,&true_lb,&true_extent);
  MPI_Type_size(filetype,&type_size);
  printf("%d: lb=%d extent=%d true_lb=%d true_extent=%d max=%d size=%d\n",me,lb,extent,true_lb,true_extent,true_lb+true_extent,type_size);

  ierror=MPI_File_set_view(file,disp, type,filetype,"native",MPI_INFO_NULL);
  my_check_error(ierror,"File_set_view");
  int nwrite=0;
  for(k=0;k<size;k++){
		  for(j=0;j<(size/npe);j++){

				  for(i=0;i<size;i++){
					  	  dat = ((double) i+j+k) + 0.0 * I;
						  //printf("write [%d,%d,%d]->%d\n",k,j,i,pos);
						  ierror=MPI_File_write(file,&dat,1,type,MPI_STATUS_IGNORE);
						  nwrite++;
						  my_check_error(ierror,"File write");


				  }

		  }

  }
  MPI_File_close(&file);
  printf("%d: nwrite=%d\n",me,nwrite);
}
int main(int argc, char **argv){
	 int error;
	 int size=16;
	  int npe, me, i,j;
		  int provided, threads_ok;

		  char name[80];

		  error = MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
		  threads_ok = provided >= MPI_THREAD_FUNNELED;


		  error = MPI_Comm_size(MPI_COMM_WORLD, &npe);

		  error = MPI_Comm_rank(MPI_COMM_WORLD, &me);

		  if( (size/npe)*npe != size ){
			  fprintf(stderr,"processor count must be factor of %d\n",size);
			  MPI_Abort(MPI_COMM_WORLD,787);
		  }
		  sprintf(name,"output.%d",me);
		  freopen(name,"w",stdout);
		  setlinebuf(stdout);

		  printf("hello world from %d of %d\n",me,npe);
		  MPI_Datatype base;

		  // make fftw_complex datatype
		  MPI_Type_contiguous(2,MPI_DOUBLE,&base);
		  MPI_Type_commit(&base);

		  writeFile(size,MPI_COMM_WORLD,base,"junk.dat");

		  MPI_Finalize();
}
