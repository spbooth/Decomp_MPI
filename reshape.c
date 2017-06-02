#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "reshape.h"
#include "index.h"
#include "typelist.h"
#include <stdarg.h>
#include <mpi.h>
#include "logger.h"


void raise_error(MPI_Comm comm, char *msg, ...){
  va_list args;
  va_start(args,msg);
  char buffer[256];
  int me;
  MPI_Comm_rank(comm,&me);
  snprintf(buffer,256,"%d: Error in reshape:%s\n",me,msg);
  vfprintf(stderr,buffer,args);
  MPI_Abort(comm,555);
}
void freeDimDecomp(DimDecomp dim){

  if( dim->destroy != NULL ){
    dim->destroy(dim);
  }
  bzero(dim,sizeof(DimDecomp));
  free(dim);
}
DimDecomp cloneDimDecomp(DimDecomp dim){
	if( dim->cloner == NULL){
		return NULL;
	}
	return dim->cloner(dim);
}
int sizeDimDecomp(DimDecomp d){
  return d->coord_size;
}

void describeDimDecomp(FILE *out,DimDecomp d){
	fprintf(out,"DimDecomp(%s) coords=%d procs=%d",d->type,d->coord_size,d->proc_size);
	if( d->printer != NULL){
		d->printer(out,d);
	}
}

AxisDecomp makeAxisDecomp(MPI_Comm comm,DimDecomp d,int stride,int rank_stride){
  AxisDecomp result = (AxisDecomp) malloc(sizeof(struct axis_decomp));
  result->decomp = d;
  result->comm=comm;
  result->memory_stride=stride;
  result->rank_stride=rank_stride;
  int size, error;
  error = MPI_Comm_size(comm,&size); 
  if(error != MPI_SUCCESS){
    raise_error(comm,"Error getting comm size");
    return NULL;
  }
  if( rank_stride < 1 || (rank_stride*d->proc_size) > size ){
    raise_error(comm,"Illegal rank_stride");
  }
  int rank;
  error= MPI_Comm_rank(comm,&rank);
  if( error != MPI_SUCCESS){
    raise_error(comm,"Error getting rank");
    return NULL;
  }
  //printf("%d: in makeAxisDecomp\n",rank);
  result->my_rank_in_dimension = (rank/rank_stride)%d->proc_size;
  //printf("%d: my_rank_in dim = %d\n",rank,result->my_rank_in_dimension);
  result->index = makeIndex();
  int global;
  result->max_local=0;
  result->n_local=0;
  for(global=0;global<d->coord_size;global++){
	 //printf("%d: consider %d\n",rank,global);
     if( isLocal(result,global) ){
    	 result->n_local++;
    	 int local = d->type_offset(d,global);
    	 if( local > result->max_local){
    		 result->max_local=local;
    	 }
    	 //printf("%d: global=%d local=%d\n",rank,global,local);
    	 addIndex(result->index,local);
     }
   }
  //fprintf(stderr,"%d: type=%s dimension-rank=%d rank_stride=%d proc_size=%d\n",rank,d->type,result->my_rank_in_dimension,result->rank_stride,d->proc_size);
  return result;
}
void freeAxisDecomp(AxisDecomp a){
  freeDimDecomp(a->decomp);
  free(a);
}

void describeAxisDecomp(FILE *out,AxisDecomp d){
	fprintf(out,"AxisDecomp: mem_stride=%d rank_stride=%d ",d->memory_stride,d->rank_stride);
	describeDimDecomp(out,d->decomp);

}

int isLocal(AxisDecomp d, int global){
  return d->my_rank_in_dimension ==  d->decomp->rank_offset(d->decomp,global);
}
Index *makeIndexList(AxisDecomp mine,DimDecomp remote){
  int count = remote->proc_size;
  Index *result  = (Index *) calloc(sizeof(Index),count+1);
  int i,global;
  for(i=0;i<count;i++){
    result[i] = makeIndex();
  }
  result[count]=NULL;
  int my_size,remote_size;
  my_size=mine->decomp->coord_size;
  remote_size=remote->coord_size;
  if( my_size != remote_size ){
    raise_error(mine->comm,"non-conformant sizes in makeIndexList %d!=%d",my_size,remote_size);
    return NULL;
  }
  // now loop over global coordinates
  for(global=0;global<my_size;global++){
	//printf("global=%d",global);
    if( isLocal(mine,global) ){
      // rank this coord maps to in the remote decomposition dimension
      int rank=remote->rank_offset(remote,global);
      // local memory offset
      // we don't add in the local memory_stride as we want to
      // generate repeated blocks if we can. This will make
      // the datatype generation harder later but should take less memory.
      int offset=mine->decomp->type_offset(mine->decomp,global);
      //printf(" rank-offest=%d mem-offset=%d",rank,offset);
      addIndex(result[rank],offset);
    }
    //printf("\n");
  }
  return result;
}
Index makeGlobalIndex(AxisDecomp mine){
  Index result = makeIndex();
  int global;
  // now loop over global coordinates
  for(global=0;global<mine->decomp->coord_size;global++){
    if( isLocal(mine,global) ){
      addIndex(result,global);
    }
  }

  return result;
}


void freeIndexList(Index *list){
  Index *p;
  for(p=list; *p != NULL ; p++){
    freeIndex(*p);
  }
  free(list);
}
int expandIOType(MPI_Datatype *result,MPI_Aint extent,MPI_Datatype src,AxisDecomp mine,int stride){
  //int me;
  //MPI_Comm_rank(mine->comm,&me);
  logger(mine->comm,"expandIOType %s extent=%d stride=%d ",mine->decomp->type,extent,stride);

  MPI_Datatype extended_type;
  MPI_Type_create_resized(src,0,extent*stride,&extended_type);
  Index index = makeGlobalIndex(mine);
  logIndex(index);
  fragment_logger("\n");
  MPI_Type_indexed(index->count,index->counts,index->offsets,extended_type,result);
  MPI_Type_free(&extended_type);
  freeIndex(index);
  return stride * mine->decomp->coord_size;
}

Decomp makeDecomp(int ndim,AxisDecomp dims[ndim]){
  int i,j, max, prod=1;
  Decomp result = (Decomp ) malloc(sizeof(struct decomp));
  result->ndim=ndim;
  result->dims=(AxisDecomp *) malloc(ndim * sizeof(AxisDecomp));
  max = 0;
  for(i=0;i<ndim;i++){
	  //printf("dim=%d rank_stride=%d proc_size=%d\n",i,dims[i]->rank_stride,dims[i]->decomp->proc_size);
    result->dims[i]=dims[i];
    max += dims[i]->rank_stride * (dims[i]->decomp->proc_size -1);
    prod *= dims[i]->decomp->proc_size;
  }
  for(i=1;i<ndim;i++){
    if( dims[0]->comm != dims[1]->comm ){
      raise_error(dims[0]->comm,"Different communicators in Decomp");
      return NULL;
    }
  }
  result->comm=dims[0]->comm;
  int size, error;
  error = MPI_Comm_size(result->comm,&size); 
  if(error != MPI_SUCCESS){
    raise_error(result->comm,"Error getting comm size");
    return NULL;
  }
  if( max >= size ){
    raise_error(result->comm,"Maximum processor rank out of range max=%d size=%d",max,size);
    return NULL;
  }
  //printf("max=%d prod=%d size=%d\n",max,prod,size);
  result->proc_used=max+1;
  for(i=0;i<ndim;i++){
    for(j=i+1;j<ndim;j++){
      // allow duplicate with size-1 dims as these may be single/local
      if( dims[i]->decomp->proc_size > 1 && dims[j]->decomp->proc_size > 1
    		  && dims[i]->rank_stride > 1 && dims[j]->rank_stride > 1
    		  && dims[i]->rank_stride == dims[j]->rank_stride){
	raise_error(result->comm,"duplicate rank strides dim=%d size=%d rank_stride=%d, dim=%d size=%d rank_stride=%d",i,dims[i]->decomp->proc_size,dims[i]->rank_stride,j,dims[j]->decomp->proc_size,dims[j]->rank_stride);
	return NULL;
      }
    }
  }
  return result;
}
Decomp makePackedDecomp(MPI_Comm comm,int ndim, DimDecomp dims[ndim]){
	AxisDecomp *axis_dims = malloc(ndim * sizeof(AxisDecomp));
    //int me;
    //MPI_Comm_rank(comm,&me);
    //printf("%d: in makePackedDecomp ndim=%d\n",me,ndim);
    if( axis_dims == NULL){
		raise_error(comm,"malloc failed in makePackedDecomp\n");
		return NULL;
	}
	int i;
	int rank_stride=1;
	int mem_stride=1;
	for(i=ndim-1;i>=0;i--){
		//printf("%d: before axis dim=%d mem_stride=%d rank_stride=%d\n",me,i,mem_stride,rank_stride);
		axis_dims[i] =makeAxisDecomp(comm,dims[i],mem_stride,rank_stride);
		mem_stride *= (axis_dims[i]->max_local+1);
		rank_stride *= dims[i]->proc_size;
		//printf("%d: dim=%d mem_stride=%d rank_stide=%d\n",me,mem_stride,rank_stride);
	}
    Decomp result = makeDecomp(ndim,axis_dims);
    free(axis_dims);
    return result;

}
void freeDecomp(Decomp d){
  int i;
  for(i=0;i<d->ndim;i++){
    freeAxisDecomp(d->dims[i]);
  }
  free(d);
}
void describeDeomp(FILE *out, Decomp d){
	int i;
	fprintf(out,"Decomp ndim=%d procs_used=%d\n",d->ndim,d->proc_used);
	for(i=0;i<d->ndim;i++){
		describeAxisDecomp(out,d->dims[i]);
		fprintf(out,"\n");
	}
}

int compatible(Decomp a, Decomp b){
  if(a->ndim != b->ndim){
    //fprintf(stderr,"Dimensions not compatible %d,%d\n",a->ndim,b->ndim);
    return NOT_COMPATIBLE;
  }
  if( a->comm != b->comm ){
    //fprintf(stderr,"Communicators not compatible\n");
    return NOT_COMPATIBLE;
  }
  int i;
  for(i=0;i<a->ndim;i++){
    int size_a =sizeDimDecomp(a->dims[i]->decomp);
    int size_b = sizeDimDecomp(b->dims[i]->decomp);
    if(size_a != size_b){
      //fprintf(stderr,"size in dim=%d not compatible %d!=%d\n",i,size_a,size_b);
      return NOT_COMPATIBLE;
    }
  }

  return COMPATIBLE;
  
}

int myRank(Decomp d){
  int rank=0;
  int i;

  for(i=0;i<d->ndim;i++){
    int local_rank=d->dims[i]->my_rank_in_dimension;
    int rank_stride =d->dims[i]->rank_stride;
    char *type = d->dims[i]->decomp->type;
    rank += local_rank * rank_stride;
    //fprintf(stderr,"i=%d type=%s rank=%d local_rank=%d rank_stride=%d\n",i,type,rank,local_rank,rank_stride);
  //fflush(stderr);
  }
  return rank;

}
int isCoordLocal(Decomp d, int dim, int pos){
  return isLocal(d->dims[dim],pos);
}
int areCoordsLocal(Decomp d, int *pos){
  int i;
  for(i=0;i<d->ndim;i++){
    if( ! isLocal(d->dims[i],pos[i]) ){
      return 0;
    }
  }
  return 1;
}
int getOffset(Decomp d, int *pos){
  int offset=0;
  int i;
  for(i=0;i<d->ndim;i++){
    offset += (d->dims[i]->memory_stride * d->dims[i]->decomp->type_offset(d->dims[i]->decomp,pos[i]));
  }
  return offset;
}

int extent(Decomp d){
	int size=1;
	int i;
	for(i=0;i<d->ndim;i++){
		size += (d->dims[i]->memory_stride *(d->dims[i]->max_local));
	}
	return size;
}

/** Work out what order to process dimensions. lower entries
 * move faster in the message.
 * @private
 */
void sortOrder(int ndim, int order[ndim], Decomp src, Decomp dest){
  int i,j;
  int tmp;
  // start assuming C order last index fastest.
  for(i=0;i<ndim;i++){
    order[i] = ndim-1-i;
  }
  // now sort by considering pairs
  // assume src ordering is the most important
  for(i=0;i<ndim-1;i++){
    for(j=i+1;j<ndim;j++){
      if(src->dims[order[j]]->memory_stride < src->dims[order[i]]->memory_stride ){
	tmp = order[j];
	order[j] = order[i];
	order[i]=tmp;
      }
    }

  }

}
ReshapePlan makeReshapePlan(MPI_Comm comm,MPI_Datatype base,Decomp src,Decomp dest){
  if( NOT_COMPATIBLE == compatible(src,dest)){
    raise_error(comm,"Decomp not compatible in makeReshapePlan");
    return NULL;
  }
  int ndim=src->ndim;
  int me,npe;
  MPI_Comm_rank(comm,&me);
  MPI_Comm_size(comm,&npe);

  if( src->proc_used < 1 || src->proc_used > npe){
	  raise_error(comm,"bad src proc count %d\n",src->proc_used);
  }
  if( dest->proc_used < 1 || dest->proc_used > npe){
  	  raise_error(comm,"bad dest proc count %d\n",src->proc_used);
    }
  MPI_Aint lb,extent;
  ReshapePlan result = (ReshapePlan) calloc(sizeof(struct reshape_plan),1);
  result->comm=src->comm;
  result->src=src;
  result->dest=dest;
  result->order = (int *) calloc(sizeof(int),ndim);
  MPI_Type_get_extent(base,&lb,&extent);
  // Have to explicitly exclude processors out of range as
  // stride values will still overlap
  if( me < src->proc_used){
	  result->sends=makeTypeEntryList();
	  addTypeEntry(result->sends,0,base);
  }else{
	  result->sends=NULL;
  }
  if( me < dest->proc_used){
	  result->recvs=makeTypeEntryList();
	  addTypeEntry(result->recvs,0,base);
  }else{
	  result->recvs=NULL;
  }
  // Now have initial type in both lists
  int i,j;
  // expand up the type list for each dimension in turn.
  // we can process the dimensions in any order but this might
  // result in very non-local access patterns
  // better to do the "fastest" dimension first.
  sortOrder(ndim,result->order,src,dest);

  for(j=0;j<ndim;j++){
    i=result->order[j];
    //printf("src->proc_used=%d dest->proc_used=%d\n",src->proc_used,dest->proc_used);
    //printf("expand dim %d src_dim_size=%d dest_dim_size=%d\n",i,src->dims[i]->decomp->proc_size,dest->dims[i]->decomp->proc_size);
    if( me < src->proc_used){
    	//printf("Do send list\n");
    	//printf("Initial Send list: "); printTypeEntryList(stdout,result->sends);printf("\n");
    	TypeEntryList new_send = expandList(dest->proc_used,extent,result->sends,src->dims[i],dest->dims[i]);
    	freeTypeEntryList(result->sends);
    	result->sends=new_send;
    	//printf("Send list is now: "); printTypeEntryList(stdout,result->sends);printf("\n");
    //}else{
    	//printf("supressing send expansion as unused\n");
    }
    if( me < dest->proc_used){
    	//printf("do recv list\n");
    	//printf("Initial Recv list: "); printTypeEntryList(stdout,result->recvs);printf("\n");
    	TypeEntryList new_recv = expandList(src->proc_used,extent,result->recvs,dest->dims[i],src->dims[i]);
    	freeTypeEntryList(result->recvs);
    	result->recvs=new_recv;
    	//printf("Recv list is now: "); printTypeEntryList(stdout,result->recvs);printf("\n");
    //}else{
      	//printf("supressing recv expansion as unused\n");
    }
  }
  // check that self sends match self recv
  // should not happen but we may have a bug that can generate this and checking turns this from a hang into
  // an abort
  int self_send=0;
  int self_recv=0;
  if( result->sends != NULL){
	  for(i=0;i<result->sends->count;i++){
		  if( result->sends->list[i].rank == me){
			  self_send++;
		  }
	  }
  }
  if( result->recvs != NULL){
	  for(i=0;i<result->recvs->count;i++){
		  if( result->recvs->list[i].rank == me){
			  self_recv++;
		  }
	  }
  }
  if( self_send != self_recv){
	  raise_error(comm,"Self send-recv don't match %d != %d\n",self_send,self_recv);
	  MPI_Abort(comm,945);
  }
  // now commit the datatypes
  // might want to consider randomising the order to prevent hotspots
  commitTypeEntryList(result->sends);
  commitTypeEntryList(result->recvs);
  return result;
}


void freeReshapePlan(ReshapePlan plan){
    freeTypeEntryList(plan->sends);
    freeTypeEntryList(plan->recvs);
    bzero(plan,sizeof(struct reshape_plan));
    free(plan);
}


#define RESHAPE_TAG 458
void runReshapePlan(MPI_Comm comm,ReshapePlan plan,void *src, void *dest){
  int count = 0;
  if( plan->recvs != NULL){
	  count+=plan->recvs->count;
  }
  if( plan->sends != NULL){
	  count += plan->sends->count;
  }
  int i,pos=0;
  if( count == 0 ){
	  logger(comm,"No participation in this reshape\n");
	  return;
  }
  MPI_Request * requests = (MPI_Request *) calloc(sizeof(MPI_Request),count);
  if( plan->recvs != NULL){
	  for(i=0;i<plan->recvs->count;i++){
		  logger(comm,"recv from %d\n",plan->recvs->list[i].rank);
		  MPI_Irecv(dest,1,plan->recvs->list[i].type,plan->recvs->list[i].rank,RESHAPE_TAG,plan->comm,requests + pos++);
	  }
  }
  if( plan->sends != NULL){
	  for(i=0;i<plan->sends->count;i++){
		  logger(comm," send to %d\n",plan->sends->list[i].rank);
		  MPI_Isend(src,1,plan->sends->list[i].type,plan->sends->list[i].rank,RESHAPE_TAG,plan->comm,requests + pos++);
	  }
  }
  logger(comm,"wait for %d\n",count);
  MPI_Waitall(count,requests,MPI_STATUSES_IGNORE);
#ifndef LEAK
  free(requests);
#endif
}

void makeIOType(MPI_Datatype *result, MPI_Datatype src, Decomp d){
  MPI_Datatype tmp[2];
  int i;
  int last;
  int stride=1;
  MPI_Aint lb,extent;
  MPI_Type_get_extent(src,&lb,&extent);
  MPI_Type_dup(src,&tmp[0]);
  last=0;

  int me;
  MPI_Comm_rank(d->comm,&me);
  if( me < d->proc_used){
	  for(i=d->ndim-1;i>=0;i--){
		  logger(d->comm,"expand IO type dim=%d last=%d stride=%d\n",i,last,stride);
		  stride = expandIOType(&tmp[1-last],extent,tmp[last],d->dims[i],stride);
		  MPI_Type_free(&tmp[last]);
		  last=1-last;
	  }
	  extent=stride*extent;
  }else{
	  extent=0;
  }
  MPI_Type_create_resized(tmp[last],lb,extent,result);
  MPI_Type_commit(result);
  MPI_Type_free(&tmp[last]);

}

int checkReshapePlanBounds(ReshapePlan plan, int type_size, int src_len,int dest_len){
  int send_total=0;
  int recv_total=0;
  int i;
  for(i=0;i<plan->recvs->count;i++){
    int size;
    MPI_Aint lb,extent;
    MPI_Type_size(plan->recvs->list[i].type,&size);
    MPI_Type_get_true_extent(plan->recvs->list[i].type,&lb,&extent);
    recv_total += size;
    if( lb < 0 ){
      fprintf(stderr,"Type recv[%d] has lb=%d below start of array\n",i,lb);
      return 1;
    }
    if( (lb+extent) > (type_size*dest_len) ){
      fprintf(stderr,"Type recv[%d] extends beyond end of array top=%d len=%d\n",i,lb+extent,type_size*dest_len);
      return 1;
    }
  }
  if( recv_total > type_size*dest_len){
    fprintf(stderr,"More data read than fits in array %d\n",recv_total);
    return 1;
  }
  for(i=0;i<plan->sends->count;i++){
    int size;
    MPI_Aint lb,extent;
    MPI_Type_size(plan->sends->list[i].type,&size);
    MPI_Type_get_true_extent(plan->sends->list[i].type,&lb,&extent);
    send_total += size;
    if( lb < 0 ){
      fprintf(stderr,"Type send[%d] has lb=%d below start of array\n",i,lb);
      return 1;
    }
    if( (lb+extent) > (type_size*src_len) ){
      fprintf(stderr,"Type send[%d] extends beyond end of array top=%d len=\n",i,lb+extent,type_size*src_len);
      return 1;
    }
  }
  if( send_total > type_size*src_len){
    fprintf(stderr,"More data sent than fits in array %d\n",send_total);
    return 1;
  }
  return 0;

}
