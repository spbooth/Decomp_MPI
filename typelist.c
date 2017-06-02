#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "reshape.h"
#include "index.h"

TypeEntryList makeTypeEntryList(){
  TypeEntryList result = (TypeEntryList) calloc(sizeof(struct type_entry_list),1);
  result->count=0;
  result->size=INITIAL_TYPELIST_SIZE;
  result->list = (struct type_entry *) calloc(sizeof(struct type_entry),result->size);
  return result;
}
void commitTypeEntryList(TypeEntryList list){
  int i;
  if( list == NULL){
	  return;
  }
  for(i=0;i<list->count;i++){
    MPI_Type_commit(&list->list[i].type);
    int size;
    //    MPI_Aint ub,lb;
    //MPI_Type_size(list->list[i].type,&size);
    //MPI_Type_get_true_extent(list->list[i].type,&lb,&ub);
    //fprintf(stderr,"Type[%d] size=%d lb=%d ub=%d\n",i,size,lb,ub);
  }
}
void  freeTypeEntryList(TypeEntryList list){
  int i;
  if( list == NULL){
	  return;
  }
  for(i=0;i<list->count;i++){
    MPI_Type_free(&list->list[i].type);
  }
  bzero(list->list,list->size*sizeof(struct type_entry));
  free(list->list);
  bzero(list,sizeof(struct type_entry_list));
  free(list);
}

void printTypeEntryList(FILE *out, TypeEntryList list){
	 int i;
	  for(i=0;i<list->count;i++){
		  int size;
		  int extent;
		  MPI_Type_size(list->list[i].type,&size);
		  fprintf(out," [%d](%d)",list->list[i].rank,size);
	  }
}

void addTypeEntry(TypeEntryList list,int rank, MPI_Datatype type){
  if(list->count >= list->size){
	  struct type_entry *tmp=list->list;
	  int i;
	  int old_size = list->size;
	  list->size *=2;
	  list->list = (struct type_entry *) calloc(sizeof(struct type_entry),list->size);
	  for(i=0;i<list->count;i++){
	    list->list[i].rank = tmp[i].rank;
	    MPI_Type_dup(tmp[i].type,&list->list[i].type);
	    MPI_Type_free(&tmp[i].type);
	  }
	  bzero(tmp,old_size*sizeof(struct type_entry));
	  free(tmp);
  }
  list->list[list->count].rank=rank;
  MPI_Type_dup(type,&list->list[list->count].type);
  list->count++;
}


void expandEntry(int max_proc,MPI_Aint extent,TypeEntryList result,struct type_entry *src,AxisDecomp mine,AxisDecomp remote,Index *list){
	int rank;
	int i;
	MPI_Datatype extended_type;
	MPI_Type_create_resized(src->type,0,extent*mine->memory_stride,&extended_type);

	for(rank=0;rank<remote->decomp->proc_size;rank++){

		//printf("rank=%d list[%d]->count=%d my_size=%d remote_size=%d max_proc=%d\n",rank,rank,list[rank]->count,mine->decomp->proc_size,remote->decomp->proc_size,max_proc);
		if(list[rank]->count > 0 ){
			MPI_Datatype index_type;
			if( list[rank]->count == 1 && list[rank]->offsets[0] == 0 ){
				// use contiguous if we can.

				MPI_Type_contiguous(list[rank]->counts[0],extended_type,&index_type);
			}else{

				MPI_Type_indexed(list[rank]->count,list[rank]->counts,list[rank]->offsets,extended_type,&index_type);
			}
			int new_rank = src->rank + (rank * remote->rank_stride);
			if( new_rank < max_proc){
				//printf("Add type to rank-offset %d\n",new_rank);
				addTypeEntry(result, new_rank, index_type);
			//}else{
				//printf("new rank out of range %d>=%d\n",new_rank,max_proc);
			}
			MPI_Type_free(&index_type);
		//}else{
			//printf("skipping\n");
		}
	}
	MPI_Type_free(&extended_type);
}

TypeEntryList expandList(int max_proc,MPI_Aint extent, TypeEntryList list,AxisDecomp mine,AxisDecomp remote){
  //fprintf(stderr,"Expanding list list-size=%d mine=%s remote=%s remote-size=%d\n",list->count,mine->decomp->type,remote->decomp->type,remote->decomp->proc_size);
  TypeEntryList result = makeTypeEntryList();
  Index *index_list = makeIndexList(mine,remote->decomp);
  int i;
  for(i=0;i<list->count;i++){
    expandEntry(max_proc,extent,result,list->list+i,mine,remote,index_list);
  }
  freeIndexList(index_list);
  //fprintf(stderr,"Done expanding result-size=%d\n",result->count);
  return result;
}


