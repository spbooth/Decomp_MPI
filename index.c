#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "index.h"
#include "logger.h"

Index makeIndex(){
  Index result = (Index) calloc(sizeof(struct index),1);
  result->size=INDEX_INITIAL_SIZE;
  result->count=0;
  result->offsets = (int *) calloc(sizeof(int),INDEX_INITIAL_SIZE);
  result->counts = (int *) calloc(sizeof(int),INDEX_INITIAL_SIZE);
  return result;
}
void freeIndex(Index i){
  bzero(i->offsets,sizeof(int)*i->size);
  free(i->offsets);
  bzero(i->counts,sizeof(int)*i->size);
  free(i->counts);
  bzero(i,sizeof(struct index));
  free(i);
}
void addIndex(Index i,int offset){
  if(i->count >= i->size){
    i->size *=2;
    i->offsets = (int *)realloc(i->offsets,i->size*sizeof(int));
    i->counts = (int *)realloc(i->counts,i->size*sizeof(int));
  }
  if( i->count == 0 ){
    i->count=1;
    i->counts[0]=1;
    i->offsets[0]=offset;
  }else{
    if((i->offsets[i->count-1]+i->counts[i->count-1]) == offset ){
      i->counts[i->count-1]++;
    }else{
      i->counts[i->count]=1;
      i->offsets[i->count]=offset;
      i->count++;
    }
  }
}

void printIndex(FILE *out, Index index){
	int i,j;
	for(i=0;i<index->count;i++){
		for(j=0; j < index->counts[i]; j++){
			fprintf(out,"%d ",j+index->offsets[i]);
		}
	}
}
void logIndex(Index index){
	int i,j;
	for(i=0;i<index->count;i++){
		for(j=0; j < index->counts[i]; j++){
			fragment_logger("%d ",j+index->offsets[i]);
		}
	}
}

