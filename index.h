#ifndef INDEX_H
#define INDEX_H
#include <stdio.h>

/** an Index is a set of offset and count values encoding one dimension of
 * a local data patch.
 */
struct index{
  int size;    // size of internal arrays 
  int count;     // number of entries
  int *offsets; // offset values
  int *counts;  // count values
};

typedef struct index *Index;

#define INDEX_INITIAL_SIZE 16

Index makeIndex();
void freeIndex(Index);
void addIndex(Index i,int offset);
void freeIndexList(Index *list);
void printIndex(FILE *out, Index index);
void logIndex(Index index);
#endif
