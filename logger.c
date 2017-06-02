#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "logger.h"

void logger(MPI_Comm comm, char *msg, ...){
#ifdef DEBUG
  va_list args;
  va_start(args,msg);
  char buffer[256];
  int me;
  if( ((int)comm) == 0){
	  me = 0; // for test debugging
  }else{
	  MPI_Comm_rank(comm,&me);
  }
  snprintf(buffer,256,"%d: %s",me,msg);
  vfprintf(stdout,buffer,args);
#endif
}
void fragment_logger(char *msg, ...){
#ifdef DEBUG
  va_list args;
  va_start(args,msg);
  vfprintf(stdout,msg,args);
#endif
}
