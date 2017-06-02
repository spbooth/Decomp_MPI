#include <time.h>
#include "timer.h"

int called=0;
struct timespec base;


double timer(){
	if( called == 0 ){
		called=1;
		clock_gettime(CLOCK_REALTIME,&base);
	}else{
		double result;
		struct timespec now;
		clock_gettime(CLOCK_REALTIME,&now);

		result = now.tv_sec - base.tv_sec;
		result+= (1e-9)*(now.tv_nsec-base.tv_nsec);
		return result;
	}

}
