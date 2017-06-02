#ifndef SPLIT_FFT
#define SPLIT_FFT
#include <complex.h>
#include "reshape.h"
#include <fftw3.h>
//
// Routines to implement a 2-stage split fft
//

struct split_fft_plan{
	MPI_Comm comm;
	int *A_factors;
	Decomp A;
	Decomp B;
	Decomp B_flipped; // B decomp with factors reversed
	fftw_complex *src;
	fftw_complex *work;
	int extent;  // number of elements to hold either an A or a B decomp
	fftw_plan plan_a;
	fftw_plan plan_b;
	fftw_plan plan_a_back;
	fftw_plan plan_b_back;
	ReshapePlan start_to_A;
	ReshapePlan A_to_start;
	_Complex double **twiddle;
	ReshapePlan A_to_B;
	ReshapePlan B_to_A;
	double *times;
};

// timer fields
enum Timers {
  RESHAPE_1=0,
  FFT_1,
  TWIDDLE,
  RESHAPE_2,
  FFT_2,
  ALL,
  NUM_TIMERS
};

enum SearchFlags{
	NO_FLAGS 			   = 0x00,
	PREFER_LARGE_FFT = 0x01,
	PREFER_STAGE1	= 0x02,
	FORCE_USE_FIRST = 0x04
};
typedef struct split_fft_plan *SplitPlan;

typedef void (*data_visitor)(int ndim, int *global_position, _Complex double *dat );
void visitData(Decomp d, data_visitor vis, _Complex double *dat);
SplitPlan buildParallelPlan(int flags, int use_first, int use_second,Decomp start, int ndim, int A[ndim], int B[ndim], int L[ndim],int a_grid[ndim], int b_grid[ndim],fftw_complex *dat,int length, fftw_complex workspace[length]);
SplitPlan proposeParallelPlan(int flags,Decomp start,int *initial_factor, fftw_complex *dat,int length, fftw_complex workspace[length]);
SplitPlan autotuneParallelPlan(Decomp start, fftw_complex *dat,int length, fftw_complex workspace[length]);
SplitPlan makeSplitPlan(int slab,Decomp start,fftw_complex *dat,int length, fftw_complex workspace[length]);
SplitPlan makeParallelPlan(Decomp start,int *initial_factor, fftw_complex *dat,int length, fftw_complex workspace[length]);

Decomp getTransformed(SplitPlan plan);
void runSplitPlan(SplitPlan plan, int direction);
void freeSplitPlan(SplitPlan plan);
Decomp makeInterleavedDecomp(int ndim,MPI_Comm comm, int lengths[ndim], int counts[ndim], int grid[ndim]);
Decomp makeBlockedDecomp(int ndim, MPI_Comm comm, int A[ndim], int B[ndim], int grid[ndim]);
unsigned int gcd(unsigned int u, unsigned int v);

#endif
