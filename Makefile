
INC=reshape.h local_decomp.h block_decomp.h single_decomp.h interleave_decomp.h \
	rotate_decomp.h index.h typelist.h split_fft.h factor_reverse_decomp.h
SRC=reshape.c local_decomp.c block_decomp.c single_decomp.c interleave_decomp.c \
	rotate_decomp.c index.c typelist.c split_fft.c factor_reverse_decomp.c \
	dimscale.c processor_grid.c logger.c timer.c
OBJ=$(SRC:.c=.o)
FFTW_LIBS=-lfftw3 -lfftw3_threads -lrt 
CC=cc 
CFLAGS= -DUSE_TIMER -DNO_SAVE -DREDIRECT 
all: test fftw_test fftw_bench reshape_test reshape_bench split_test slab_test split_search plan_compare


$(OBJ): Makefile

fftw_test: fftw_test.c $(OBJ) $(INC)
	$(CC) $(CFLAGS) -o $@ fftw_test.c $(OBJ) $(FFTW_LIBS)

reshape_test: fftw_test.c $(OBJ) $(INC)
	$(CC) $(CFLAGS) -DUSE_RESHAPE -o $@ fftw_test.c $(OBJ) $(FFTW_LIBS)

split_test: split_plan_test.c $(OBJ) $(INC)
	$(CC) $(CFLAGS) -o $@ split_plan_test.c $(OBJ) $(FFTW_LIBS)

split_search: split_plan_search.c $(OBJ) $(INC)
	$(CC) $(CFLAGS) -o $@ split_plan_search.c $(OBJ) $(FFTW_LIBS)


slab_test: split_plan_test.c $(OBJ) $(INC)
	$(CC) $(CFLAGS) -DSLAB -o $@ split_plan_test.c $(OBJ) $(FFTW_LIBS)
	
fftw_bench: fftw_bench.c $(OBJ) $(INC)
	$(CC) $(CFLAGS) -o $@ fftw_bench.c $(OBJ) $(FFTW_LIBS)

reshape_bench: fftw_bench.c $(OBJ) $(INC)
	$(CC) $(CFLAGS) -DUSE_RESHAPE -o $@ fftw_bench.c $(OBJ) $(FFTW_LIBS)

plan_compare: plan_compare.c $(OBJ) $(INC)
	$(CC) $(CFLAGS) -DUSE_RESHAPE -o $@ plan_compare.c $(OBJ) $(FFTW_LIBS)

test: test.o $(OBJ) $(INC)
	$(CC) $(CFLAGS) -o $@ test.o $(OBJ) $(FFTW_LIBS)

$(OBJ): $(INC)

clean:
	rm test.o $(OBJ)

tar:
	tar cvf reshape.tar Makefile test.c $(SRC) $(INC)
