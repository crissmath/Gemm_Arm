#-----------------------------------

CC = gcc
CLINKER = gcc
OPTFLAGS = -O3

#-----------------------------------

TEST_GEMM = test_gemm.x

default: $(TEST_GEMM)

test_gemm.x: test_gemm.o sutils.o gemm_blis.o gemm_blis_neon.o
	$(CLINKER) $(OPTFLAGS) -o test_gemm.x test_gemm.o sutils.o gemm_blis.o gemm_blis_neon.o -lm

#-----------------------------------

.c.o:
	$(CC) $(OPTFLAGS) -c $*.c

#-----------------------------------

clean:
	rm *.o $(TEST_GEMM) 

#-----------------------------------

