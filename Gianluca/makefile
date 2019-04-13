CC=gcc
NVCC=nvcc
CFLAGS=-I -Wall -pthread -std=c11 -g
NVFLAGS=-ccbin clang++-3.8 -arch=sm_52

LIBS=-lm

nbodymake: test_pthread.c
	$(CC) $(CFLAGS) -o cputest.o test_pthread.c $(LIBS)
	$(CC) $(CFLAGS) -o cputest2file.o test_pthread_tofile.c $(LIBS)
	$(NVCC) $(NVFLAGS) -o gputest.o test_gpu.cu

.PHONY: clean

clean:
	rm -f *.o *.csv
