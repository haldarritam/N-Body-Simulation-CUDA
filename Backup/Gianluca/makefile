CC=gcc
NVCC=nvcc
CFLAGS=-Wall -pthread -std=c11 -g -march=znver1 -O3
NVFLAGS=-ccbin g++ -arch=sm_52

LIBS=-lm

nbodymake: test_pthread.c
	$(CC) $(CFLAGS) -o cputest.o test_pthread.c $(LIBS)
	#$(CC) $(CFLAGS) -o cputest2file.o test_pthread_tofile.c $(LIBS)
	#$(NVCC) $(NVFLAGS) -o gputest.o test_gpu.cu

.PHONY: clean

clean:
	rm -f *.o *.csv
