CC=gcc
NVCC=nvcc
CFLAGS=-I -Wall -pthread -std=c11 -g
NVFLAGS=-arch=sm_52

LIBS=-lm

nbodymake: gpu.cu
	$(CC) $(CFLAGS) -o cpu.o cpu.c $(LIBS)
	$(NVCC) $(NVFLAGS) -o gpu.o gpu.cu

.PHONY: clean

clean:
	rm -f *.o *.out *.csv
