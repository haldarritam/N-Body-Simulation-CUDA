CC=gcc
NVCC=nvcc
CFLAGS=-I -Wall -pthread -std=c11 -g
NVFLAGS=-arch=sm_52 -Xcompiler="-pthread -Wall"

LIBS=-lm

nbodymake: gpu.cu
	$(CC) $(CFLAGS) -o cpu.o cpu.c $(LIBS)
	$(NVCC) $(NVFLAGS) -o gpu.o gpu.cu
	$(NVCC) $(NVFLAGS) -o gpu2.o gpu2.cu

.PHONY: clean

clean:
	rm -f *.o *.out *.csv
