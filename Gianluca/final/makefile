CC=gcc
NVCC=nvcc
CFLAGS=-I -Wall -pthread -std=c11 -g
NVFLAGS=-ccbin clang++-3.8 -I -m64 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -Xcompiler="-pthread -Wall"

LIBS=-lm

nbodymake: gpu2.cu
	#$(CC) $(CFLAGS) -o cpu.o cpu.c $(LIBS)
	#$(NVCC) $(NVFLAGS) -o gpu.o gpu.cu
	$(NVCC) $(NVFLAGS) -o gpu.o nbody_helper2.cu gpu2.cu 

.PHONY: clean

clean:
	rm -f *.o *.out *.csv
