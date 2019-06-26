CC=gcc
NVCC=nvcc
CFLAGS=-I -Wall -pthread -std=c11 -g -m64
NVFLAGS=-ccbin clang++-3.8 -m64 -O3 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -Xcompiler="-pthread -Wall -O3"

LIBS=-lm

nbodymake: gpu2.cu
	#$(NVCC) $(NVFLAGS) -o gpu_streams.o nbody_helper2.cu gpu2_streams.cu
	$(NVCC) $(NVFLAGS) -o gpu.o nbody_helper2.cu gpu2.cu
	#$(NVCC) $(NVFLAGS) -o gpu_slow.o test_gpu.cu

.PHONY: clean

clean:
	rm -f *.o *.out *.csv
