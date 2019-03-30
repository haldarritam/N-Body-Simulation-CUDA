#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

//remove this if run on ug machine
#include <device_launch_parameters.h>

#define G 6.67e-2f
#define BLOCK_DIM 1024
#define MAX_RANGE 100.0f
#define MASS 10000.0f
#define EPS 1.0f




void initialize(float3* h_s, float3* h_v, float3* h_a, float dt, int num_bodies) {
	int i, j;
	float3 r;
	float r2, inv_r3;
	for (i = 0; i < num_bodies; i++) {
		h_s[i].x = MAX_RANGE * rand() / (float)RAND_MAX;
		h_s[i].y = MAX_RANGE * rand() / (float)RAND_MAX;
		h_s[i].z = MAX_RANGE * rand() / (float)RAND_MAX;
	}

	for (i = 0; i < num_bodies; i++) {
		for (j = 0; j < num_bodies; j++) {
			r.x = h_s[j].x - h_s[i].x;
			r.y = h_s[j].y - h_s[i].y;
			r.z = h_s[j].z - h_s[i].z;
			r2 = r.x*r.x + r.y*r.y + r.z*r.z + EPS;
			inv_r3 = 1.0f / sqrtf(r2*r2*r2);
			h_a[i].x += inv_r3 * r.x*MASS;
			h_a[i].y += inv_r3 * r.y*MASS;
			h_a[i].z += inv_r3 * r.z*MASS;
		}
	}

	for (i = 0; i < num_bodies; i++) {
		h_v[i].x = h_a[i].x*dt*0.5;
		h_v[i].y = h_a[i].y*dt*0.5;
		h_v[i].z = h_a[i].z*dt*0.5;
	}
}

void cpu_func(float3* h_s, float3* h_v, float3* h_a, float dt, int num_bodies, int num_iteration) {
	int i, j,k;
	float3 r;
	float r2, inv_r3;
	FILE* fp = fopen("pos_cpu.txt","w");

	for (i = 0; i < num_iteration; i++) {
		
		for (j = 0; j< num_bodies; j++) {
			for (k = 0; k < num_bodies; k++) {
				r.x = h_s[k].x - h_s[j].x;
				r.y = h_s[k].y - h_s[j].y;
				r.z = h_s[k].z - h_s[j].z;
				r2 = r.x*r.x + r.y*r.y + r.z*r.z + EPS;
				inv_r3 = 1.0f / sqrtf(r2*r2*r2);
				h_a[j].x += inv_r3 * r.x*MASS;
				h_a[j].y += inv_r3 * r.y*MASS;
				h_a[j].z += inv_r3 * r.z*MASS;
			}
		}

		
		for (j = 0; j < num_bodies; j++) {
			
			h_s[j].x +=  h_v[j].x*dt;
			h_s[j].y +=  h_v[j].y*dt;
			h_s[j].z +=  h_v[j].z*dt;
			h_v[j].x +=  h_a[j].x*dt;
			h_v[j].y +=  h_a[j].y*dt;
			h_v[j].z +=  h_a[j].z*dt;
			fprintf(fp, "%.6f %.6f %.6f\n", h_s[j].x, h_s[j].y, h_s[j].z);
		}
	}
	fclose(fp);

}


/*
__global__ void initialize(float3* s,float3* v,float3* a,int num_bodies){

}*/

__device__ void force_calc(float3 s, float3* a) {
	__shared__ float3 shared_s[BLOCK_DIM];
	int i;
	float3 r;
	float r2, inv_r3;
	for (i = 0; i < blockDim.x; i++) {
		r.x = shared_s[i].x - s.x;
		r.y = shared_s[i].y - s.y;
		r.z = shared_s[i].z - s.z;
		r2 = r.x*r.x + r.y*r.y + r.z*r.z + EPS;
		inv_r3 = 1.0f / sqrtf(r2*r2*r2);
		(*a).x += inv_r3 * r.x*MASS*G;
		(*a).y += inv_r3 * r.y*MASS*G;
		(*a).z += inv_r3 * r.z*MASS*G;
	}
}


__global__ void accel_update(float3* s, float3* a, int num_bodies) {
	__shared__ float3 shared_s[BLOCK_DIM];
	int gidx = blockDim.x*blockIdx.x + threadIdx.x;
	float3 accel = { 0.0f,0.0f,0.0f }, myPos = s[gidx];
	int idx, i, tile = 0;

	if (idx < num_bodies) {
		for (i = 0; i < num_bodies; i += blockDim.x) {
			idx = tile * blockDim.x + threadIdx.x;
			shared_s[threadIdx.x] = s[idx];
			__syncthreads();
			force_calc(myPos, &accel);
			__syncthreads();
			tile++;
		}
	}
	a[gidx] = accel;
}

__global__ void pos_update(float3* s, float3* v, float3* a, float dt, int num_bodies) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < num_bodies) {
		s[i].x += v[i].x*dt;
		s[i].y += v[i].y*dt;
		s[i].z += v[i].z*dt;
		v[i].x += a[i].x*dt;
		v[i].y += a[i].y*dt;
		v[i].z += a[i].z*dt;
	}
}

void gpu_func(float3* s, float3* v, float3* a, float3* h_s, float3* h_a, float dt, int num_bodies, int num_iteration) {
	int i, j;
	FILE *fp = fopen("pos.txt", "w");


	for (i = 0; i < num_iteration; i++) {
		pos_update <<<(num_bodies + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM >>> (s, v, a, dt, num_bodies);
		cudaDeviceSynchronize();
		accel_update <<<(num_bodies + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM >>> (s, v, num_bodies);
		cudaDeviceSynchronize();
		cudaMemcpy(h_s, s, num_bodies * sizeof(float3), cudaMemcpyDeviceToHost);
		//cudaMemcpy( h_a, a, num_bodies * sizeof(float3), cudaMemcpyDeviceToHost );
		for (j = 0; j < num_bodies; j++)
			fprintf(fp, "%.6f %.6f %.6f\n", h_s[j].x, h_s[j].y, h_s[j].z);
	}
	fclose(fp);
}


int main(int argc, char *argv[]) {
	if (argc < 2) {
		printf("Error: wrong number of args\n");
		exit(0);
	}
	int num_bodies = atoi(argv[1]);
	float dt = 1;
	if (argc > 2) dt = atof(argv[2]);
	int num_iteration = 10;
	if (argc > 3) num_iteration = atoi(argv[3]);

	//host memory allocation
	float3 *h_s, *h_v, *h_a;
		cudaError_t err00 = cudaMallocHost((void**)&h_s, num_bodies * sizeof(float3));
	if (err00 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err00), __FILE__, __LINE__);
	}
	cudaError_t err01 = cudaMallocHost((void**)&h_v, num_bodies * sizeof(float3));
	if (err01 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err01), __FILE__, __LINE__);
	}
	cudaError_t err02 = cudaMallocHost((void**)&h_a, num_bodies * sizeof(float3));
	if (err02 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err02), __FILE__, __LINE__);
	}

	//initialization
	initialize(h_s, h_v, h_a, dt, num_bodies);

	//device memory allocation
	float3 *s, *v, *a;
	cudaError_t err10 = cudaMalloc((void**)&s, num_bodies * sizeof(float3));
	if (err10 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err10), __FILE__, __LINE__);
	}
	cudaError_t err11 = cudaMalloc((void**)&v, num_bodies * sizeof(float3));
	if (err11 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err11), __FILE__, __LINE__);
	}
	cudaError_t err12 = cudaMalloc((void**)&a, num_bodies * sizeof(float3));
	if (err12 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err12), __FILE__, __LINE__);
	}

	//copy initialized data from host to device
	cudaError_t err20 = cudaMemcpy(s, h_s, num_bodies * sizeof(float3), cudaMemcpyHostToDevice);
	if (err20 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err20), __FILE__, __LINE__);
	}
	cudaError_t err21 = cudaMemcpy(v, h_v, num_bodies * sizeof(float3), cudaMemcpyHostToDevice);
	if (err21 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err21), __FILE__, __LINE__);
	}
	cudaError_t err22 = cudaMemcpy(a, h_a, num_bodies * sizeof(float3), cudaMemcpyHostToDevice);
	if (err22 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err22), __FILE__, __LINE__);
	}
	//cpu code
	cpu_func(h_s,h_v,h_a,dt,num_bodies, num_iteration);

	//run gpu code
	gpu_func(s, v, a, h_s, h_a, dt, num_bodies, num_iteration);

	

	cudaDeviceReset();
	cudaFree(s); cudaFree(v); cudaFree(a);
	cudaFreeHost(h_s); cudaFreeHost(h_v); cudaFreeHost(h_a);
}