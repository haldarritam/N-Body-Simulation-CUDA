#include <nbody_helper2.h>




UNIVERSE US;

// time stamp function in seconds 
double getTimeStamp()
{     
    struct timeval tv;
	gettimeofday (&tv, NULL);
	return (double) tv.tv_usec/1000000 + tv.tv_sec;
}


// void print_BodyStats (const float3 *r, const float3 *v, const float3 *a, const unsigned long nElem)
// {
//     printf("\n");
//     unsigned long idx;
//     // print body number
//     for (idx=0; idx<nElem; idx++) {
//         if (idx == nElem-1)
//             printf("Mass %ld\n", idx);
//         else
//             printf("Mass %ld\t", idx);
//     }

//     // print Mass
//     for (idx=0; idx<nElem; idx++) {
//         if (idx == nElem-1)
//             printf("%.2f\n", r[idx].z);
//         else
//             printf("%.2f\t", m[idx]);
//     }

// 	// print position
// 	for (dim=0; dim<ND; dim++) {
// 		for (unsigned long idx=0; idx<nElem; idx++) {
// 			if (idx == nElem-1)
// 				printf("%.2f\n", *(r+idx*(ND+1)+dim));
// 			else
// 				printf("%.2f\t", *(r+idx*(ND+1)+dim));
// 		}
// 	}	
	
// 	// print velocity
// 	for (dim=0; dim<ND; dim++) {
// 		for (unsigned long idx=0; idx<nElem; idx++) {
// 			if (idx == nElem-1)
// 				printf("%.2f\n", v[ND*idx + dim]);
// 			else
// 				printf("%.2f\t", v[ND*idx + dim]);
// 		}
// 	}	

// 	// print acceleration
// 	for (dim=0; dim<ND; dim++) {
// 		for (unsigned long idx=0; idx<nElem; idx++) {
// 			if (idx == nElem-1)
// 				printf("%.2f\n", a[ND*idx + dim]);
// 			else
// 				printf("%.2f\t", a[ND*idx + dim]);
// 		}
// 	}	
// }

void init_MassPositionVelocity (float3 *r, float3 *v, const unsigned long nElem, const unsigned int config)
{
	// generate different seed for pseudo-random number generator
	// time_t t;
	// srand ((unsigned int) time(&t));
	srand ((unsigned int) 1000);

	// populating mass, position, & velocity arrays
	unsigned long idx;
	float mass_range = MAX_MASS - MIN_MASS;
	float x_mid = X_RES/2;
	float x_max = (X_RES + X_WIDTH)/2;
	float x_min = (X_RES - X_WIDTH)/2;
	float y_mid = Y_RES/2;
	float y_max = (Y_RES + Y_WIDTH)/2;
	float y_min = (Y_RES - Y_WIDTH)/2;

	switch (config) {
		case RANDOM_SQUARE_NO_VEL:
			for (idx=0; idx<nElem; idx++) {
				r[idx].x = (float) (rand()/RAND_MAX) * X_WIDTH + x_min;
				r[idx].y = (float) (rand()/RAND_MAX) * Y_WIDTH + y_min;
				r[idx].z = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				v[idx]   = {0.0f, 0.0f, 0.0f};
			}
			break;

		case RANDOM_CIRCLE_NO_VEL:
			float radius, x, y;
			for (idx=0; idx<nElem; idx++) {
				radius = (float) (rand()/RAND_MAX) * Y_WIDTH/2;
				x = (float) (rand()/RAND_MAX) * radius * rand_sign();
				y = sqrt(radius*radius - x*x) * rand_sign();
				r[idx].x = x_mid + x;
				r[idx].y = y_mid + y;;
				r[idx].z = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				v[idx]   = {0.0f, 0.0f, 0.0f};
			}
			break;

		case EXPAND_SHELL:
			float radius, x, y, angle;
			float shell_radius = Y_WIDTH/2;
			float shell_thickness = 0.25*shell_radius;
			float2 CoM = {0.0f, 0.0f};
			float system_mass = 0.0;
			float2 dist;
			float speed_factor=0.1;

			for (idx=0; idx<nElem; idx++) {
				// radius is the distance of point from center of window
				radius = (float) (rand()/RAND_MAX)*shell_thickness - shell_thickness/2 + shell_radius;
				x = (float) (rand()/RAND_MAX) * radius * rand_sign();
				y = sqrt(radius*radius - x*x) * rand_sign();
				r[idx].x = x_mid + x;
				r[idx].y = y_mid + y;;
				r[idx].z = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				CoM.x += r[idx].z * r[idx].x;
				CoM.y += r[idx].z * r[idx].y;
				system_mass += r[idx].z;
			}

			CoM.x /= system_mass;
			CoM.y /= system_mass;

			for (idx=0; idx<nElem; idx++) {
				// radius is now the distance of point from Center of Mass (CoM)
				dist.x = r[idx].x - CoM.x;
				dist.y = r[idx].y - CoM.y;
				angle = (float) atan(dist.y/dist.x);
				radius = sqrtf(dist.x*dist.x + dist.y*dist.y);
				radial_velocity = speed_factor * sqrtf(2*G*system_mass/radius);
				v[idx].x = radial_velocity * (float) cos(angle);
				v[idx].y = radial_velocity * (float) sin(angle);
				v[idx].z = 0.0f;
			}
			break;

		case SPIRAL_SINGLE_GALAXY:
			float radius;
			float2 CoM = {0.0f, 0.0f};
			float system_mass = 0.0;
			float2 dist, unit_dist;
			float tangential_speed;
			for (idx=0; idx<nElem; idx++) {
				if (idx == 0) {
					r[idx].x = x_mid;
					r[idx].y = y_mid;
					r[idx].z = ((float) (rand()/RAND_MAX) * mass_range + MIN_MASS)*10000;
				} else {
					r[idx].x = (float) (rand()/RAND_MAX) * X_WIDTH + x_min;
					r[idx].y = (float) (rand()/RAND_MAX) * Y_WIDTH + y_min;
					r[idx].z = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				}
				CoM.x += r[idx].z * r[idx].x;
				CoM.y += r[idx].z * r[idx].y;
				system_mass += r[idx].z;
			}

			CoM.x /= system_mass;
			CoM.y /= system_mass;

			for (idx=0; idx<nElem; idx++) {
				// radius is now the distance of point from Center of Mass (CoM)
				dist.x = r[idx].x - CoM.x;
				dist.y = r[idx].y - CoM.y;
				radius = sqrtf(dist.x*dist.x + dist.y*dist.y);
				unit_dist.x = dist.x / radius;
				unit_dist.y = dist.y / radius;
				tangential_speed = sqrtf(G*system_mass/radius) * 1.1;
				
				v[idx].x =    unit_dist.y * tangential_speed;
				v[idx].y = -1*unit_dist.x * tangential_speed;
				v[idx].z = 0.0f;
			}
			break;

		case SPIRAL_DOUBLE_GALAXY:
			float radius;
			float2 CoM = {0.0f, 0.0f};
			float system_mass = 0.0;
			float2 dist, unit_dist;
			float tangential_speed;
			for (idx=0; idx<nElem; idx++) {
				if (idx == 0) {
					r[idx].x = x_mid;
					r[idx].y = y_mid;
					r[idx].z = ((float) (rand()/RAND_MAX) * mass_range + MIN_MASS)*10000;
				} else {
					r[idx].x = (float) (rand()/RAND_MAX) * X_WIDTH + x_min;
					r[idx].y = (float) (rand()/RAND_MAX) * Y_WIDTH + y_min;
					r[idx].z = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				}
				CoM.x += r[idx].z * r[idx].x;
				CoM.y += r[idx].z * r[idx].y;
				system_mass += r[idx].z;
			}

			CoM.x /= system_mass;
			CoM.y /= system_mass;

			for (idx=0; idx<nElem; idx++) {
				// radius is now the distance of point from Center of Mass (CoM)
				dist.x = r[idx].x - CoM.x;
				dist.y = r[idx].y - CoM.y;
				radius = sqrtf(dist.x*dist.x + dist.y*dist.y);
				unit_dist.x = dist.x / radius;
				unit_dist.y = dist.y / radius;
				tangential_speed = sqrtf(G*system_mass/radius) * 1.1;
				
				v[idx].x =    unit_dist.y * tangential_speed;
				v[idx].y = -1*unit_dist.x * tangential_speed;
				v[idx].z = 0.0f;
			}
			break;

		case SPIRAL_QUAD_GALAXY:
			float radius;
			float2 CoM = {0.0f, 0.0f};
			float system_mass = 0.0;
			float2 dist, unit_dist;
			float tangential_speed;
			for (idx=0; idx<nElem; idx++) {
				if (idx == 0) {
					r[idx].x = x_mid;
					r[idx].y = y_mid;
					r[idx].z = ((float) (rand()/RAND_MAX) * mass_range + MIN_MASS)*10000;
				} else {
					r[idx].x = (float) (rand()/RAND_MAX) * X_WIDTH + x_min;
					r[idx].y = (float) (rand()/RAND_MAX) * Y_WIDTH + y_min;
					r[idx].z = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				}
				CoM.x += r[idx].z * r[idx].x;
				CoM.y += r[idx].z * r[idx].y;
				system_mass += r[idx].z;
			}

			CoM.x /= system_mass;
			CoM.y /= system_mass;

			for (idx=0; idx<nElem; idx++) {
				// radius is now the distance of point from Center of Mass (CoM)
				dist.x = r[idx].x - CoM.x;
				dist.y = r[idx].y - CoM.y;
				radius = sqrtf(dist.x*dist.x + dist.y*dist.y);
				unit_dist.x = dist.x / radius;
				unit_dist.y = dist.y / radius;
				tangential_speed = sqrtf(G*system_mass/radius) * 1.1;
				
				v[idx].x =    unit_dist.y * tangential_speed;
				v[idx].y = -1*unit_dist.x * tangential_speed;
				v[idx].z = 0.0f;
			}
			break;

		default:
			float radius, x, y;
			for (idx=0; idx<nElem; idx++) {
				radius = (float) (rand()/RAND_MAX) * Y_WIDTH/2;
				x = (float) (rand()/RAND_MAX) * radius * rand_sign();
				y = sqrt(radius*radius - x*x) * rand_sign();
				r[idx].x = x_mid + x;
				r[idx].y = y_mid + y;;
				r[idx].z = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				v[idx]   = {0.0f, 0.0f, 0.0f};
			}
			break;
	}


	for (idx=0; idx<nElem; idx++) 
	{
		// initializing mass and position
		if (idx % 100 == 0) {
			US.m[idx] = 128000*MAX_MASS;
			for (dim=0; dim<ND; dim++) {
				US.r1[ND*idx+dim] = (float) ((double) rand() / (double) (RAND_MAX/(1000*2)) - 1000);
			}
		} else {
			US.m[idx] = (float) ((double) rand() / (double) (RAND_MAX/MAX_MASS));
			for (dim=0; dim<ND; dim++) {
				US.r1[ND*idx+dim] = (float) ((double) rand() / (double) (RAND_MAX/(MAX_POS*2)) - MAX_POS);
			}
		}

		// calculating cross product along z-plane for rotational initial velocity about k-hat
		if (idx % 100 == 0) {
			US.v1[ND*idx + 0] = 0;
			US.v1[ND*idx + 1] = 0;
			US.v1[ND*idx + 2] = 0;
		} else {
			rx = US.r1[ND*idx + 0]; ry = US.r1[ND*idx + 1]; rz = US.r1[ND*idx + 2];
			mag_cross_sq = sqrtf(rx*rx + ry*ry + rz*rz);
			US.v1[ND*idx + 0] =    MAX_VEL * ry/mag_cross_sq;
			US.v1[ND*idx + 1] = -1*MAX_VEL * rx/mag_cross_sq;
			US.v1[ND*idx + 2] = 0.0;
		}
	}
}

// void *init_Acceleration_SMT (void *arg)
// {
// 	// define local variables for convenience
// 	unsigned long start, end, len, offset, nElem;

// 	nElem = US.nElem;
// 	offset = (unsigned long) arg;
// 	len = (unsigned long) US.nElem / NUM_CPU_THREADS;
// 	start = offset * len;
// 	end = start + len;

// 	unsigned long i, j;
// 	float ax_ip1, ay_ip1, az_ip1;
// 	float dx_ip1, dy_ip1, dz_ip1, rDistSquared, MinvDistCubed;
// 	float **i_r = &(US.r1);
// 	float **o_a = &(US.a1);

// 	// calculating NEXT acceleration of each body from the position of every other bodies
// 	// ... and NEXT velocity of each body utilizing the next acceleration
// 	for (i=start; i<end; i++) {
// 		ax_ip1 = 0.0;
// 		ay_ip1 = 0.0;
// 		az_ip1 = 0.0;
// 		for (j=0; j<nElem; j++) {
// 			dx_ip1 = *(*i_r + (ND*j+0)) - *(*i_r + (ND*i+0));
// 			dy_ip1 = *(*i_r + (ND*j+1)) - *(*i_r + (ND*i+1));
// 			dz_ip1 = *(*i_r + (ND*j+2)) - *(*i_r + (ND*i+2));
// 			rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + dz_ip1*dz_ip1 + SOFTENING;
// 			MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
// 			ax_ip1 += dx_ip1 * MinvDistCubed;
// 			ay_ip1 += dy_ip1 * MinvDistCubed;
// 			az_ip1 += dz_ip1 * MinvDistCubed;
// 		}

// 		*(*o_a + (ND*i+0)) = G*ax_ip1;
// 		*(*o_a + (ND*i+1)) = G*ay_ip1;
// 		*(*o_a + (ND*i+2)) = G*az_ip1;
// 	}

// 	pthread_exit (NULL);
// }

// HELPER FUNCTIONS
inline float3 scalevec (float3 v0, float scalar)
{
	float3 rt = v0;
	rt.x *= scalar;
	rt.y *= scalar;
	return rt;
}

inline float normalize (float2 v0)
{
	float dist = sqrtf(dot(v0, v0));
	if (dist > 1e-6) {
		v0.x /= dist;
		v0.y /= dist;
	} else {
		v0.x *= 1e6;
		v0.y *= 1e6;
	}
	
	return dist;
}

inline float dot (float2 v0, float2 v1)
{
	return v0.x*v1.x + v0.y*v1.y;
}

inline float3 cross (float3 v0, float3 v1)
{
	float3 v2;
	v2.x = v0.y*v1.z - v0.z*v1.y;
	v2.y = v0.z*v1.x - v0.x*v1.z;
	v2.z = v0.z*v1.y - v0.y*v1.x;
	return v2;
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline float rand_sign ()
{
	return (rand()-RAND_MAX) >= 0 ? 1.0 : -1.0;
}

void print_simulationParameters (unsigned long nElem, unsigned long nIter, unsigned int cpu_threads)
{
	printf("\n===== Simulation Parameters =====\n\n");
	printf("  Number of Bodies = %ld\n", nElem);
	printf("  Number of Time Steps = %ld\n", nIter);
	printf("  Number of CPU Threads = %d\n\n", cpu_threads);
	printf("=================================\n\n\n");
}


void print_deviceProperties (int dev, int driverVersion, int runtimeVersion, cudaDeviceProp deviceProp)
{
	printf("\n===== Device Properties ======\n\n");
	printf("  Device %d: %s\n", dev, deviceProp.name);
	printf("  CUDA Driver Version / Runtime Version: %d.%d / %d.%d\n",
		driverVersion/1000, (driverVersion%100)/10,
		runtimeVersion/1000, (runtimeVersion%100)/10);
	printf("  CUDA Capability Major/Minor version number: %d.%d\n",
		deviceProp.major, deviceProp.minor);
	printf("  Number of SMs: %d\n", deviceProp.multiProcessorCount);
	printf("  Total amount of global memory: %.2f GB (%llu B)\n",
		(float) deviceProp.totalGlobalMem/pow(1024.0,3),
		(unsigned long long) deviceProp.totalGlobalMem);
	printf("  Total amount of constant memory: %4.2f kB\n", 
		deviceProp.totalConstMem/1024.0);
	printf("  Total amount of shared memory per block: %4.2f kB\n",
		deviceProp.sharedMemPerBlock/1024.0);
	printf("  Total number of registers available per block: %d\n",
		deviceProp.regsPerBlock);
	printf("  Warp size: %d\n", deviceProp.warpSize);
	printf("  Maximum number of threads per block: %d\n",
		deviceProp.maxThreadsPerBlock);
	printf("  Maximum number of threads per SM: %d\n",
		deviceProp.maxThreadsPerMultiProcessor);
	printf("  Maximum number of warps per SM: %d\n",
		deviceProp.maxThreadsPerMultiProcessor/32);
	printf("  Maximum size of each block dimension: %d x %d x %d\n",
		deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]);
	printf("  Maximum size of each grid dimension: %d x %d x %d\n",
		deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
		deviceProp.maxGridSize[2]);
	printf("  Maximum memory pitch: %lu B\n", deviceProp.memPitch);
	printf("  Memory Clock Rate (MHz): %.1f\n", 
		deviceProp.memoryClockRate/1e3);
	printf("  Memory Bus Width (b): %d\n", deviceProp.memoryBusWidth);
	printf("  Peak Memory Bandwidth (GB/s): %.2f\n\n",
		2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1e6);
}

__device__ float2 bodyBodyInteraction (float2 ai, float3 bi, float3 bj)
{
	float2 dist;
	
	dist.x = bj.x - bi.x;
	dist.y = bj.y - bi.y;
	
	float distSqr = dot(dist, dist) + SOFTENING;
	float invDistCube = rsqrtf(distSqr * distSqr * distSqr);
	
	float s = bj.w * invDistCube
	
	ai.x += s * r.x;
	ai.y += s * r.y;
	return ai;
}

__global__ void initAcceleration (float3 *devA, float3 *devX, const unsigned nTiles)
{
	unsigned gtid = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ float3[] shPosition3;
	
	float3 myPosition3;
	float2 acc2 = {0.0f, 0.0f};
	
	myPosition3 = devX[gtid];
	for (unsigned tile=0; tile<nTiles; tile++) {
		shPosition3[threadIdx] = devX[ tile*blockDim.x + threadIdx ];
		__syncthreads();	// Wait for all threads in block to load data
							// ... into shared memory
		#pragma unroll 4
		for (unsigned j=0, j<blockDim.x; j++)
			acc2 = bodyBodyInteraction(acc2, myPosition3, shPosition3[j]);
		
		__syncthreads();	// wait for all threads in block to complete their
							// ... computations to not overwrite sh. mem.
	}
	
	devA[gtid] = (float3) {G*acc2.x, G*acc2.y, 0.0f};
}

__device__ float3 calcAcceleration (float3 *devX, unsigned nTiles)
{
	unsigned gtid = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ float3[] shPosition3;
	
	float3 myPosition4;
	float2 acc2 = {0.0f, 0.0f};
	
	myPosition3 = devX[gtid];
	for (unsigned tile=0; tile<nTiles; tile++) {
		shPosition3[threadIdx] = devX[ tile*blockDim.x + threadIdx ];
		__syncthreads();	// Wait for all threads in block to load data
							// ... into shared memory
		#pragma unroll 4
		for (unsigned j=0, j<blockDim.x; j++)
			acc2 = bodyBodyInteraction(acc2, myPosition3, shPosition3[j]);
		
		__syncthreads();	// wait for all threads in block to complete their
							// ... computations to not overwrite sh. mem.
	}
	
	float3 acc3 = {G*acc2.x, G*acc2.y, 0.0f};
	return acc3;
}

__global__ void calcIntegration (float3 *devX_ip1, const float3 *devX_i,
	float3 *devV_i, float3 *devA_i, const unsigned nElem, const unsigned nTiles)
{
	unsigned gtid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gtid < nElem) {
		float3 old_acc = devA_i[gtid];
		float3 old_vel = devV_i[gtid];
		float3 old_pos = devX_i[gtid];
		
		devX_ip1[gtid].x = old_pos.x + old_vel.x*DT + old_acc.x*DTSQd2;
		devX_ip1[gtid].y = old_pos.y + old_vel.y*DT + old_acc.y*DTSQd2;
		float3 new_acc   = calcAcceleration (devX_i, nTiles);
		devV_i  [gtid].x = old_vel.x + (old_acc.x + new_acc.x)*DTd2;
		devV_i  [gtid].y = old_vel.y + (old_acc.y + new_acc.y)*DTd2;
		devA_i  [gtid]   = new_acc;
	}
}