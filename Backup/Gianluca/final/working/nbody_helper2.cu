#include "nbody_helper2.h"

// time stamp function in seconds 
double getTimeStamp()
{     
    struct timeval tv;
	gettimeofday (&tv, NULL);
	return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

// HELPER FUNCTIONS
inline float3 scalevec (float3 &v0, float scalar)
{
	float3 rt = v0;
	rt.x *= scalar;
	rt.y *= scalar;
	return rt;
}

inline float dot (float2 v0, float2 v1)
{
	return v0.x*v1.x + v0.y*v1.y;
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

inline float3 cross (float3 v0, float3 v1)
{
	float3 v2;
	v2.x = v0.y*v1.z - v0.z*v1.y;
	v2.y = v0.z*v1.x - v0.x*v1.z;
	v2.z = v0.z*v1.y - v0.y*v1.x;
	return v2;
}



inline float rand_sign ()
{
	return (rand()-RAND_MAX) >= 0 ? 1.0 : -1.0;
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
	float x_width = 300.0;
	float y_width = 300.0;
	float x_mid = X_RES/2;
	//float x_max = (X_RES + x_width)/2;
	float x_min = (X_RES - x_width)/2;
	float y_mid = Y_RES/2;
	//float y_max = (Y_RES + y_width)/2;
	float y_min = (Y_RES - y_width)/2;

	float x, y, radius, angle, system_mass, speed_factor, tangential_speed;
	float shell_radius, shell_thickness, radial_velocity;
	float2 CoM, dist, unit_dist;

	switch (config) {
		case RANDOM_SQUARE_NO_VEL:
			printf("Initializing positions and mass\n");
			for (idx=0; idx<nElem; idx++) {
				r[idx].x = (float) ((double) rand()/RAND_MAX) * x_width + x_min;
				r[idx].y = (float) ((double) rand()/RAND_MAX) * y_width + y_min;
				r[idx].z = (float) ((double) rand()/RAND_MAX) * mass_range + MIN_MASS;
				v[idx]   = (float3) {0.0f, 0.0f, 0.0f};
				// printf("Body %ld\t x: %.6f\ty: %.6f\t m: %.6f\n",
				// 	idx, r[idx].x, r[idx].y, r[idx].z);
			}
			break;

		case RANDOM_CIRCLE_NO_VEL:
			for (idx=0; idx<nElem; idx++) {
				radius = (float) (rand()/RAND_MAX) * y_width/2;
				x = (float) (rand()/RAND_MAX) * radius * rand_sign();
				y = sqrt(radius*radius - x*x) * rand_sign();
				r[idx].x = x_mid + x;
				r[idx].y = y_mid + y;;
				r[idx].z = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				v[idx]   = (float3) {0.0f, 0.0f, 0.0f};
			}
			break;

		case EXPAND_SHELL:
			shell_radius = y_width/2;
			shell_thickness = 0.25*shell_radius;
			CoM = (float2) {0.0f, 0.0f};
			system_mass = 0.0;
			speed_factor=0.1;

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
			CoM = (float2) {0.0f, 0.0f};
			system_mass = 0.0;
			for (idx=0; idx<nElem; idx++) {
				if (idx == 0) {
					r[idx].x = x_mid;
					r[idx].y = y_mid;
					r[idx].z = ((float) (rand()/RAND_MAX) * mass_range + MIN_MASS)*10000;
				} else {
					r[idx].x = (float) (rand()/RAND_MAX) * x_width + x_min;
					r[idx].y = (float) (rand()/RAND_MAX) * y_width + y_min;
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
			CoM = (float2) {0.0f, 0.0f};
			system_mass = 0.0;
			for (idx=0; idx<nElem; idx++) {
				if (idx == 0) {
					r[idx].x = x_mid;
					r[idx].y = y_mid;
					r[idx].z = ((float) (rand()/RAND_MAX) * mass_range + MIN_MASS)*10000;
				} else {
					r[idx].x = (float) (rand()/RAND_MAX) * x_width + x_min;
					r[idx].y = (float) (rand()/RAND_MAX) * y_width + y_min;
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
			CoM = (float2) {0.0f, 0.0f};
			system_mass = 0.0;
			for (idx=0; idx<nElem; idx++) {
				if (idx == 0) {
					r[idx].x = x_mid;
					r[idx].y = y_mid;
					r[idx].z = ((float) (rand()/RAND_MAX) * mass_range + MIN_MASS)*10000;
				} else {
					r[idx].x = (float) (rand()/RAND_MAX) * x_width + x_min;
					r[idx].y = (float) (rand()/RAND_MAX) * y_width + y_min;
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
			for (idx=0; idx<nElem; idx++) {
				radius = (float) (rand()/RAND_MAX) * y_width/2;
				x = (float) (rand()/RAND_MAX) * radius * rand_sign();
				y = sqrt(radius*radius - x*x) * rand_sign();
				r[idx].x = x_mid + x;
				r[idx].y = y_mid + y;;
				r[idx].z = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				v[idx]   = (float3) {0.0f, 0.0f, 0.0f};
			}
			break;
	}
}

/*void *init_Acceleration_SMT (void *arg)
{
	// define local variables for convenience
	unsigned long start, end, len, offset, nElem;

	nElem = US.nElem;
	offset = (unsigned long) arg;
	len = (unsigned long) US.nElem / NUM_CPU_THREADS;
	start = offset * len;
	end = start + len;

	unsigned long i, j;
	float ax_ip1, ay_ip1, az_ip1;
	float dx_ip1, dy_ip1, dz_ip1, rDistSquared, MinvDistCubed;
	float **i_r = &(US.r1);
	float **o_a = &(US.a1);

	// calculating NEXT acceleration of each body from the position of every other bodies
	// ... and NEXT velocity of each body utilizing the next acceleration
	for (i=start; i<end; i++) {
		ax_ip1 = 0.0;
		ay_ip1 = 0.0;
		az_ip1 = 0.0;
		for (j=0; j<nElem; j++) {
			dx_ip1 = *(*i_r + (ND*j+0)) - *(*i_r + (ND*i+0));
			dy_ip1 = *(*i_r + (ND*j+1)) - *(*i_r + (ND*i+1));
			dz_ip1 = *(*i_r + (ND*j+2)) - *(*i_r + (ND*i+2));
			rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + dz_ip1*dz_ip1 + SOFTENING;
			MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
			ax_ip1 += dx_ip1 * MinvDistCubed;
			ay_ip1 += dy_ip1 * MinvDistCubed;
			az_ip1 += dz_ip1 * MinvDistCubed;
		}

		*(*o_a + (ND*i+0)) = G*ax_ip1;
		*(*o_a + (ND*i+1)) = G*ay_ip1;
		*(*o_a + (ND*i+2)) = G*az_ip1;
	}

	pthread_exit (NULL);
}*/



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

__device__ float2 bodyBodyInteraction (float2 ai, const float3 bi, const float3 bj)
{
	float2 dist;
	
	dist.x = bj.x - bi.x;
	dist.y = bj.y - bi.y;
	
	float distSqr = dist.x*dist.x + dist.y*dist.y + SOFTENING;
	float invDistCube = rsqrtf(distSqr * distSqr * distSqr);
	
	float s = bj.z * invDistCube;
	
	ai.x += s * dist.x;
	ai.y += s * dist.y;
	return ai;
}

__global__ void initAcceleration (float3 *devA, const float3 *__restrict__ devX, const unsigned nTiles)
{
	unsigned int gtid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	__shared__ float3 shPosition3[BLOCK_SIZE];
	
	float3 myPosition3;
	float2 acc2 = {0.0f, 0.0f};
	
	myPosition3 = devX[gtid];
	for (unsigned tile=0; tile<nTiles; tile++) {
		shPosition3[threadIdx.x] = devX[ tile*BLOCK_SIZE + threadIdx.x ];
		__syncthreads();	// Wait for all threads in block to load data
							// ... into shared memory
		#pragma unroll (16)
		for (unsigned j=0; j<BLOCK_SIZE; j++)
			acc2 = bodyBodyInteraction(acc2, myPosition3, shPosition3[j]);

		__syncthreads();	// wait for all threads in block to complete their
							// ... computations to not overwrite sh. mem.
	}
	
	devA[gtid] = (float3) {G*acc2.x, G*acc2.y, 0.0f};
}

__device__ float3 calcAcceleration (const float3 *__restrict__ devX, const unsigned nTiles)
{
	unsigned int gtid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	__shared__ float3 shPosition3[BLOCK_SIZE];
	
	float3 myPosition3;
	float2 acc2 = {0.0f, 0.0f};
	
	myPosition3 = devX[gtid];
	for (unsigned tile=0; tile<nTiles; tile++) {
		shPosition3[threadIdx.x] = devX[ tile*BLOCK_SIZE + threadIdx.x ];
		__syncthreads();	// Wait for all threads in block to load data
							// ... into shared memory

		#pragma unroll (16)
		for (unsigned j=0; j<BLOCK_SIZE; j++)
			acc2 = bodyBodyInteraction(acc2, myPosition3, shPosition3[j]);
		
		__syncthreads();	// wait for all threads in block to complete their
							// ... computations to not overwrite sh. mem.
	}
	
	float3 acc3 = {G*acc2.x, G*acc2.y, 0.0f};
	return acc3;
}

__global__ void calcIntegration (float3 *devX_ip1, const float3 *__restrict__ devX_i,
	float3 *devV_i, float3 *devA_i, const unsigned nElem, const unsigned nTiles)
{
	unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gtid < nElem) {
		// if (gtid == 1) 
		// 	printf("x: %.6f\ty: %.6f\tm: %.6f\n", devX_i[gtid].x, devX_i[gtid].y, devX_i[gtid].z);
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

/*void *computeHost_SMT (void *arg)
{
	// define local variables for convenience
	unsigned long start, end, len, offset, nElem, nIter;

	nElem = US.nElem;
	nIter = US.nIter;
	offset = (unsigned long) arg;
	len = (unsigned long) nElem / NUM_CPU_THREADS;
	start = offset * len;
	end = start + len;

	unsigned long i, j;
	float3 pos_i, accel_ip1, dpos_ip1;
	float rDistSquared, MinvDistCubed;
	float **i_r, **i_v, **i_a;
	float **o_r, **o_v, **o_a;
	for (unsigned long iter=0; iter<nIter; iter++) {

		// since the computation cannot be done inplace, we constantly need to 
		// swap where the input and output data locations are
		if (iter % 2 == 0) {
			i_r = &(US.r1);
			i_v = &(US.v1);
			i_a = &(US.a1);
			o_r = &(US.r2);
			o_v = &(US.v2);
			o_a = &(US.a2);
		} else {
			i_r = &(US.r2);
			i_v = &(US.v2);
			i_a = &(US.a2);
			o_r = &(US.r1);
			o_v = &(US.v1);
			o_a = &(US.a1);
		}

		// calculating NEXT position of each body
		for (i=start; i<end; i++) {
			if (i % 100 == 0) {
				*(*o_r + (ND*i))   = *(*i_r + (ND*i));
				*(*o_r + (ND*i+1)) = *(*i_r + (ND*i+1));
				*(*o_r + (ND*i+2)) = *(*i_r + (ND*i+2));
			} else {
				*(*o_r + (ND*i))   = *(*i_r + (ND*i))   + *(*i_v + (ND*i))*DT   + *(*i_a + (ND*i))  *DTSQd2;
				*(*o_r + (ND*i+1)) = *(*i_r + (ND*i+1)) + *(*i_v + (ND*i+1))*DT + *(*i_a + (ND*i+1))*DTSQd2;
				*(*o_r + (ND*i+2)) = *(*i_r + (ND*i+2)) + *(*i_v + (ND*i+2))*DT + *(*i_a + (ND*i+2))*DTSQd2;
			}
		}

		// position computation done
		pthread_mutex_lock (&count_mutex);
		count++;

		if (count == NUM_CPU_THREADS*(2*iter+1)) {
			pthread_cond_broadcast (&count_condition);
			//printf("Broadcasting by tid=%ld\n", offset);
		} else {
			do {
				pthread_cond_wait (&count_condition, &count_mutex);
				//printf("Condition Broadcast received by tid=%ld\n", offset);
			} while (count < NUM_CPU_THREADS*(2*iter+1));
		}

		pthread_mutex_unlock (&count_mutex);


		// calculating NEXT acceleration of each body from the position of every other bodies
		// ... and NEXT velocity of each body utilizing the next acceleration
		for (i=start; i<end; i++) {
			pos_i.x = *(*o_r + (ND*i+0));
			pos_i.y = *(*o_r + (ND*i+1));
			pos_i.z = *(*o_r + (ND*i+2));

			accel_ip1 = (float3) {.x=0.0f, .y=0.0f, .z=0.0f};
			

			// unrolling this loop 8x for ~2% performance improvement
			j = 0;
			while (j < nElem) {
				dpos_ip1.x = *(*o_r + (ND*j+0)) - pos_i.x;
				dpos_ip1.y = *(*o_r + (ND*j+1)) - pos_i.y;
				dpos_ip1.z = *(*o_r + (ND*j+2)) - pos_i.z;
				rDistSquared = dpos_ip1.x*dpos_ip1.x + dpos_ip1.y*dpos_ip1.y + dpos_ip1.z*dpos_ip1.z + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				accel_ip1.x += dpos_ip1.x * MinvDistCubed;
				accel_ip1.y += dpos_ip1.y * MinvDistCubed;
				accel_ip1.z += dpos_ip1.z * MinvDistCubed;

				j++; // unroll #1

				dpos_ip1.x = *(*o_r + (ND*j+0)) - pos_i.x;
				dpos_ip1.y = *(*o_r + (ND*j+1)) - pos_i.y;
				dpos_ip1.z = *(*o_r + (ND*j+2)) - pos_i.z;
				rDistSquared = dpos_ip1.x*dpos_ip1.x + dpos_ip1.y*dpos_ip1.y + dpos_ip1.z*dpos_ip1.z + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				accel_ip1.x += dpos_ip1.x * MinvDistCubed;
				accel_ip1.y += dpos_ip1.y * MinvDistCubed;
				accel_ip1.z += dpos_ip1.z * MinvDistCubed;

				j++; // unroll #2

				dpos_ip1.x = *(*o_r + (ND*j+0)) - pos_i.x;
				dpos_ip1.y = *(*o_r + (ND*j+1)) - pos_i.y;
				dpos_ip1.z = *(*o_r + (ND*j+2)) - pos_i.z;
				rDistSquared = dpos_ip1.x*dpos_ip1.x + dpos_ip1.y*dpos_ip1.y + dpos_ip1.z*dpos_ip1.z + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				accel_ip1.x += dpos_ip1.x * MinvDistCubed;
				accel_ip1.y += dpos_ip1.y * MinvDistCubed;
				accel_ip1.z += dpos_ip1.z * MinvDistCubed;

				j++; // unroll #3

				dpos_ip1.x = *(*o_r + (ND*j+0)) - pos_i.x;
				dpos_ip1.y = *(*o_r + (ND*j+1)) - pos_i.y;
				dpos_ip1.z = *(*o_r + (ND*j+2)) - pos_i.z;
				rDistSquared = dpos_ip1.x*dpos_ip1.x + dpos_ip1.y*dpos_ip1.y + dpos_ip1.z*dpos_ip1.z + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				accel_ip1.x += dpos_ip1.x * MinvDistCubed;
				accel_ip1.y += dpos_ip1.y * MinvDistCubed;
				accel_ip1.z += dpos_ip1.z * MinvDistCubed;

				j++; // unroll #4

				dpos_ip1.x = *(*o_r + (ND*j+0)) - pos_i.x;
				dpos_ip1.y = *(*o_r + (ND*j+1)) - pos_i.y;
				dpos_ip1.z = *(*o_r + (ND*j+2)) - pos_i.z;
				rDistSquared = dpos_ip1.x*dpos_ip1.x + dpos_ip1.y*dpos_ip1.y + dpos_ip1.z*dpos_ip1.z + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				accel_ip1.x += dpos_ip1.x * MinvDistCubed;
				accel_ip1.y += dpos_ip1.y * MinvDistCubed;
				accel_ip1.z += dpos_ip1.z * MinvDistCubed;

				j++; // unroll #5

				dpos_ip1.x = *(*o_r + (ND*j+0)) - pos_i.x;
				dpos_ip1.y = *(*o_r + (ND*j+1)) - pos_i.y;
				dpos_ip1.z = *(*o_r + (ND*j+2)) - pos_i.z;
				rDistSquared = dpos_ip1.x*dpos_ip1.x + dpos_ip1.y*dpos_ip1.y + dpos_ip1.z*dpos_ip1.z + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				accel_ip1.x += dpos_ip1.x * MinvDistCubed;
				accel_ip1.y += dpos_ip1.y * MinvDistCubed;
				accel_ip1.z += dpos_ip1.z * MinvDistCubed;

				j++; // unroll #6

				dpos_ip1.x = *(*o_r + (ND*j+0)) - pos_i.x;
				dpos_ip1.y = *(*o_r + (ND*j+1)) - pos_i.y;
				dpos_ip1.z = *(*o_r + (ND*j+2)) - pos_i.z;
				rDistSquared = dpos_ip1.x*dpos_ip1.x + dpos_ip1.y*dpos_ip1.y + dpos_ip1.z*dpos_ip1.z + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				accel_ip1.x += dpos_ip1.x * MinvDistCubed;
				accel_ip1.y += dpos_ip1.y * MinvDistCubed;
				accel_ip1.z += dpos_ip1.z * MinvDistCubed;

				j++; // unroll #7

				dpos_ip1.x = *(*o_r + (ND*j+0)) - pos_i.x;
				dpos_ip1.y = *(*o_r + (ND*j+1)) - pos_i.y;
				dpos_ip1.z = *(*o_r + (ND*j+2)) - pos_i.z;
				rDistSquared = dpos_ip1.x*dpos_ip1.x + dpos_ip1.y*dpos_ip1.y + dpos_ip1.z*dpos_ip1.z + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				accel_ip1.x += dpos_ip1.x * MinvDistCubed;
				accel_ip1.y += dpos_ip1.y * MinvDistCubed;
				accel_ip1.z += dpos_ip1.z * MinvDistCubed;

				j++; // unroll #8
			}
			
			*(*o_a + (ND*i+0)) = G*accel_ip1.x;
			*(*o_a + (ND*i+1)) = G*accel_ip1.y;
			*(*o_a + (ND*i+2)) = G*accel_ip1.z;

			*(*o_v + (ND*i+0)) = *(*i_v + (ND*i+0)) + (*(*i_a + (ND*i+0))+accel_ip1.x)*DTd2;
			*(*o_v + (ND*i+1)) = *(*i_v + (ND*i+1)) + (*(*i_a + (ND*i+1))+accel_ip1.y)*DTd2;
			*(*o_v + (ND*i+2)) = *(*i_v + (ND*i+2)) + (*(*i_a + (ND*i+2))+accel_ip1.z)*DTd2;
		}


		// computation completed on thread. Acquire mutex to increment count variable...
		pthread_mutex_lock (&count_mutex);
		count++;

		if (count == NUM_CPU_THREADS*(2*iter+2)) {
			// writing to file
			for (unsigned int idx=0; idx<nElem; idx++) {
				fprintf(destFile, "%f,%f,%f\n", *(*o_r+ND*idx+0), *(*o_r+ND*idx+1), *(*o_r+ND*idx+2));
			}

			pthread_cond_broadcast (&count_condition);
			//printf("Broadcasting by tid=%ld\n", offset);
		} else {
			do {
				pthread_cond_wait (&count_condition, &count_mutex);
				//printf("Condition Broadcast received by tid=%ld\n", offset);
			} while (count < NUM_CPU_THREADS*(2*iter+2));
		}

		pthread_mutex_unlock (&count_mutex);

		if (offset == 1) {
			printf("%ld:\tx: %.6f\ty: %.6f\tz: %.6f\n",
				iter, *(*o_r + (ND*offset)), *(*o_r + (ND*offset)+1), *(*o_r + (ND*offset)+2));
		}

		//if (offset == 0)
		//	print_BodyStats (US.m, *o_r, *o_v, *o_a);
	}

	pthread_exit (NULL);
}*/
