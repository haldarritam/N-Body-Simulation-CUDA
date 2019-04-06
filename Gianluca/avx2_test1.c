#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

double getTimeStamp ()
{
	struct timeval tv;
	gettimeofday (&tv, NULL);
	return ((double) tv.tv_usec/1000000 + (double) tv.tv_sec);
}

void initialize (float *a, float *b, float *c, const unsigned nElem)
{
	time_t t;
	srand ((unsigned int) time(&t));
	
	for (unsigned int idx=0; idx<nElem; idx++) {
		a[idx] = 1000 * (float) rand() / RAND_MAX;
		b[idx] = 1000 * (float) rand() / RAND_MAX;
		c[idx] = 0.0f;
	}
}

void addition_avx (const float *a, const float *b, float *c, const unsigned nElem)
{
	__m256 a_avx, b_avx, c_avx;
	for (unsigned int i=0; i<nElem; i += 8) {
		a_avx = _mm256_load_ps (&a[i]);
		b_avx = _mm256_load_ps (&b[i]);
		c_avx = _mm256_add_ps (a_avx, b_avx);
		_mm256_store_ps (&c[i], c_avx);
	}
}

void addition_normal (const float *a, const float *b, float *c, const unsigned nElem)
{
	for (unsigned int i=0; i<nElem; i++)
		c[i] = a[i] + b[i];
}

void print_vectors (const float *a, const float *b, const float *c, const unsigned nElem)
{
	for (unsigned int i=0; i<nElem; i++)
		printf("%.2f\t%.2f\t%.2f\n", a[i], b[i], c[i]);
	
	printf("\n");
}

bool compare (const float *c1, const float *c2, const unsigned nElem)
{
	for (unsigned int i=0; i<nElem; i++)
		if (c1[i] != c2[i]) return false;
	
	return true;
}

int main (int argc, char *argv[])
{
	unsigned int nElem = 32;
	if (argc == 2) nElem = (unsigned int) atoi(argv[1]);
	
	size_t nBytes = nElem * sizeof(float);
	size_t alignment = 32;
	
	float *a, *b, *c1, *c2;
	assert (!posix_memalign ((void**)&a,  alignment, nBytes));
	assert (!posix_memalign ((void**)&b,  alignment, nBytes));
	assert (!posix_memalign ((void**)&c1, alignment, nBytes));
	assert (!posix_memalign ((void**)&c2, alignment, nBytes));
	
	initialize (a, b, c1, nElem);
	memcpy (c2, c1, nBytes);
	//print_vectors (a, b, c, nElem);
	double time1 = getTimeStamp ();
	addition_avx (a, b, c1, nElem);
	double time2 = getTimeStamp ();
	addition_normal (a, b, c2, nElem);
	double time3 = getTimeStamp ();
	//print_vectors (a, b, c, nElem);
	assert (compare (c1, c2, nElem));
	
	free (a);
	free (b);
	free (c1);
	free (c2);
	
	printf("Elapsed Time (AVX2): %.4lfs\n", time2-time1); 
	printf("Elapsed Time (norm): %.4lfs\n", time3-time2);
	
	return 0;
}
