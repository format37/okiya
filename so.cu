#include <stdio.h>

#define N 16

__global__ void test(int *out, int n)
{
    for(int i = 0; i < n; i++) out[i]=round(pow(2,i));
}

int main()
{
    int *a;
    int *out;
    a   = (int*)malloc(sizeof(int) * N);
    cudaMalloc((void**)&out, sizeof(int) * N);
    test<<<1, 1>>>(out, N);
    cudaMemcpy(a, out, sizeof(int) * N, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; i++) printf("%d\n",a[i]);
}