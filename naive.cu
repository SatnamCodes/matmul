#include <iostream>
#include <cuda.h>
#include<cudaProfiler.h>
#include <cuda_runtime.h>
#include <vector>

//cuda error checker macro
#define CHECK_CUDA(call)                                      \
do {                                                           \
    cudaError_t err = call;                                   \
    if(err != cudaSuccess){                                   \
        std::cerr << "CUDA Error: "                           \
                  << cudaGetErrorString(err)                  \
                  << " at line " << __LINE__ << std::endl;    \
        exit(EXIT_FAILURE);                                   \
    }                                                          \
} while(0)


//natnul function.
__global__ void matmul_naive(int M, int N, int K, float alpha , const float *A,const float *B, float beta , float *C){
    const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y+threadIdx.y;
    if (x<M && y<N){
        float temp = 0.0;
        for (int i = 0 ; i<K;++i){
            temp+=A[x*K+i] * B[i*N+y];
        }
        C[x*N+y] = alpha * temp + beta * C[x*N+y];
    }
}
int main (){
    int M = 1024;
    int N = 1024;
    int K = 1024;
    float alpha = 1.0f; //scalar 1
    float beta = 0.0f; //scalar 2

    //GEMM eq: αAB+βC

    //host memory calculation
    /* A-> M × K
    B-> K × N
    C -> M × N*/
    size_t sizeA = M*K*sizeof(float);
    size_t sizeB = K*N*sizeof(float);
    size_t sizeC = M*N*sizeof(float); 
    //size_t is used to hold the largest possible size of an object.

    //assigning vectors
    std::vector<float> h_A(M*K);
    std::vector<float> h_B(K*N);
    std::vector<float> h_C(M*N);

    //Initialising Matrices
    for (int i = 0 ; i< M*K ;i++){
        h_A[i] = 1.0f;
    }
    for (int i = 0 ; i< K*N ;i++){
        h_B[i] = 1.0f;
    }
    for (int i = 0 ; i< M*N ;i++){
        h_C[i] = 1.0f;
    }

    //device pointers.
    //cudaMalloc (pointer,bytes)
    float *d_A, *d_B,*d_C;
    CHECK_CUDA(cudaMalloc(&d_A,sizeA));
    CHECK_CUDA(cudaMalloc(&d_B,sizeB));
    CHECK_CUDA(cudaMalloc(&d_C,sizeC));

    // copy from host to device
    //parameters -> destination , source ,size,direction
    cudaMemcpy(
        d_A,
        h_A.data(),
        sizeA,
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_B,
        h_B.data(),
        sizeB,
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_C,
        h_C.data(),
        sizeC,
        cudaMemcpyHostToDevice
    );

    //Thread Configuration 
    dim3 block(16,16); //256 threads/block 
    // grid configuration 
    dim3 grid(
        (N+block.x-1)/block.x, (M+block.y-1)/block.y); //Ceiling division

    //CUDA Events for timing.    
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);



    cudaEventRecord(start); //starts GPU timer.
    matmul_naive<<<grid,block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);//call the function
    cudaEventRecord(stop);//record the stopping time.
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms,start,stop);

    //copy result from device to host 
    cudaMemcpy(h_C.data(),d_C,sizeC,cudaMemcpyDeviceToHost);

    bool correct= true;

    for (int i=0;i<M*N;i++){
        if (h_C[i]!=K){
            correct = false;
            break;
        }
    }

    if (correct){
        std::cout<<"Result is correct."<<std::endl;
        }

    else{
        std::cout<<"The result is incorrect."<<std::endl;
    }
    std::cout<<"Kernel Time : "<<ms<<" ms"<<std::endl;


    //gflops calc

    float gflops = ((2.0*M*N*K)/(ms*1e6)); //each multiplication performs 1 multiplication and 1 add 

    std::cout<<"GFLOPS: "<<gflops<<std::endl;


    //freeup 

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}