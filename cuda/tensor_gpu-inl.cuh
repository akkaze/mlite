#ifndef MLITE_CUDA_GPU_INL_CUH_
#define MLITE_CUDA_GPU_INL_CUH_
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif
#include "tensor.h"
#define MLITE_CUDA_POST_KERNEL_CHECK(x) \
  /* Code block avoids redefinition of cudaError_t err */ \
  do { \
    cudaError err = cudaPeekAtLastError(); \
    CHECK_EQ(err, cudaSuccess) << "Name: " << #x << " ErrStr:" << cudaGetErrorString(err); \
  } while (0)
#define MLITE_CUDA_1D_KERNEL_LOOP(i, n)                             \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n,		\
		i += blockDim.x * gridDim.x)
namespace mlite {
namespace cuda {
/* load unit for memory access, if CUDAARCH not defined, this is advanced nvcc */
#if MLITE_OLD_CUDA
const int kMemUnitBits = 4;
const int kMaxThreadsPerBlock = 512;
#else
const int kMemUnitBits = 5;
const int kMaxThreadsPerBlock = 1024;
#endif
/*! \brief number of units that can do synchronized update, half warp size */
const int kMemUnit = 1 << kMemUnitBits;
/*! \brief mask that could be helpful sometime */
const int kMemUnitMask = kMemUnit - 1;
/*! \brief suggested thread number(logscale) for mapping kernel */
const int kBaseThreadBits = 8;
/*! \brief suggested thread number for mapping kernel */
const int kBaseThreadNum = 1 << kBaseThreadBits;
/*! \brief maximum value of grid */
const int kMaxGridNum = 65535;
/*! \brief suggested grid number for mapping kernel */
const int kBaseGridNum = 1024;
inline void CheckLaunchParam(dim3 dimGrid, dim3 dimBlock, const char *estr = "") {
	if (dimBlock.x * dimBlock.y * dimBlock.z > static_cast<unsigned>(kMaxThreadsPerBlock) ||
		dimGrid.x > 65535 || dimGrid.y > 65535) {
		LOG(FATAL) << "too large launch parameter: "
			<< estr << "["
			<< dimBlock.x << ","
			<< dimBlock.y << ","
			<< dimBlock.z << "]";
	}
}
}
}
#endif
