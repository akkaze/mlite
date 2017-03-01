#ifndef MLITE_BASE_H_
#define MLITE_BASE_H_
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif
#define NOMINMAX
#endif
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <cfloat>
#include <climits>
#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <numeric>
#include <random>
#include <type_traits>
#ifdef _MSC_VER
typedef signed char int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned char uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#else
#include <inttypes.h>
#endif
// macro defintiions
/*!
* \brief if this macro is define to be 1,
* mshadow should compile without any of other libs
*/
#ifndef MLITE_STAND_ALONE
#define MLITE_STAND_ALONE 0
#endif


#if MLITE_STAND_ALONE
#define MLITE_USE_CBLAS 0
#define MLITE_USE_MKL   0
#define MLITE_USE_CUDA  0
#endif

/*!
* \brief force user to use GPU stream during computation
*  error will be shot when default stream NULL is used
*/
#ifndef MLITE_FORCE_STREAM
#define MLITE_FORCE_STREAM 1
#endif
/*! \brief use CBLAS for CBLAS */
#ifndef MLITE_USE_CBLAS
#define MLITE_USE_CBLAS 0
#endif
/*! \brief use MKL for BLAS */
#ifndef MLITE_USE_MKL
#define MLITE_USE_MKL   0
#endif
/*!
* \brief use CUDA support, must ensure that the cuda include path is correct,
* or directly compile using nvcc
*/
#ifndef MLITE_USE_CUDA
#define MLITE_USE_CUDA   0
#endif
/*!
* \brief use CUDA support, must ensure that the cuda include path is correct,
* or directly compile using nvcc
*/
#ifndef MLITE_USE_OCL
#define MLITE_USE_OCL    1
#endif
/*!
* \brief use CUDNN support, must ensure that the cudnn include path is correct
*/
#ifndef MLITE_USE_CUDNN
#define MLITE_USE_CUDNN 0
#endif
/*!
* \brief seems CUDAARCH is deprecated in future NVCC
* set this to 1 if you want to use CUDA version smaller than 2.0
*/
#ifndef MLITE_OLD_CUDA
#define MLITE_OLD_CUDA 0
#endif
/*!
* \brief macro to decide existence of c++11 compiler
*/
#ifndef MLITE_IN_CXX11
#define MLITE_IN_CXX11 (defined(__GXX_EXPERIMENTAL_CXX0X__) ||\
                          __cplusplus >= 201103L || defined(_MSC_VER))
#endif
/*! \brief whether use SSE */
#ifndef MLITE_USE_SSE
#define MLITE_USE_SSE 1
#endif
/*! \brief whether use NVML to get dynamic info */
#ifndef MLITE_USE_NVML
#define MLITE_USE_NVML 0
#endif
// SSE is conflict with cudacc
#ifdef __CUDACC__
#undef MLITE_USE_SSE
#define MLITE_USE_SSE 0
#endif
#if MLITE_USE_CBLAS
extern "C" {
#include <cblas.h>
}
#elif MLITE_USE_MKL
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_vsl.h>
#include <mkl_vsl_functions.h>
#endif
#if MLITE_USE_OCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif
#endif
#if MLITE_USE_CUDA
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#endif
#if MLITE_USE_CUDNN
#include <cudnn.h>
#endif
#if MLITE_USE_NVML
#include <nvml.h>
#endif
// --------------------------------
// MLITE_XINLINE is used for inlining template code for both CUDA and CPU code
#ifdef MLITE_XINLINE
#error "MLITE_XINLINE must not be defined"
#endif
#ifdef _MSC_VER
#define MLITE_FORCE_INLINE __forceinline
#pragma warning(disable : 4068)
#else
#define MLITE_FORCE_INLINE inline __attribute__((always_inline))
#endif
#ifdef __CUDACC__
#define MLITE_XINLINE MLTE_FORCE_INLINE __device__ __host__
#else
#define MLITE_XINLINE MLITE_FORCE_INLINE
#endif
/*! \brief cpu force inline */
#define MLITE_CINLINE MLITE_FORCE_INLINE

#if defined(__GXX_EXPERIMENTAL_CXX0X) ||\
    defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L
#define MLITE_CONSTEXPR constexpr
#else
#define MLITE_CONSTEXPR const
#endif

/*!
* \brief default data type for tensor string
*  in code release, change it to default_real_t
*  during development, change it to empty string so that missing
*  template arguments can be detected
*/
#ifndef MLITE_DEFAULT_DTYPE
#define MLITE_DEFAULT_DTYPE = default_real_t
#endif
#if MLITE_IN_CXX11	
#define MLITE_THROW_EXCEPTION noexcept(false)
#define MLITE_NO_EXCEPTION  noexcept(true)
#else
#define MLITE_THROW_EXCEPTION
#define MLITE_NO_EXCEPTION
#endif

/*!
* \brief Protected cuda call in mLITE
* \param func Expression to call.
* It checks for CUDA errors after invocation of the expression.
*/
#define MLITE_CUDA_CALL(func)                                      \
  {                                                                \
    cudaError_t e = (func);                                        \
    if (e == cudaErrorCudartUnloading) {                           \
      throw dmlc::Error(cudaGetErrorString(e));                    \
    }                                                              \
    CHECK(e == cudaSuccess)                                        \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

/*!
* \brief Run function and catch error, log unknown error.
* \param func Expression to call.
*/
#define MLITE_CATCH_ERROR(func)                                         \
  {                                                                     \
    try {                                                               \
      (func);                                                           \
    } catch (const mlite::Error &e) {                                   \
      std::string what = e.what();                                      \
      if (what.find("driver shutting down") == std::string::npos) {     \
        LOG(ERROR) << "Ignore CUDA Error " << what;                     \
      }                                                                 \
    }                                                                   \
  }

/*! \brief namespace for MLITE */
namespace mlite {
/*! \brief buffer size for each random number generator */
const unsigned kRandBufferSize = 1000000;
/*! \brief pi  */
const float kPi = 3.1415926f;
/*! \brief type that will be used for index */
typedef unsigned index_t;
const index_t kMaxIndex = std::numeric_limits<index_t>::max();
#ifdef _WIN32
/*! \brief openmp index for windows */
typedef int64_t openmp_index_t;
#else
/*! \brief openmp index for linux */
typedef index_t openmp_index_t;
#endif
typedef float default_real_t;

}

#endif