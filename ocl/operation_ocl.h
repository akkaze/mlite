#ifndef MLITE_OCL_OPERATION_H_
#define MLITE_OCL_OPERATION_H_

#include "../tensor.h"
#include "../operation.h"
#include "executor.h"
#include "indices_ocl.h"
namespace mlite {
/////////////////////////////binary operations//////////////////////////////////
template <typename DType>
class MatMul<ocl, DType> : BinaryOperation<ocl, ReturnType::kSimple, DType> {
public:
	MatMul() = default;
	~MatMul() = default;
	MatMul(Tensor<ocl, DType>* operand1,
		Tensor<ocl, DType>* operand2) :
		BinaryOperation<ocl, ReturnType::kSimple, DType>(operand1, operand2) {
		CHECK_EQ(operand1->dims(), 2) << "Tensor must be a matrix!";
		CHECK_EQ(operand2->dims(), 2) << "Tensor must be a matrix!";
		CHECK_EQ(operand1->dim_size(1), operand2->dim_size(0))
			<< "column number of left matrix must equal to row number of right matrix!";
	}
	void Execute(Tensor<ocl, DType>* dst) {
		std::string matmul_kernel_src = "										\
#define TILE_DIM 32																\
			__kernel MatMulKernel(__global float* mat1,							\
				__global uint* shape1,											\
				__global float *mat2,											\
				__global uint* shape2,											\
				__global float* mat3) {											\
			__local float ds_M[TILE_DIM][TILE_DIM];								\
			__local float ds_N[TILE_DIM][TILE_DIM];								\
			int c = get_global_id(0);											\
			int r = get_global_id(1);											\
			int tx = get_local_id(0);											\
			int ty = get_local_id(1);											\
			float dst = 0;														\
			for (int m = 0; m < shape1[0] / TILE_DIM + 1; m++) {				\
				if (r < shape1[1] && m * TILE_DIM + tx < shape1[1])				\
					ds_M[ty][tx] = mat1[r * shape1[0] + m * TILE_DIM * tx];		\
				else															\
					ds_M[ty][tx] = 0;											\
				if (c < shape2[0] && m * TILE_DIM + ty < shape2[0])				\
					ds_N[ty][tx] = mat2[(m * TILE_DIM + ty) * shape2[0] + c];	\
				else															\
					ds_M[ty][tx] = 0;											\
				barrier(CLK_LOCAL_MEM_FENCE);									\
				for (int k = 0; k < TILE_DIM; k++)								\
					dst += ds_M[ty][k] * ds_N[k][tx];							\
				barrier(CLK_LOCAL_MEM_FENCE);									\
			}																	\
			if (r < shape2[1] && c < shape1[0])									\
				c[r * shape1[0] + col]	= dst;									\
		}";
	}
};
}
#endif // !MLITE_OCL_OPERATION_H_
