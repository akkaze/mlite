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
	string OCLCodeGen() {
		std::string matmul_kernel_src = " \
#define block_size $BLOCK_SIZE \
			__kernel __attribute__((reqd_work_group_size(block_size, block_size, 1))) \
			void matmul( \
				__global const $DType * A, \
				__global const unsigned* A_shp, \
				__global const $DType * B, \
				__global const unsigned* B_shp, \
				__global $DType * C) { \
			__local $DType bufA[block_size][block_size]; \
			__local $DType bufB[block_size][block_size]; \
			unsigned int row_block_id = get_group_id(0); \
			unsigned int col_block_id = get_group_id(1); \
			unsigned int row_thread_id = get_local_id(0); \
			unsigned int col_thread_id = get_local_id(1); \
			unsigned int row_id = get_global_id(0); \
			unsigned int col_id = get_global_id(1); \
			unsigned  int block_num = (A_shp[1] + block_size - 1) / block_size; \
			$DType Csub = 0; \
			for (unsigned int block = 0; block < block_num; block++) { \
				if (row_id < A_shp[0] && block * block_size + col_thread_id  < A_shp[1]) \
					bufA[row_thread_id][col_thread_id] = A[row_id * A_shp[1] + block * block_size + col_thread_id]; \
				else \
					bufA[row_thread_id][col_thread_id] = 0; \
				if (col_id < B_shp[1] && block * block_size + row_thread_id < B_shp[0]) \
					bufB[row_thread_id][col_thread_id] = B[(block * block_size + row_thread_id) * B_shp[1] + col_id]; \
				else \
					bufB[row_thread_id][col_thread_id] = 0; \
				barrier(CLK_LOCAL_MEM_FENCE); \
				for (unsigned int thread = 0; thread < block_size; thread++) \
					Csub += bufA[row_thread_id][thread] * bufB[thread][col_thread_id]; \
				barrier(CLK_LOCAL_MEM_FENCE); \
			} \
			if (row_id < A_shp[0] && col_id < B_shp[1]) \
				C[row_id * B_shp[1] + col_id] = Csub; \
		} \
		}";
		return matmul_kernel_src;
	}
	void Execute(Tensor<ocl, DType>* dst) {
		
	}
};
}
#endif // !MLITE_OCL_OPERATION_H_
