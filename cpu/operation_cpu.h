#ifndef MLITE_OPERATION_CPU_H_
#define MLITE_OPERATION_CPU_H_

#include "../operation.h"
#include "tensor_cpu-inl.h"

namespace mlite{
////////////////////////////unary operations/////////////////////////
//@brief cpu tensor class
template <typename DType>
class Slice<cpu, DType> {
	void Execute(Tensor<cpu, DType>* dst) {
		//get source tensor
		Tensor<cpu, DType>* src = operands_[0];
		index_t dst_id = 0;
		Indices indices;
		for (dst_id = 0; dst_id < dst->size(); dst_id++) {
			indices = dst->Index1DToND(dst_id);
			for (index i = 0; i < src->sdims(); i++)
				indices[i] += ranges_[i].begin();
			index_t src_id = 0;
			src->IndexNDTo1D(src_id, indices);
			*(dst->dptr(dst_id)) = *(src->dptr(src_id));
		}
	}
};

template <typename DType>
class UniformRandom<cpu,DType>:public Random<cpu,DType> {
public:
	UniformRandom() = default;
	~UniformRandom() = default;
	UniformRandom(Tensor<cpu, DType>* src,
		const DType& min,
		const DType& max) :
		Random<cpu, DType>(src), min_(min), max_(max) {}
	void Execute(Tensor<cpu,DType>* dst) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<DType> dis(min_, max_);
#pragma omp parallel for
		for (index_t i = 0; i < operands_[0]->size(); i++)
			*(dst->dptr(i)) = dis(gen);
	}
protected:
	DType min_;
	DType max_;
};

template <typename DType,
	ReturnType return_type,
	typename Op>
class  ElementWise<cpu, DType,return_type, Op> :
	UnaryOperation<cpu,return_type,DType>{
	ElementWise() = default;
	~ElementWise() = default;
	ElementWise(Tensor<cpu, DType>* src) :
		UnaryOperation<cpu, DType>(src) {}
	void Execute(Tensor<cpu, DType>* dst) {
#pragma omp parallel for
		for (index_t i = 0; i < operands_[0]->size(); i++)
			*(dst->dptr(i)) = Op()(*(operands_[0]->dptr(i)));
	}
};
/////////////////////////////binary operations//////////////////////////////////
template <typename DType>
class MatMul<cpu,DType> : BinaryOperation<cpu, ReturnType::kSimple, DType> {
public:
	MatMul() = default;
	~MatMul() = default;
	MatMul(Tensor<cpu, DType>* operand1,
		Tensor<cpu, DType>* operand2) :
		BinaryOperation<cpu, ReturnType::kSimple, DType>(operand1, operand2) {
		CHECK_EQ(operand1->dims(), 2) << "Tensor must be a matrix!";
		CHECK_EQ(operand2->dims(), 2) << "Tensor must be a matrix!";
		CHECK_EQ(operand1->dim_size(1), operand2->dim_size(0))
			<< "column number of left matrix must equal to row number of right matrix!";
	}
	Shape& ReturnShape() {
		return Shape({ operands_[0]->dim_size(0),operands_[1]->dim_size(1) });
	}
	void Execute(Tensor<cpu, DType>* dst) {
#pragma omp parallel for
		for (index_t r = 0; r < dst->dim_size(0); r++) {
			for (index_t c = 0; c < dst->dim_size(1); r++) {
				index_t dst_idx = dst->IndexNDTo1D({ r, c });
				for (index_t m = 0; m < operands_[0]->dim_size(1); m++) {
					index_t operand0_idx = operands_[0]->IndexNDTo1D({ c, m });
					index_t operand1_idx = operands_[1]->IndexNDTo1D({ m, r });
					*(dst->dptr(dst_idx)) += *(operands_[0]->dptr(operand0_idx)) *
						*(operands_[1]->dptr(operand1_idx));
				}
			}
		}
	}
};
} ///namespace mlite
#endif