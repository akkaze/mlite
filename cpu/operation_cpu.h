#ifndef MLITE_OPERATION_CPU_H_
#define MLITE_OPERATION_CPU_H_

#include "operation.h"
#include "tensor_cpu-inl.h"

namespace mlite{
//@brief cpu tensor class
template <typename DType>
class Slice<cpu, DType> {
	void Execute(Tensor<cpu, DType>* dst) {
		//get source tensor
		Tensor<cpu, DType>* src = operands_[0];
		index_t dst_id = 0;
		Indices indices;
		for (dst_id = 0; dst_id < dst->size(); dst_id++) {
			indices = dst->IndexPhysicalToLogical(dst_id);
			for (index i = 0; i < src->sdims(); i++)
				indices[i] += ranges_[i].begin();
			index_t src_id = 0;
			src->IndexLogicalToPhysical(src_id, indices);
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

//template <typename DType,typename Op>
//class Map<cpu, DType, Op> {
//	void Execute(Tensor<cpu, DType>* dst) {
//#pragma omp parallel for
//		for (index_t i = 0; i < operands_[0]->size(); i++)
//			*(dst->dptr(i)) = *(operands_[0]->dptr(i));
//	}
//};
} ///namespace mlite
#endif