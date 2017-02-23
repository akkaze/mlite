#ifndef MLITE_OPERATION_H_
#define MLITE_OPERARION_H_
#include <vector>

#include "tensor.h"
namespace mlite {
enum ReturnType {
	kSimple = 0,
	kInplace = 1,
};
template <typename xpu,typename DType>
class Operation {
public:
	Operation() = default;
	~Operation() = default;
	Operation(const std::vector<Tensor<xpu, DType>*>& operands) {
		operands_ = operands;
	}
	virtual Shape ReturnShape() = 0;
	virtual Execute(Tensor<xpu, DType>*) = 0;
	////////////////getters//////////////////////////////
	const ReturnType& get_return_type() const {
		return return_type_;
	}
protected:
	std::vector<Tensor<xpu, DType>*> operands_;
	Shape dst_shape_;
	ReturnType return_type_;
};
template <typename xpu,typename DType>
class Slice : Operation<xpu, DType> {
public:
	Slice() = default;
	~Slice() = default;
	Slice(const std::vector<Tensor<xpu, DType>*>& operands,
		const std::vector<Range>& ranges) : 
		Operation(operands), ranges_(ranges) {
		CHECK_EQ(operand.size(), 1);
		CHECK_EQ(ranges.size(), operands[0]->dims());
	}
	virtual Shape ReturnShape() {
		dst_shape_.Resize(operands[0]->dims());
#pragma unroll
		for (index_t i = 0; i < src.dims(); i++)
			dst_shape_[i] = ranges[i].size();
		CHECK_GT(dst_shape_.Size(), 0);
		return dst_shape_;
	}
protected:
	std::vector<Range> ranges_;
};
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
			*dst->dptr(dst_id) = *src->dptr(src_id);
		}
	}
};
}
#endif
