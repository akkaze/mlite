#ifndef MLITE_OPERATION_H_
#define MLITE_OPERARION_H_
#include <vector>
#include <type_traits>
#include "tensor.h"
//#include "type_traits.h"

namespace mlite {
enum ReturnType {
	kSimple = 0,
	kInplace = 1,
};
template <ReturnType return_type>
struct is_not_inplace {
	static const bool value = false;
};
template <>
struct is_not_inplace<ReturnType::kSimple> {
	static const bool value = true;
};

template <typename xpu,
	ReturnType return_type,
	typename DType MLITE_DEFAULT_DTYPE>
class Operation {
public:
	Operation() = default;
	~Operation() = default;
	Operation(const std::vector<Tensor<xpu, DType>*>& operands) {
		operands_ = operands;
	}
	virtual typename std::enable_if<
		is_not_inplace<return_type>::value,
		Shape>::type
		ReturnShape() = 0;
	virtual void Execute(Tensor<xpu, DType>*) = 0;
	///////////////////////getters//////////////////////////////
	const ReturnType& get_return_type() const {
		return return_type_;
	}

	std::vector<Tensor<xpu, DType>*> operands_;
	Shape dst_shape_;
};
template <typename xpu,
	ReturnType return_type,
	typename DType MLITE_DEFAULT_DTYPE>
class UnaryOperation : public Operation<xpu, return_type, DType> {
public:
	UnaryOperation() = default;
	~UnaryOperation() = default;
	UnaryOperation(Tensor<xpu,DType>* src) : 
		Operation<xpu, DType>({ src }) {}
};
//@breif silce operation,take a tensor and ranges return subtensor
template <typename xpu,
	typename DType MLITE_DEFAULT_DTYPE>
class Slice : UnaryOperation<xpu,ReturnType::kSimple, DType> {
public:
	Slice() = default;
	~Slice() = default;
	Slice(Tensor<xpu, DType>* src,
		const std::vector<Range>& ranges) :
		UnaryOperation<xpu,DType>(src), 
		ranges_(ranges)
		{
			CHECK_EQ(ranges.size(), operands_[0]->dims());
		}
	virtual Shape ReturnShape() {
		dst_shape_.Resize(operands_[0]->dims());
#pragma unroll
		for (index_t i = 0; i < src.dims(); i++)
			dst_shape_[i] = ranges[i].size();
		CHECK_GT(dst_shape_.Size(), 0);
		return dst_shape_;
	}
protected:
	std::vector<Range> ranges_;
};

template <typename xpu,
	typename DType MLITE_DEFAULT_DTYPE>
class Random : public UnaryOperation<xpu,ReturnType::kInplace,DType> {
public:
	Random() = default;
	~Random() = default;
	Random(Tensor<xpu,DType>* src):
		UnaryOperation<xpu, DType>(src) {}
	void Execute(Tensor<xpu, DType>* dst) {}
};
template <typename xpu,
	typename DType MLITE_DEFAULT_DTYPE>
class UniformRandom : Random<xpu, DType> {
public:
	UniformRandom() = default;
	~UniformRandom() = default;
	UniformRandom(Tensor<xpu,DType>* src,
		const DType& min,
		const DType& max):
		Random<xpu,DType>(src),min_(min),max_(max) {}
protected:
	DType min_;
	DType max_;
};
//template <typename xpu, typename DType, typename Op>
//class Map : UnaryOperation<xpu,DType> {
//public:
//	Map() = default;
//	~Map() = default;
//	Map(Tensor<xpu,DType>* src): 
//		UnaryOperation<xpu, DType>(src),
//		return_type_(ReturnType::kSimple) {}
//	const Shape& ReturnShape() {
//		return operands_[0]->shape();
//	}
//};
}
#endif
