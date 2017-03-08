#ifndef MLITE_OPERATION_H_
#define MLITE_OPERARION_H_
#include <vector>
#include <type_traits>

#include "tensor.h"
#include "type_traits.h"
namespace mlite {
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
	virtual typename enable_if<
		is_not_inplace<return_type>::value, 
		std::add_lvalue_reference<Shape>::type>::type 
		ReturnShape() = 0;
#if MLITE_USE_OCL
	 typename enable_if<
		is_ocl<xpu>::value,std::string>::type
		OCLCodeGen();
#endif
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
		Operation<xpu,return_type, DType>({ src }) {}
};
template <typename xpu,
	ReturnType return_type,
	typename DType MLITE_DEFAULT_DTYPE>
	class BinaryOperation : public Operation<xpu, return_type, DType> {
	public:
		BinaryOperation() = default;
		~BinaryOperation() = default;
		BinaryOperation(Tensor<xpu, DType>* operand1, 
			Tensor<xpu, DType>* operand2) :
			Operation<xpu, return_type, DType>({ operand1,operand2 }) {}
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
		ranges_(ranges) {
			CHECK_EQ(ranges.size(), operands_[0]->dims());
		}
	const Shape& ReturnShape() {
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
		UnaryOperation<xpu,ReturnType::kInplace, DType>(src) {}
	void ReturnShape() {
		LOG(ERROR) << "NOT IMPLEMENTED!";
	}
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
//@brief elementwise operation
template <typename xpu,
	typename DType, 
	ReturnType return_type,
	typename Op>
class ElementWise : UnaryOperation<xpu, return_type, DType> {
public:
	ElementWise() = default;
	~ElementWise() = default;
	ElementWise(Tensor<xpu,DType>* src):
		UnaryOperation<xpu, DType>(src) {}
	const Shape& ReturnShape() {
		return operands_[0]->shape();
	}
};

////////////////////////binary operations///////////////////////////////////
//@breif matrix multiplication
template <typename xpu,typename DType>
class MatMul : BinaryOperation<xpu, ReturnType::kSimple, DType> {
public:
	MatMul() = default;
	~MatMul() = default;
	MatMul(Tensor<xpu, DType>* operand1,
		Tensor<xpu, DType>* operand2) : 
		BinaryOperation<xpu, ReturnType::kSimple, DType>(operand1,operand2) {
		CHECK_EQ(operand1->dims(), 2) << "Tensor must be a matrix!";
		CHECK_EQ(operand2->dims(), 2) << "Tensor must be a matrix!";
		CHECK_EQ(operand1->dim_size(1), operand2->dim_size(0))
			<< "column number of left matrix must equal to row number of right matrix!";
	}
	const Shape& ReturnShape() {
		return Shape({ operands_[0].dim_size(0),operands_[1].dim.size() });
	}
};
}
#endif
