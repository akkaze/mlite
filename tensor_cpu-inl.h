#ifndef MLITE_TENSOR_CPU_INL_H_
#define MLITE_TENSOR_CPU_INL_H_
#include <cstring>
#include <functional>
#include <utility>
#include <vector>
#include "./base.h"
#include "./tensor.h"
namespace mlite {
template<>
MLITE_XINLINE void InitTensorEngine<cpu>(int dev_id){
}
template<>
MLITE_XINLINE void ShutdownTensorEngine<cpu>(void) {
}
template<>
MLITE_XINLINE void SetDevice<cpu>(int devid) {
}
template<>
inline Stream<cpu> *NewStream<cpu>(bool create_blas_handle,
	bool create_dnn_handle) {
	return new Stream<cpu>();
}
template<>
MLITE_XINLINE void DeleteStream<cpu>(Stream<cpu> *stream) {
	delete stream;
}
template<int dim, typename DType>
MLITE_XINLINE void AllocSpace(Tensor<cpu, dim, DType> *obj) {
	obj->dptr_ = reinterpret_cast<DType*>(
		malloc(obj->shape_.Size() * sizeof(DType)));
}
template<typename Device,typename DType>
MLITE_XINLINE Tensor<Device, dim, DType>
NewTensor(const Shape& shape, 
	DType initv, Stream<Device> *stream_) {
	Tensor<Device, dim, DType> obj(shape);
	obj.stream_ = stream_;
	AllocSpace(&obj);
	for (index_t i = 0; i < shape.Size(); i++)
		*(obj.dptr(i)) = initv;
	return obj;
}
template<int dim, typename DType>
MLITE_XINLINE void FreeSpace(Tensor<cpu, dim, DType> *obj) {
	free(obj->data());
	obj->data() = NULL;
}
template<int dim, typename DType>
MLITE_XINLINE void Copy(Tensor<cpu, dim, DType> _dst,
	const Tensor<cpu, dim, DType> &_src,
	Stream<cpu> *stream) {
	CHECK_EQ(_dst.shape(), _src.shape())
		<< "Copy:shape mismatch:" << _dst.shape_ << " vs " << _src.shape_;
	memcpy(_dst.dptr_, _src.dptr_, sizeof(DType) * _dst.shape_.Size());
}
template<typename Op,int dim,typename DType>
MLITE_XINLINE void Map(Tensor<cpu, dim, DType>& ts) {
	size_t size = ts.shape_.Size();
#pragma omp parallel for
	for (index_t i = 0; i < size; i++)
		*(ts.dptr_ + i) = Op()(*(ts.dptr_ + i));
}
template<int dim,typename DType>
MLITE_XINLINE Tensor<cpu, dim, DType> Slice(const Tensor<cpu, dim, DType>& src,
	const std::vector<Range>& ranges) {
	CHECK_EQ(ranges.size(), dim);
	Shape<dim> dst_shape;
#pragma unroll
	for (index_t i = 0; i < dim; i++)
		dst_shape[i] = ranges[i].size();
	CHECK_GT(dst_shape.Size(), 0);
	Tensor<cpu, dim, DType> dst = NewTensor(dst_shape, 0, src.stream_);
	//datum index in the destination tensor data array
	index_t dst_id = 0;
	Indices indices;
	for (dst_id = 0; dst_id < dst.size(); dst_id++) {
		indices = dst.IndexPhysicalToLogical(dst_id);
		for (index i = 0; i < dim; i++)
			indices[i] += ranges[i].begin();
		index_t src_id = 0;
		src.IndexLogicalToPhysical(src_id, indices);
		*(dst.dptr_ + dst_id) = *(src.dptr_ + src_id);
	}
}

template<typename Op,int dim,typename DType>
MLITE_XINLINE Tensor<cpu, dim, DType> ReduceOverAxis(
	const Tensor<cpu, dim, DType>& src,
	const index_t& axis) {
	Shape<dim - 1> dst_shape;
	const Shape<dim>& src_shape = src.shape_;
	for (index_t i = 0; i < dim; i++) {
		if (i == axis)
			continue;
		else if (i < axis)
			dst_shape[i] = src_shape[i];
		else if (i > axis)
			dst_shape[i - 1] = src_shape[i];
	}
	CHECK_GT(dst_shape.Size(), 0);
	Tensor<cpu, dim, DType> dst = NewTensor(dst_shape, 0, src.stream_);
	//datum index in the destination tensor data array
	index_t dst_id = 0;
	Indices dst_indices;
	for(dst_id = 0; dst_id < dst.size(); dst_id++) {
		dst.IndexPhysicalToLogical(dst_id, dst_indices);
		index_t src_indices[dim];
		for (index_t i = 0; i < src.shape_[axis]; i++) {
			for (index_t j = 0; j < dim; j++) {
				if (j == axis)
					src_indices[j] = i;
				else if (j < axis)
					src_indices[j] = dst_indices[j];
				else if (j > axis)
					src_indices[j] = dst_indices[j + 1];
			}
			index_t src_id = 0;
			src.IndexLogicalToPhysical(src_id, src_indices);
			*(dst.dptr_ + dst_id) = 
				Op()(*(dst.dptr_ + dst_id),*(src.dptr_ + src_id));
		}
	}
}
}

#endif