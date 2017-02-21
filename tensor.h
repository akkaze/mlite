#ifndef MLITE_TENSOR_H_
#define MLITE_TENSOR_H_
#include <string>
#include <vector>
#include <initializer_list>
#include <iostream>
#include "./base.h"
#include "./logging.h"

namespace mlite {
/*! \brief device name CPU */
struct cpu {
	/*! \brief whether this device is CPU or not */
	static const bool kDevCPU = true;
	/*! \brief device flag number, identifies this device */
	static const int kDevMask = 1 << 0;
};
/*! \brief device name GPU */
struct gpu {
	/*! \brief whether this device is CPU or not */
	static const bool kDevCPU = false;
	/*! \brief device flag number, identifies this device */
	static const int kDevMask = 1 << 1;
};
template<int ndim>
struct Shape;

/**
* \brief a range [begin, end)
*/
class Range {
public:
	Range() : Range(0, 0) {}
	Range(index_t begin, index_t end) : begin_(begin), end_(end) { }
	Range(const std::initializer_list<index_t>& init_list) {
		CHECK_EQ(init_list.size(), 2);
		begin_ = *(init_list.begin());
		end_ = *(init_list.end() - 1);
	}
	index_t begin() const { return begin_; }
	index_t end() const { return end_; }
	index_t size() const { return end_ - begin_; }
private:
	index_t begin_;
	index_t end_;
};
/**
* \brief a indices (item1, item2, ..., itemN)
*/
class Indices {
public:
	Indices() { dptr_ = NULL; }
	Indices(const size_t& size):Indices() {
		size_ = size;
		dptr_ = new index_t[size];
		memset(dptr_, 0, sizeof(index_t) * size_);
	}
	Indices(const std::initializer_list<index_t>& init_list):Indices() {
		this->Resize(init_list.size());
		index_t idx = 0;
		for (auto& item : init_list) {
			*(dptr_ + idx) = item;
			idx++;
		}
	}
	Indices(const Indices& other):Indices() {
		this->Resize(other.size());
		memcpy(dptr_, other.data(), sizeof(index_t) * size_);
	}
	
	~Indices() {
		if (dptr_) 
			delete[] dptr_;
		size_ = 0;
	}
	void Resize(size_t new_size) {
		if (dptr_) 
			delete[] dptr_;
		size_ = new_size;
		dptr_ = new index_t[new_size];
	}
	Indices& operator=(Indices &other) {
		this->Resize(other.size());
		memcpy(dptr_, other.data(), sizeof(index_t) * size_);
		return *this;
	}
	Indices& operator=(const std::initializer_list<index_t>& init_list) {
		this->Resize(init_list.size());
		index_t idx = 0;
		for (auto& item : init_list) {
			*(dptr_ + idx) = item;
			idx++;
		}
		return *this;
	}
	index_t& operator[](index_t idx) { return *(dptr_ + idx); }
	const index_t& operator[](index_t idx) const { return *(dptr_ + idx); }
	index_t* data() const { return dptr_; }
	index_t size() const { return size_; }


	friend std::ostream &operator<<(std::ostream &, const Indices&);
private:
	index_t* dptr_;
	size_t size_;
};
std::ostream &operator<<(std::ostream &os, const Indices& indices) {
	for (index_t i = 0; i < indices.size_; i++)
		os << indices[i] << '\t';
	return os;
}
/*!
* \brief shape of a tensor
* \tparam dimension dimension of tensor
*/
template<int dimension>
struct Shape {
	/*! \brief dimension of current shape */
	static const int kDimension = dimension;
	/*! \brief string the dimension information */
	index_t shape_[kDimension];
	/*! \brief default constructor, do nothing */
	MLITE_XINLINE Shape(void) {}
	/*! \brief constuctor */
	MLITE_XINLINE Shape(const Shape<kDimension> &s) {
#pragma unroll
		for (int i = 0; i < kDimension; ++i) {
			this->shape_[i] = s[i];
		}
	}
	MLITE_XINLINE Shape(const std::initializer_list<int>& init_list) {
#pragma unroll
		for (int i = 0; i < kDimension; i++) {
			this->shape_[i] = init_list[i];
		}
	}
	MLITE_XINLINE index_t &operator[](index_t idx) {
		return shape_[idx];
	}
	MLITE_XINLINE const index_t &operator[](index_t idx) const {
		return shape_[idx];
	}
	MLITE_XINLINE bool operator==(const Shape<kDimension> &s) const {
#pragma unroll
		for (int i = 0; i < kDimension; ++i) {
			if (s.shape_[i] != this->shape_[i]) return false;
		}
		return true;
	}
	MLITE_XINLINE bool operator!=(const Shape<kDimension> &s) const {
		return !(*this == s);
	}
	MLITE_XINLINE Shape<1> FlatTo1D(void) const {
		Shape<1> s;
		s[0] = this->Size();
		return s;
	}
	/*! \return number of valid elements */
	MLITE_XINLINE index_t Size(void) const {
		index_t size = this->shape_[0];
#pragma unroll
		for (int i = 1; i < kDimension; ++i) {
			size *= this->shape_[i];
		}
		return size;
	}
	MLITE_XINLINE index_t ProdShape(int dimstart, int dimend) const {
		index_t num = 1;
#pragma unroll
		for (int i = dimstart; i < dimend; ++i) {
			num *= this->shape_[i];
		}
		return num;
	}

	template<int dimstart, int dimend>
	MLITE_XINLINE Shape<dimend - dimstart> Slice(void) const {
		Shape<dimend - dimstart> s;
#pragma unroll
		for (int i = dimstart; i < dimend; ++i) {
			s[i - dimstart] = this->shape_[i];
		}
		return s;
	}
};  // Shape
MLITE_XINLINE Shape<1> Shape1(index_t s0) {
	Shape<1> s; s[0] = s0;
	return s;
}
MLITE_XINLINE Shape<2> Shape2(index_t s0, index_t s1) {
	Shape<2> s; s[0] = s0; s[1] = s1;
	return s;
}
MLITE_XINLINE Shape<3> Shape3(index_t s0, index_t s1,index_t s2) {
	Shape<3> s; s[0] = s0; s[1] = s1; s[2] = s2;
	return s;
}
MLITE_XINLINE Shape<4> Shape4(index_t s0, index_t s1, index_t s2, index_t s3) {
	Shape<4> s; s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
	return s;
}
/*!
* \brief computaion stream structure, used for asynchronize computation
*/
template<typename Device>
struct Stream {
	MLITE_XINLINE void Wait(void) {}
	MLITE_XINLINE bool CheckIdle(void) {
		return true;
	}
	MLITE_XINLINE void CreateBlasHandle() {}
};
template<typename Device, int dimension,
	typename DType MLITE_DEFAULT_DTYPE>
struct Tensor  {
public:
	static const bool kDevCPU = Device::kDevCPU;
	DType *dptr_;
	Shape<dimension> shape_;
	Indices strides_;
	Stream<Device> *stream_;
	MLITE_XINLINE Tensor(void) : stream_(NULL) {
		this->GetStrides();
	}
	MLITE_XINLINE Tensor(const Shape<dimension> &shape)
		: shape_(shape), stream_(NULL) {
		this->GetStrides();
	}
	MLITE_XINLINE Tensor(DType *dptr, const Shape<dimension> &shape)
		: dptr_(dptr), shape_(shape),stream_(NULL) {
		this->GetStrides();
	}
	MLITE_XINLINE Tensor(DType *dptr, const Shape<dimension> &shape,
		Stream<Device> *stream)
		: dptr_(dptr), shape_(shape), stream_(stream) {
		this->GetStrides();
	}
	MLITE_XINLINE Tensor(DType *dptr,
		const Shape<dimension> &shape,
		index_t stride, Stream<Device> *stream)
		: dptr_(dptr), shape_(shape), stream_(stream) {
		this->GetStrides();
	}
	MLITE_XINLINE void set_stream(Stream<Device> *stream) {
		this->stream_ = stream;
		this->GetStrides();
	}
	MLITE_XINLINE size_t MemSize(int startdim) const {
		size_t memsz = 1;
//#pragma unroll
		for (int i = startdim; i < dimension; ++i) {
			memsz *= this->shape_[i];
		}
		return memsz;
	}
	MLITE_XINLINE size_t size(index_t idx) const {
		return shape_[idx];
	}
	MLITE_XINLINE Tensor<Device, 1, DType> FlatTo1D(void) const {
		return Tensor<Device, 1, DType>(dptr_, shape_.FlatTo1D(), stream_);
	}
	MLITE_XINLINE void FlatTo1D(void) {
		shape_ = shape_.FlatTo1D();
		this->GetStrides();
	}
	MLITE_XINLINE Tensor<Device, dimension, DType> &
		operator=(const Tensor<Device, dimension, DType> &other) {
		dptr_ = other.dptr_;
		shape_ = other.shape_;
		stream_ = other.stream_;
		stride_ = other.strides_;
		return *this;
	}
	MLITE_XINLINE void GetStrides(void) {
		strides_.Resize(dimension);
		for (index_t i = 0; i < dimension - 1; i++) {
			strides_[i] = MemSize(i + 1);
		}
		strides_[dimension - 1] = 1;
	}
	MLITE_XINLINE DType operator[](const Indices& indices) {
		index_t physical_index = IndexLogicalToPhysical(indices);
		return *(dptr_ + physical_index);
	}
	///////////////////////////////////////////////////////
	MLITE_XINLINE Indices IndexPhysicalToLogical(
		const index_t& physical_index) { 
		Indices logical_indices(dimension);
		index_t res_dim = physical_index;
		for (index_t i = dimension - 1; i > 0; i--) {
			logical_indices[i] = res_dim % strides_[i];
			res_dim /= strides_[i];
		}
		logical_indices[0] = res_dim;
		return logical_indices;
	}
	MLITE_XINLINE index_t IndexLogicalToPhysical(
		Indices logical_indices) {
		index_t physical_index = 0;
		for (index_t i = 0; i < dimension; i++) {
			physical_index += logical_indices[i] * strides_[i];
		}
		
		return physical_index;
	}

	/////////////////////////////////////////////////////////////////
	/*! \set given indices with datum */
	MLITE_XINLINE void Set(Indices indices, const DType& datum) {
				
	}
	MLITE_XINLINE void Reshape(const Shape<dimension>& dst_shape) {
		CHECK_EQ(dst_shape.Size(), shape_.Size()) <<
			"original and transformed shape must have same size";
		shape_ = dst_shape;
	}
	MLITE_XINLINE void Transpose(Indices indices) {
		Shape<dimension> orig_shape = shape_;
		Indices orig_indices = indices_;
		for (index_t i = 0; i < dimension; i++) {
			shape_[indices[i]] = orig_shape[i];
			strides_[indices[i]] = orig_indices[i];
		}
	}
};
template<typename Device>
inline void InitTensorEngine(int device_id = 0);
template<typename Device>
inline void ShutdownTensorEngine(void);
template<typename Device>
inline void SetDevice(int devid);
template<typename Device>
inline Stream<Device> *NewStream(bool create_blas_handle,
	bool create_dnn_handle);
template<typename Device>
inline Stream<Device> *NewStream() {
	return NewStream<Device>(true, false);
}
template<typename Device>
inline void DeleteStream(Stream<Device> *stream);
template<int dim, typename DType>
inline void FreeSpace(Tensor<cpu, dim, DType> *obj);
template<int dim, typename DType>
inline void FreeSpace(Tensor<gpu, dim, DType> *obj);
template<typename Device, typename DType, int dim>
inline Tensor<Device, dim, DType> NewTensor(const Shape<dim> &shape,
	DType initv,
	Stream<Device> *stream = NULL);
template<int dim, typename DType>
inline void Copy(Tensor<cpu, dim, DType> dst,
	const Tensor<cpu, dim, DType> &src,
	Stream<cpu> *stream = NULL);
template<int dim, typename DType>
inline void Copy(Tensor<cpu, dim, DType> dst,
	const Tensor<gpu, dim, DType> &src,
	Stream<gpu> *stream = NULL);
template<int dim, typename DType>
inline void Copy(Tensor<gpu, dim, DType> dst,
	const Tensor<cpu, dim, DType> &src,
	Stream<gpu> *stream = NULL);
}
#endif