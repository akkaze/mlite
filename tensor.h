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

struct ocl {
	static const bool kDevCPU = false;
	static const int kDevMask = 1 << 2;
};
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
	Indices() { indices_ = NULL; }
	Indices(const size_t& size):Indices() {
		size_ = size;
		indices_ = new index_t[size];
		memset(indices_, 0, sizeof(index_t) * size_);
	}
	Indices(const std::initializer_list<index_t>& init_list):Indices() {
		this->Resize(init_list.size());
		index_t idx = 0;
		for (auto& item : init_list) {
			*(indices_ + idx) = item;
			idx++;
		}
	}
	Indices(const Indices& other):Indices() {
		this->Resize(other.size());
		memcpy(indices_, other.data(), sizeof(index_t) * size_);
	}
	
	~Indices() {
		if (indices_)
			delete[] indices_;
		size_ = 0;
	}
	void Resize(size_t new_size) {
		if (indices_)
			delete[] indices_;
		size_ = new_size;
		indices_ = new index_t[new_size];
	}
	Indices& operator=(const Indices& other) {
		this->Resize(other.size());
		memcpy(indices_, other.data(), sizeof(index_t) * size_);
		return *this;
	}
	Indices& operator=(const std::initializer_list<index_t>& init_list) {
		this->Resize(init_list.size());
		index_t idx = 0;
		for (auto& item : init_list) {
			*(indices_ + idx) = item;
			idx++;
		}
		return *this;
	}
	index_t& operator[](index_t idx) { return *(indices_ + idx); }
	const index_t& operator[](index_t idx) const { return *(indices_ + idx); }
	index_t* data() const { return indices_; }
	index_t size() const { return size_; }


	friend std::ostream &operator<<(std::ostream &, const Indices&);
private:
	index_t* indices_;
	size_t size_;
};
std::ostream &operator<<(std::ostream &os, const Indices& indices) {
	for (index_t i = 0; i < indices.size_; i++)
		os << indices[i] << '\t';
	return os;
}
/*!
* \brief shape of a tensor
*/
struct Shape {
	/*! \brief dimension of current shape */
	size_t dims_;
	/*! \brief string the dimension information */
	index_t* shape_;
	/*! \brief default constructor, do nothing */
	MLITE_XINLINE Shape(void) { shape_ = NULL; }
	MLITE_XINLINE Shape(const size_t& dims) : Shape() {
		this->Resize(dims);
	}
	/*! \brief constuctor */
	MLITE_XINLINE Shape(const Shape &s) : Shape() {
		this->Resize(s.dims_);
		for (int i = 0; i < dims_; ++i) {
			this->shape_[i] = s[i];
		}
	}
	MLITE_XINLINE ~Shape() {
		if (shape_)
			delete[] shape_;
	}
	MLITE_XINLINE Shape(const std::initializer_list<int>& init_list) : Shape() {
		this->Resize(init_list.size());
		index_t idx = 0;
		for (auto& item : init_list) {
			*(shape_ + idx) = item;
			idx++;
		}
	}
	MLITE_XINLINE Shape& operator=(const Shape& other) {
		this->Resize(other.dims());
		memcpy(shape_, other.shape(), sizeof(index_t) * dims_);
		return *this;
	}
	MLITE_XINLINE Shape& operator=(const std::initializer_list<index_t>& init_list) {
		this->Resize(init_list.size());
		index_t idx = 0;
		for (auto& item : init_list) {
			*(shape_ + idx) = item;
			idx++;
		}
		return *this;
	}
	MLITE_XINLINE index_t &operator[](index_t idx) {
		return shape_[idx];
	}
	MLITE_XINLINE const index_t &operator[](index_t idx) const {
		return shape_[idx];
	}
	MLITE_XINLINE bool operator==(const Shape &s) const {
		if (this->dims_ != s.dims_)
			return false;
#pragma unroll
		for (int i = 0; i < dims_; ++i) {
			if (s.shape_[i] != this->shape_[i])
				return false;
		}
		return true;
	}
	MLITE_XINLINE bool operator!=(const Shape &s) const {
		return !(*this == s);
	}
	MLITE_XINLINE void Resize(size_t new_dims) {
		if (shape_)
			delete[] shape_;
		dims_ = new_dims;
		shape_ = new index_t[new_dims];
	}
	MLITE_XINLINE Shape FlatTo1D(void) const {
		Shape s;
		s.Resize(1);
		s[0] = this->Size();
		return s;
	}
	MLITE_XINLINE index_t dims() const {
		return dims_;
	}
	MLITE_XINLINE index_t* shape() const {
		return shape_;
	}
	/*! \return number of valid elements */
	MLITE_XINLINE index_t Size(void) const {
		index_t size = this->shape_[0];
#pragma unroll
		for (int i = 1; i < dims_; ++i) {
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
	MLITE_XINLINE Shape Slice(int dimstart,int dimend) const {
		Shape s;
#pragma unroll
		for (int i = dimstart; i < dimend; ++i) {
			s[i - dimstart] = this->shape_[i];
		}
		return s;
	}
};  // Shape
MLITE_XINLINE Shape Shape1(index_t s0) {
	Shape s(1); s[0] = s0;
	return s;
}
MLITE_XINLINE Shape Shape2(index_t s0, index_t s1) {
	Shape s(2); s[0] = s0; s[1] = s1;
	return s;
}
MLITE_XINLINE Shape Shape3(index_t s0, index_t s1,index_t s2) {
	Shape s(3); s[0] = s0; s[1] = s1; s[2] = s2;
	return s;
}
MLITE_XINLINE Shape Shape4(index_t s0, index_t s1, index_t s2, index_t s3) {
	Shape s(4); s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
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
template<typename Device,typename DType MLITE_DEFAULT_DTYPE>
struct Tensor  {
public:
	static const bool kDevCPU = Device::kDevCPU;
#if MLITE_USE_OCL
	cl_mem cl_data_;
#endif
	DType *dptr_;
	Shape shape_;
	Indices strides_;
	Stream<Device> *stream_;
	size_t dims_;
	MLITE_XINLINE Tensor(void) : stream_(NULL) {
	}
	MLITE_XINLINE Tensor(const Shape& shape)
		: dims_(shape.dims()), shape_(shape), stream_(NULL) {
		this->GetStrides();
	}
	MLITE_XINLINE Tensor(DType *dptr, const Shape& shape)
		: dptr_(dptr), dims_(shape.dims()), shape_(shape),stream_(NULL) {
		this->GetStrides();
	}
	MLITE_XINLINE Tensor(DType *dptr, const Shape& shape,
		Stream<Device> *stream)
		: dptr_(dptr), dims_(shape.dims()), shape_(shape), stream_(stream) {
		this->GetStrides();
	}
	MLITE_XINLINE Tensor(DType *dptr,
		const Shape &shape,
		index_t stride, Stream<Device> *stream)
		: dptr_(dptr), dims_(shape.dims()), shape_(shape), stream_(stream) {
		this->GetStrides();
	}
	///////////////getters////////////////////////////
	MLITE_XINLINE DType* data() const {
		return dptr_;
	}
#ifdef MLITE_USE_OCL
	MLITE_XINLINE const cl_mem& cl_data() const {
		return cl_data_;
	}
#endif
	MLITE_XINLINE DType* dptr(const index_t idx) const {
		return dptr_ + idx;
	}
	MLITE_XINLINE Shape shape() const {
		return shape_;
	}
	MLITE_XINLINE size_t size() const {
		return shape_.Size();
	}
	MLITE_XINLINE size_t dims() const {
		return dims_;
	}
	MLITE_XINLINE const Stream<Device>& stream() const {
		return stream_;
	}
	MLITE_XINLINE size_t MemSize(int startdim) const {
		size_t memsz = 1;
#pragma unroll
		for (int i = startdim; i < dims_; ++i) {
			memsz *= this->shape_[i];
		}
		return memsz;
	}
	MLITE_XINLINE size_t dim_size(index_t idx) const {
		return shape_[idx];
	}
	MLITE_XINLINE Tensor<Device, DType> FlatTo1D(void) const {
		return Tensor<Device, DType>(dptr_, shape_.FlatTo1D(), stream_);
	}
	MLITE_XINLINE void FlatTo1D(void) {
		dims_ = 1;
		shape_ = shape_.FlatTo1D();
		this->GetStrides();
	}
	MLITE_XINLINE Tensor<Device, DType> &
		operator=(const Tensor<Device, DType> &other) {
		dptr_ = other.dptr_;
		shape_ = other.shape_;
		stream_ = other.stream_;
		strides_ = other.strides_;
		dims_ = other.dims_;
		return *this;
	}
	MLITE_XINLINE void GetStrides(void) {
		strides_.Resize(dims_);
		for (index_t i = 0; i < dims_ - 1; i++) {
			strides_[i] = MemSize(i + 1);
		}
		strides_[dims_ - 1] = 1;
	}
	MLITE_XINLINE DType& operator[](const Indices& indices) {
		index_t physical_index = IndexNDTo1D(indices);
		return *(dptr_ + physical_index);
	}
	///////////////setters/////////////////////////////////
	MLITE_XINLINE void set_dptr(DType* dptr) {
		dptr_ = dptr;
	}
#ifdef MLITE_USE_OCL
	MLITE_XINLINE void set_cl_data(const cl_mem& cl_data) {
		cl_data_ = cl_data;
	}
#endif
	MLITE_XINLINE void set_stream(Stream<Device>* stream) {
		stream_ = stream;
	}
	///////////////////////////////////////////////////////
	MLITE_XINLINE Indices Index1DToND(
		const index_t& physical_index) {
		CHECK_LT(physical_index, shape_.Size()) << "\
			Physical index should be smaller than number of elements";
		Indices indices_nd(dims_);
		index_t res_dim = index_1d;
		for (index_t i = dims_ - 1; i > 0; i--) {
			indices_nd[i] = res_dim % strides_[i];
			res_dim /= strides_[i];
		}
		indices_nd[0] = res_dim;
		return indices_nd;
	}
	MLITE_XINLINE index_t IndexNDTo1D(
		Indices indices_nd) {
		index_t index_1d = 0;
		for (index_t i = 0; i < dims_; i++) {
			CHECK_LT(indices_nd[i], shape_[i]) << "\
				Logical index along an axis should be smaller than dimension of that aixs";
			index_1d += indices_nd[i] * strides_[i];
		}
		return index_1d;
	}

	/////////////////////////////////////////////////////////////////
	/*! \set given indices with datum */
	MLITE_XINLINE void Set(Indices indices, const DType& datum) {
		index_t idx = IndexNDTo1D(indices);
		std::cout << idx << std::endl;
		*(dptr_ + idx) = datum;
	}
	MLITE_XINLINE void Reshape(const Shape& dst_shape) {
		CHECK_EQ(dst_shape.Size(), shape_.Size()) <<
			"original and transformed shape must have same size";
		shape_ = dst_shape;
	}
	MLITE_XINLINE void Transpose(Indices indices) {
		Shape orig_shape = shape_;
		Indices orig_strides = strides_;
		for (index_t i = 0; i < dims_; i++) {
			shape_[indices[i]] = orig_shape[i];
			strides_[indices[i]] = orig_strides[i];
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
	return Stream<Device>(true, false);
}
template<typename Device>
inline void DeleteStream(Stream<Device> *stream);
template<typename DType>
inline void FreeSpace(Tensor<cpu, DType> *obj);
template<typename DType>
inline void FreeSpace(Tensor<gpu, DType> *obj);
template<typename Device, typename DType>
inline Tensor<Device, DType> NewTensor(const Shape &shape,
	DType initv,
	Stream<Device> *stream = NULL);
template<typename DType>
inline void Copy(Tensor<cpu, DType> dst,
	const Tensor<cpu, DType> &src,
	Stream<cpu> *stream = NULL);
template<typename DType>
inline void Copy(Tensor<cpu, DType> dst,
	const Tensor<gpu, DType> &src,
	Stream<gpu> *stream = NULL);
template<typename DType>
inline void Copy(Tensor<gpu, DType> dst,
	const Tensor<cpu, DType> &src,
	Stream<gpu> *stream = NULL);
}
#endif