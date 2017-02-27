#ifndef MLITE_UTILS_SHARED_PTR_H_
#define MLITE_UTILS_SHARED_PTR_H_
namespace mlite {
// @brief Reference counting class for the shared_ptr implementation */
class count
{
public:
	count(unsigned int val) : val_(val) { }
	void dec() { --val_; }
	void inc() { ++val_; }
	bool is_null() { return val_ == 0; }
	unsigned int val() { return val_; }
private:
	unsigned int val_;
};
template <typename U>
struct default_deleter {
	void operator()(U* p) const {
		delete p;
	}
};
template <typename T,typename Deleter = default_deleter<T>>
class shared_ptr {
	//template<typename U,typename UDeleter>
	//friend class shared_ptr<U, UDeleter>;
	T* ptr_;
	count* count_;
	Deleter* deleter_;
protected:
	void destory(){
		deleter_(ptr_);
	}
public:
	shared_ptr() : count_(new count(1)), ptr_(NULL), deleter_(NULL) {}
	template <typename U,typename Deleter> 
	shared_ptr(U* pu, Deleter deleter) : 
		count_(new count(1)), ptr_(pu), deleter_(new Deleter()) {}
	template <typename U>
	shared_ptr(U* pu) : 
		count_(new count(1)),ptr_(pu),deleter_(new default_deleter<U>()) {}
	T* get() const {
		return ptr_;
	}
	T* operator->() const {
		return ptr_;
	}
	T& operator*() const {
		return *ptr_;
	}
	shared_ptr(const shared_ptr& other) :
		count_(other.count_), ptr_(other.ptr_), deleter_(other.deleter_) {
		inc();
	}
	template <typename U>
	shared_ptr(const shared_ptr<U>& other):
		count_(other.count_), ptr_(other.ptr_), deleter_(other.deleter_) {
		inc();
	}
	~shared_ptr() {
		dec();
	}
	void reset() {
		shared_ptr<T>().swap(*this);
	}
	void swap(shared_ptr<T> & other)
	{
		std::swap(count_, other.count_);
		std::swap(ptr_, other.ptr_);
		std::swap(deleter_, other.deleter_);
	}
	shared_ptr& operator=(const shared_ptr& other) {
		if (this != &other) {
			dec();
			count_ = other.count_;
			ptr_ = other.ptr_;
			deleter = other.deleter_;
			inc();
		}
		return *this;
	}
	void inc() {
		if (count_)
			count_->inc();
	}
	void dec() {
		if (count_) {
			count_->dec();
			if (count_->is_null()) {
				destory();
				delete count_;
				count_ = NULL;
			}
		}
	}
};
}
#endif // !MLITE_UTILS_SHARED_PTR_H_
