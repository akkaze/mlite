#ifndef MLITE_OCL_EXECUTOR_H_
#define MLITE_OCL_EXECUTOR_H_

#include <unordered_map>
#include <memory>
#include <typeinfo>
#include <typeindex>
#include "../base.h"
#include "../logging.h"
#include "../utils/format.h"

//#include "stream_ocl-inl.h"
namespace mlite {
class Executor {
public:
	Executor() {
		// Create a command queue
		GetGPUPlatformId();
		GetAllDeviceIds();
		// set default warp size
		warp_size_ = 16;
		// create type name dictionary
		type_names = std::unordered_map<std::type_index, std::string>{
				{ std::type_index(typeid(short)),"short" },
				{ std::type_index(typeid(int)),"int" },
				{ std::type_index(typeid(long)),"long" },
				{ std::type_index(typeid(unsigned int)),"unsigned int" },
				{ std::type_index(typeid(float)),"float" },
				{ std::type_index(typeid(double)),"double" }
		};
	}
	~Executor() {
		for (Iterator it = dev2context_.begin(); it != dev2context_.end(); it++) {
			clReleaseContext(it->second);
		}
	}
	static int GetDeviceId() {
		return cur_dev_id_;
	}
	void SetDevice(const int& dev_id) {
		cur_dev_id_ = dev_id;
	}
	void ReleaseDevice(const int& dev_id) {
		clReleaseDevice(dev_ids_[dev_id]);
	}
	void QueryWarpSize(void) {
		std::string query_kernel = "";
	}
	static const cl_context GetContext() {
		// fisrt check if context already exists
		if (executor.dev2context_.find(cur_dev_id_) == executor.dev2context_.end()) {
			// Create a compute context
			cl_int err;
			cl_context context =
				clCreateContext(NULL, 1, &executor.dev_ids_[cur_dev_id_], NULL, NULL, &err);
			executor.dev2context_.insert(std::make_pair(cur_dev_id_, context));
			CHECK_EQ(err, CL_SUCCESS) << "Error during creating context";
		}
		Iterator it = executor.dev2context_.find(cur_dev_id_);
		return it->second;
	}
	cl_kernel Complie(const std::string& kernel_src, const std::string& kernel_name) {
		cl_int err;
		// Create the compute program from the source buffer
		const char* kernel_src_cstr = kernel_src.c_str();
		cl_context context = dev2context_[cur_dev_id_];
		cl_program program = clCreateProgramWithSource(
			context, 1, (const char **)&kernel_src_cstr, NULL, &err);
		CHECK_NE(err, 0) << "Error during creating program!";
		// Build the program
		err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		CHECK_NE(err, 0) << "Error during building program!";
		// Create the compute kernel from the program
		cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
		CHECK_NE(err, 0) << "Error during creating kernel!";
		return kernel;
	}
	//@brief run the kernel function
	void Run(const cl_kernel& kernel,
		const std::vector<size_t>& data_size,
		const std::vector<size_t>& local_size) {
		cl_int err;
		CHECK_EQ(data_size.size(),local_size.size()) 
			<< "data must have same dimension with block";
		std::vector<size_t> global_size(data_size.size());
		for (index_t i = 0; i < data_size.size(); i++)
			global_size[i] = (data_size[i] + local_size[i] - 1) / local_size[i];

	}
	//@brief register command queue from stream
	void RegisterCmdQueue(const std::shared_ptr<cl_command_queue>& queue) {
		dev2queue_.insert(std::make_pair(cur_dev_id_, queue));
	}
	const std::shared_ptr<cl_command_queue>& GetCmdQueue() {
		std::unordered_map<int, std::shared_ptr<cl_command_queue>>::const_iterator
			cit = dev2queue_.find(cur_dev_id_);
		CHECK_NE(cit, dev2queue_.end());
		return cit->second;
	}
	static Executor* Get() {
		static Executor executor;
		return &executor;
	}
	std::string LoadPragram(const std::string& src_file) {
		std::ifstream in;
		in.open(src_file.c_str(), std::ios::in);
		//CHECK_NOTNULL(in) << "Cannot read this file,check if this file name correct!";
		return std::string(
			std::istreambuf_iterator<char>(in),
			(std::istreambuf_iterator<char>()));
	}
	std::string Inlcude(const std::string& src,
		const std::string& include) {
		std::string included;
		included.append(include);
		included.append("\n");
		included.append(src);
		return included;
	}

	template <typename DType>
	void ReplaceTemplateParams(std::string& src) {
		std::string dtype_name = type_names[std::type_index(typeid(DType))];
		src = StringReplace(src, { {"DType",dtype_name} });
	}
	void LoadParams(std::string& src,
		const std::unordered_map<std::string, std::string>& param_from_to) {
		src = StringReplace(src, param_from_to);
	}
protected:
	//@brief helper function for getting all device ids for a gpu platform
	void GetAllDeviceIds(void) {
		cl_int err;
		cl_uint num_dev_ids = 0;
		err = clGetDeviceIDs(gpu_platform_id_, CL_DEVICE_TYPE_GPU, 0, NULL,
			&num_dev_ids);
		CHECK_EQ(err, CL_SUCCESS) << "Failing getting device count!";
		//get all avaliable devices
		err = clGetDeviceIDs(gpu_platform_id_,
			CL_DEVICE_TYPE_GPU, num_dev_ids, dev_ids_.data(), NULL);
		CHECK_EQ(err, CL_SUCCESS) << "Failing getting device ids!";
	}
	//@brief helper function for getting available gpu platform id
	void GetGPUPlatformId(void) {
		cl_int err;
		// Set up platform and GPU device
		cl_uint num_platforms;
		// Find number of platforms
		err = clGetPlatformIDs(0, NULL, &num_platforms);
		CHECK_EQ(err, CL_SUCCESS)
			<< "Error during Finding platforms!";
		CHECK_NE(num_platforms, 0)
			<< "Found 0 platforms!";
		// Get all platforms
		cl_platform_id *platform = new cl_platform_id[num_platforms];
		err = clGetPlatformIDs(num_platforms, platform, NULL);
		CHECK_EQ(err, CL_SUCCESS)
			<< "Error during getting platforms!";
		// Secure a GPU
		cl_device_id dev_id;
		for (index_t i = 0; i < num_platforms; i++) {
			err = clGetDeviceIDs(
				platform[i], CL_DEVICE_TYPE_GPU, 1, &dev_id, NULL);
			if (err == CL_SUCCESS) {
				gpu_platform_id_ = platform[i];
				break;
			}
		}
		CHECK_NE(cur_dev_id_, 0) << "Found 0 devices!";

		delete platform;
	}
private:
	//typenames
	//Stream< ocl>* stream;
	std::unordered_map<std::type_index, std::string> type_names;
	index_t warp_size_;
	// gpu platforom id
	cl_platform_id gpu_platform_id_;
	// all gpu device ids
	std::vector<cl_device_id> dev_ids_;
	// map from device id to context
	std::unordered_map<int,cl_context> dev2context_;
	typedef std::unordered_map<int, cl_context>::iterator Iterator;
	std::unordered_map<int,std::shared_ptr<cl_command_queue>> dev2queue_;

	// compute device id
	static thread_local int	cur_dev_id_;
	//singleton
	static Executor executor;
};
Executor Executor::executor = Executor();
thread_local int Executor::cur_dev_id_ = 0;
//load typenames

}
#endif