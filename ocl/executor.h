#ifndef MLITE_OCL_EXECUTOR_H_
#define MLITE_OCL_EXECUTOR_H_

#include <unordered_map>
#include <memory>
#include <typeinfo>
#include <typeindex>
#include "../base.h"
#include "../logging.h"
#include "../utils/format.h"


namespace mlite {
class Executor {
public:
	Executor() {
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
		for (index_t i = 0; i < num_platforms; i++) {
			err = clGetDeviceIDs(
				platform[i], CL_DEVICE_TYPE_DEFAULT, 1, &cur_dev_id_, NULL);
			if (err == CL_SUCCESS) 
				break;
		}
		CHECK_NE(cur_dev_id_,0) << "Found 0 devices!";
		// Create a command queue
		SetDevice(cur_dev_id_);
		delete[] platform;
	}
	~Executor() {
		for (Iterator it = dev2context_.begin(); it != dev2context_.end(); it++) {
			clReleaseContext(it->second);
		}
	}
	static cl_device_id GetDeviceId() {
		return cur_dev_id_;
	}
	void SetDevice(const cl_device_id& dev_id) {
		cur_dev_id_ = dev_id;
	}
	void ReleaseDevice(void) {
		clReleaseDevice(cur_dev_id_);
	}
	static const cl_context GetContext() {
		static Executor executor;
		// fisrt check if context already exists
		if (executor.dev2context_.find(cur_dev_id_) == executor.dev2context_.end()) {
			// Create a compute context
			cl_int err;
			cl_context context =
				clCreateContext(NULL, 1, &cur_dev_id_, NULL, NULL, &err);
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
	void RegisterCmdQueue(const std::shared_ptr<cl_command_queue>& queue) {
		dev2queue_.insert(std::make_pair(cur_dev_id_, queue));
	}
	const std::shared_ptr<cl_command_queue>& GetCmdQueue() {
		std::unordered_map<cl_device_id, std::shared_ptr<cl_command_queue>>::const_iterator
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
		StringReplace(src, { {"DType",dtype_name} });
	}
	void LoadParams(std::string& src,
		const std::unordered_map<std::string, std::string>& param_from_to) {
		StringReplace(src, param_from_to);
	}
private:
	//typenames
	static std::unordered_map<std::type_index, std::string> type_names;
	static index_t warp_size_;
	// map from device id to context
	std::unordered_map<cl_device_id, cl_context> dev2context_;
	std::unordered_map<cl_device_id, std::shared_ptr<cl_command_queue>> dev2queue_;
	typedef std::unordered_map<cl_device_id, cl_context>::iterator Iterator;
	// compute device id
	static thread_local cl_device_id	cur_dev_id_;
};
std::unordered_map<std::type_index, std::string> Executor::type_names = 
	std::unordered_map<std::type_index, std::string>{
		{ std::type_index(typeid(short)),"short" },
		{ std::type_index(typeid(int)),"int"},
		{ std::type_index(typeid(long)),"long" },
		{ std::type_index(typeid(unsigned int)),"unsigned int" },
		{ std::type_index(typeid(float)),"float" },
		{ std::type_index(typeid(double)),"double" }
	};
index_t Executor::warp_size_ = 16;
thread_local cl_device_id Executor::cur_dev_id_ = NULL;
//load typenames

}
#endif