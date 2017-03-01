#ifndef MLITE_OCL_EXECUTOR_H_
#define MLITE_OCL_EXECUTOR_H_

#include "../base.h"
#include "../logging.h"
#include "tensor_ocl_inl.h"
#if MLITE_IN_CXX11
#include <unordered_map>
#else
#include <map>
#endif
namespace mlite {
class Executor {
public:
	Executor() {
		cl_int err;
		// Set up platform and GPU device
		cl_uint num_platforms;
		// Find number of platforms
		CHECK_NE(clGetPlatformIDs(0, NULL, &num_platforms),CL_FALSE) 
			<< "Error during Finding platforms!";
		CHECK_NE(num_platforms, 0)
			<< "Found 0 platforms!";
		// Get all platforms
		cl_platform_id *platform = new cl_platform_id[num_platforms];
		CHECK_NE(clGetPlatformIDs(num_platforms, platform, NULL), CL_FALSE)
			<< "Error during getting platforms!";
		// Secure a GPU
		//for (index_t i = 0; i < num_platforms; i++) {
		//	err = clGetDeviceIDs(
		//		platform[i], CL_DEVICE_TYPE_DEFAULT, 1, &device_id_, NULL);
		//	if (err == CL_SUCCESS) 
		//		break;
		//}
		//CHECK_NE(device_id_,0) << "Found 0 devices!";
		// Create a compute context
		// Create a command queue
		commands_ = clCreateCommandQueue(context_, device_id_, 0, &err);
	}
	static const cl_context GetContext(const cl_device_id& dev_id) {
		static Executor executor;
		// fisrt check if context already exists
		if (executor.dev2context_.find(dev_id) == executor.dev2context_.end()) {
			// Create a compute context
			cl_int err;
			cl_context context =
				clCreateContext(NULL, 1, &dev_id, NULL, NULL, &err);
			CHECK_NE(err, 0) << "Error during creating context";
		}
		Iterator it = executor.dev2context_.find(dev_id);
		return it->second;
	}
	void Run(const std::string& kernel_src,const std::string& kernel_name) {
		cl_int err;
		// Create the compute program from the source buffer
		const char* kernel_src_cstr = kernel_src.c_str();
		program_ = clCreateProgramWithSource(
			context_, 1, (const char **)&kernel_src_cstr, NULL, &err);
		CHECK_NE(err, 0) << "Error during creating program!";
		// Build the program
		err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);
		CHECK_NE(err, 0) << "Error during building program!";
		// Create the compute kernel from the program
		clCreateKernel(program_, kernel_name.c_str(), &err);
		CHECK_NE(err, 0) << "Error during creating kernel!";

	}
	static Executor* Get() {
		static Executor executor;
		return &executor;
	}

private:
	// map from device id to context
#if MLITE_IN_CXX11
	std::unordered_map<cl_device_id, cl_context> dev2context_;
	typedef std::unordered_map<cl_device_id, cl_context>::iterator Iterator;
#endif
	// compute device id
	cl_device_id	device_id_; 
	// compute context
	cl_context       context_;  
	// compute program
	cl_program       program_;
	// compute kernel
	cl_kernel        kernel_;       
};
}
#endif