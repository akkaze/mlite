#ifndef MLITE_OCL_STREAM_H_
#define MLITE_OCL_STREAM_H_
#include "../base.h"
#include "../tensor.h"
#include "../logging.h"
#include "tensor_ocl-inl.h"
#include "executor.h"
namespace mlite {
template<>
struct Stream<ocl> {
	/*! \brief handle state */
	enum HandleState {
		NoHandle = 0,
		OwnHandle = 1,
	};
	Stream(void) {
		cl_int err;
		cl_device_id cur_dev_id = Executor::GetDeviceId();
		cl_context context = Executor::GetContext();
		// Create a compute context
		commands_queue_ = clCreateCommandQueue(context, cur_dev_id, 0, &err);
	}
	~Stream(void) {
		clFinish(commands_queue_);
	}
	// compute command queue
	cl_command_queue commands_queue_;
};
}
#endif // !MLITE_OCL_STREAM_H_
