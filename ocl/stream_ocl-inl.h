#ifndef MLITE_OCL_STREAM_H_
#define MLITE_OCL_STREAM_H_

#include <memory>

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
		cl_command_queue command_queue = 
			clCreateCommandQueue(context, cur_dev_id, 0, &err);
		queue_.reset(command_queue);
		Executor::Get()->RegisterCmdQueue(queue_);
	}
	~Stream(void) {
		clFinish(*queue_);
	}
	// compute command queue
	std::shared_ptr<cl_command_queue> queue_;
};
}
#endif // !MLITE_OCL_STREAM_H_
