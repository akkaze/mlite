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
		// get a compute context
		queue_ = Executor::GetCmdQueue();
	}
	~Stream(void) {
		clFinish(queue_);
	}
	// compute command queue
	cl_command_queue queue_;
};
}
#endif // !MLITE_OCL_STREAM_H_
