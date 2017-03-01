#ifndef MLITE_OCL_STREAM_H_
#define MLITE_OCL_STREAM_H_
#include "../base.h"
#include "../tensor.h"
#include "../logging.h"

namespace mlite {
namespace ocl {
struct Stream<ocl> {
	/*! \brief handle state */
	enum HandleState {
		NoHandle = 0,
		OwnHandle = 1,
	};
	// compute command queue
	cl_command_queue commands_queue_;
};
}
}
#endif // !MLITE_OCL_STREAM_H_
