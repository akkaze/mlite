#ifndef MLITE_OCL_TENSOR_INL_H_
#define MLITE_OCL_TENSOR_INL_H_
#include "../base.h"
#include "../tensor.h"
namespace mlite {
static index_t kOclDefualtDeviceId = 0;
template<>
MLITE_XINLINE void InitTensorEngine<ocl>(int dev_id) {
	
}
}

//@breif opencl only routines
namespace mlite {
namespace ocl{
inline void CheckLaunchParam() {}
}
}
#endif // ! MLITE_OCL_TENSOR_INL_H_
