#include "logging.h"
#include "base.h"
#include "tensor.h"
#include "tensor_cpu-inl.h"
#include "operation_cpu.h"
using namespace mlite;
template <typename DType>
class AddOne {
public:
	DType operator()(DType a) {
		return a + 1;
	}
};
void Print2DTensor(Tensor<cpu> m) {
	CHECK_EQ(m.dims(), 2);
	std::cout << std::setprecision(8);
	for (index_t i = 0; i < m.dim_size(0); i++) {
		for (index_t j = 0; j < m.dim_size(1); j++)
			std::cout << m[Indices({i, j})]  << '\t';
		std::cout << '\n';
	}
}
void TestTranpose() {
	Tensor<cpu> ts2(Shape2(5, 3));
	ts2 = NewTensor<cpu>(Shape2(5, 3), 0.f);
	UniformRandom<cpu> random(&ts2,0.f,1.f);
	random.Execute(&ts2);
	Print2DTensor(ts2);
	ts2.Transpose({1,0});
	//Map<AddOne>(ts2);
	std::cout << '\n';
	Print2DTensor(ts2);
	//LOG(INFO) << ts2.strides_;
	//ts2[{1, 2}] = 3;
	//std::cout << ts2[{1, 2}] << std::endl;
}

void TestSlice() {
	Tensor<cpu> ts2(Shape2(5, 3));
	ts2 = NewTensor<cpu>(Shape2(5, 3), 0.f);
	Print2DTensor(ts2);
	FreeSpace(&ts2);
}
void TestIndices() {
	Tensor<cpu> ts2(Shape2(5, 3));
	index_t id = 3;
	Indices indices = {3,2};
	LOG(INFO) << indices;
	indices = { 2 , 1 };
	LOG(INFO) << indices;
	id = ts2.IndexLogicalToPhysical(indices);
	LOG(INFO) << id;
}

int main(size_t argc, char** argv) {
	TestTranpose();
	//TestIndices();
	system("pause");
	return 0;
}