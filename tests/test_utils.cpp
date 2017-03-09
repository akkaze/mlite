#include "base.h"
#include "utils\shared_ptr.h"
#include "utils\timer.h"
using namespace mlite;
void TestSharedPtr() {
	shared_ptr<int> p(new int(2));
}
void TestTimer() {
	Timer t;
	t.Start();
	std::cout << t.Elapse() << std::endl;
}
//int main(int argc, char **argv) {
//	TestSharedPtr();
//	system("pause");
//	return 0;
//}