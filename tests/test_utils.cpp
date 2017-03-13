#include "../base.h"
#include "../utils/shared_ptr.h"
#include "../utils/timer.h"
#include "../utils/format.h"

using namespace mlite;
void TestSharedPtr() {
	shared_ptr<int> p(new int(2));
}
void TestTimer() {
	Timer t;
	t.Start();
	std::cout << t.Elapse() << std::endl;
}
void TestStringReplace() {
	std::string src = "$DType hello";
	std::string from = "$DType";
	std::string to = "float";
	std::string ret = StringReplace(
		src, std::unordered_map<std::string,std::string>{ {from,to} });
	std::cout << ret << '\n';
}
int main(int argc, char **argv) {
	TestStringReplace();
//	TestSharedPtr();
	getchar();
	return 0;
}