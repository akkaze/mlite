#ifndef EVAL_VARIABLE_H_
#define EVAL_VARIABLE_H_
#include <string>
#include <vector>

#include "../tensor.h"

using namespace mlite;
namespace eval {

class Variable {
public:
	Variable() = default;
	Variable(const std::string& name) {
		name_ = name;
		ref_count_ = 0;
	}
private:
	std::string name_;
	size_t ref_count_;
	Tensor* data_;
	std::vector<Variable*> children_;
};
}
#endif // !EVAL_VARIABLE_H_
