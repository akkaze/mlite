#ifndef EVAL_INSTRUCTION_H_
#define EVAL_INSTRUCTION_H_
#include <vector>
#include "variable.h"

#include "tensor.h"
#include "logging.h"

using namespace	mlite;
namespace eval {
class Instruction {
public:
	Instruction() = default;
	~Instruction() = default;
	Instruction(const std::vector<Variable*> operand, Variable* result) {
		operand_ = operand;
		result_ = result;
	}
	Instruction(const Instruction& other) = default;
	Instruction& operator=(const Instruction& other) = default;
	virtual void Fire(void) = 0;
private:
	std::vector<Variable*> operand_;
	Variable* result_;
};
class AllocInstr : public Instruction {
	AllocInstr() = default;
	~AllocInstr() = default;
	AllocInstr(const Shape& shape, Variable* result) {
		result_ = result;
		shape_ = shape;
	}
	void Fire(void) {
	}
private:
	Shape shape_;
	Variable* result_;
};

}
#endif
