#ifndef EVAL_EXECUTION_GRAPH_H_
#define EVAL_EXECUTION_GRAPH_H_
#include <vector>

#include "instruction.h"
#include "variable.h"
#include "node.h"
typedef size_t index_t;

namespace eval {
class ExecutionGraph {
public:
	static ExecutionGraph graph;
	static ExecutionGraph* Get() {
		return &graph;
	}
	void AddInstr(Instruction* instr) {
		instrs_.push_back(instr);
	}
	void AddVar(Variable* var) {
		vars_.push_back(var);
	}
private:
	std::vector<Instruction*> instrs_;
	std::vector<Variable*> vars_;
	std::vector<Node*> nodes_;
	std::vector<std::vector<index_t>> node2children_;
};
}
#endif