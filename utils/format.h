#ifndef MLITE_FORMAT_H_
#define MLITE_FORMAT_H_
#include "../base.h"
#include "../logging.h"
#include <unordered_map>
#include <regex>
namespace mlite {
void StringReplace(std::string& str,const 
	std::unordered_map<std::string,std::string>& from_to) {
	for (std::unordered_map<std::string, std::string>::const_iterator 
		it = from_to.begin();
		it != from_to.end(); it++) {
		CHECK(it->first[0] == '$');
		std::string pattern_str = "(?:\s*)\\" + it->first + "\\b";
		try {
			std::regex pattern(pattern_str);
			std::regex_replace(std::back_inserter(str), 
				str.begin(), str.end(),
				pattern, it->second);
		}
		catch (const std::regex_error& e) {
			LOG(ERROR) << e.what();
		}
	}
}
}
#endif