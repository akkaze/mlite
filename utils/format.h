#ifndef MLITE_FORMAT_H_
#define MLITE_FORMAT_H_
#include <unordered_map>
#include <regex>

#include "../logging.h"

namespace mlite {
std::string StringReplace(std::string& str,
	const std::unordered_map<std::string,std::string>& from_to) {
	std::string ret;
	for (std::unordered_map<std::string, std::string>::const_iterator 
		it = from_to.cbegin();
		it != from_to.cend(); it++) {
		CHECK(it->first[0] == '$');
		std::string pattern_str = "(?:\s*)\\" + it->first + "\\b";
		try {
			std::regex pattern(pattern_str);
			std::regex_replace(std::back_inserter(ret), 
				str.begin(), str.end(),
				pattern, it->second);
			str = ret;
		}
		catch (const std::regex_error& e) {
			LOG(ERROR) << e.what();
		}
	}
	return ret;
}
}
#endif