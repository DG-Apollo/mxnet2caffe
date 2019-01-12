#include "common.hpp"

bool IsEndWith(const std::string &strString, const std::string &strSuffix) {
	if (strString.length() >= strSuffix.length()) {
		return (0 == strString.compare(strString.length() - strSuffix.length(),
				strSuffix.length(), strSuffix));
	}
	return false;
}

