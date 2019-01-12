#ifndef _COMMON_HPP
#define _COMMON_HPP

#include <vector>
#include <string>
#include <utility>

using Shape = std::vector<size_t>;
using StringPair = std::pair<std::string, std::string>;

bool IsEndWith(const std::string &strString, const std::string &strSuffix);

#endif // _COMMON_HPP
