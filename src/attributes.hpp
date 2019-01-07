/*
 * attributes.hpp
 *
 *  Created on: Jan 5, 2019
 *      Author: devymex
 */

#ifndef ATTRIBUTES_HPP_
#define ATTRIBUTES_HPP_

#include "common.hpp"
class Attributes : public std::vector<StringPair> {
public:
	Attributes() = default;
	Attributes(const std::vector<StringPair> &baseObj);
	std::string GetValue(const std::string &strKey, bool bRequired) const;
	bool HasValue(const std::string &strKey) const;
	bool RemoveValue(const std::string &strKey);
private:
	std::vector<StringPair>::iterator _Find(const std::string &strKey);
	std::vector<StringPair>::const_iterator _Find(
			const std::string &strKey) const;
};


#endif /* ATTRIBUTES_HPP_ */
