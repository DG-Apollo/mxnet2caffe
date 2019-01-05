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
	std::string GetValue(std::string strMxnetKey, bool bRequired) const;
};


#endif /* ATTRIBUTES_HPP_ */
