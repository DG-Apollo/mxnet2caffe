/*
 * attributes.cpp
 *
 *  Created on: Jan 5, 2019
 *      Author: devymex
 */

#include "attributes.hpp"
#include <algorithm>
#include "logging.hpp"

Attributes::Attributes(const std::vector<StringPair> &baseObj) :
		std::vector<StringPair>(baseObj) {
}

std::string Attributes::GetValue(std::string strMxnetKey, bool bRequired) const {
	auto iAttr = std::find_if(begin(), end(),
			[&](const StringPair &attr) {
				return attr.first == strMxnetKey;
			}
		);
	if (iAttr == end()) {
		if (bRequired) {
			LOG(FATAL) << "Key " << strMxnetKey << " not found";
		}
		return std::string();
	}
	return iAttr->second;
}
