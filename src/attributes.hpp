/**
* Copyright (C) DeepGlint, Inc - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
*
* Statement of Attributes class.
*	Attributes is designed for MxNet Nodes and Caffe layers to describe
*	their Hyperparameters.
*
* Written by Devymex <yumengwang@deepglint.com>, Jan. 2019
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
