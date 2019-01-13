/**
* Copyright (C) DeepGlint, Inc - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
*
* Parsing json and params files of a MxNet model
*
* Written by Devymex <yumengwang@deepglint.com>, Jan. 2019
*/

#ifndef _MXNET_PARSER_HPP
#define _MXNET_PARSER_HPP

#include <string>
#include <vector>
#include <map>
#include "attributes.hpp"

using MxnetInput = std::pair<size_t, size_t>;

struct MxnetNode {
	std::string strName;
	std::string strOp;
	std::vector<MxnetInput> inputs;
	Attributes attrs;
};

struct MxnetParam {
	std::string strName;
	std::vector<float> data;
};

std::pair<std::vector<MxnetNode>, std::vector<size_t>> ParseMxnetJson(
		const std::string &strFile);

std::vector<MxnetParam> LoadMxnetParam(std::string strModelFn);

#endif
