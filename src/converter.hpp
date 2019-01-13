/**
* Copyright (C) DeepGlint, Inc - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
*
* Conversion from structurized MxNet nodes to a caffe::NetParameter
*
* Written by Devymex <yumengwang@deepglint.com>, Jan. 2019
*/

#ifndef CONVERTER_HPP_
#define CONVERTER_HPP_

#include <map>
#include <string>
#include <vector>

#define CPU_ONLY
#include <caffe/caffe.hpp>

#include "mxnet_parser.hpp"

using InputInfo = std::pair<std::string, Shape>;

//blobMapping: mapping layername to input parameter names in mxnet node
//eg. conv1 -> {conv1_weight, conv1_bias}
caffe::NetParameter MxnetNodes2CaffeNet(
		const std::vector<MxnetNode> &mxnetNodes,
		const std::vector<size_t> &headIndices,
		const std::vector<InputInfo> &inputInfos,
		std::map<std::string, std::vector<std::string>> &blobMapping);

int GuessBlobIDFromInputName(std::string strInputName);

bool IsEndWith(const std::string &strString, const std::string &strSuffix);

#endif /* CONVERTER_HPP_ */
