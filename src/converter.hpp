/*
 * convertor.hpp
 *
 *  Created on: Jan 5, 2019
 *      Author: devymex
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

bool IsEndWith(const std::string &strString, const std::string &strSuffix);
int GuessBlobIDFromInputName(std::string strInputName);

#endif /* CONVERTER_HPP_ */
