/*
 * convertor.hpp
 *
 *  Created on: Jan 5, 2019
 *      Author: devymex
 */

#ifndef CONVERTER_HPP_
#define CONVERTER_HPP_

#include <vector>

#define CPU_ONLY
#include <caffe/caffe.hpp>

#include "mxnet_parser.hpp"
#include "caffe_layer.hpp"

caffe::NetParameter MxnetNodes2CaffeNet(
		const std::vector<MxnetNode> &mxnetNodes,
		const std::vector<size_t> &headIndices,
		const std::vector<InputInfo> &inputInfos);
		
bool IsEndWith(const std::string &strString, const std::string &strSuffix);

#endif /* CONVERTER_HPP_ */
