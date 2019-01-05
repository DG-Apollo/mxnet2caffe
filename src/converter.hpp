/*
 * convertor.hpp
 *
 *  Created on: Jan 5, 2019
 *      Author: devymex
 */

#ifndef CONVERTER_HPP_
#define CONVERTER_HPP_

#include <vector>

#include "mxnet_parser.hpp"
#include "caffe_layer.hpp"

std::vector<CaffeLayer> MxnetNodes2CaffeLayers(
		const std::vector<MxnetNode> &mxnetNodes,
		const std::vector<size_t> &headIndices,
		const std::vector<MxnetParam> &mxnetParams,
		const std::vector<InputInfo> &inputInfos);


#endif /* CONVERTER_HPP_ */
