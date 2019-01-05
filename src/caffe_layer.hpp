#ifndef _CAFFE_LAYER_HPP
#define _CAFFE_LAYER_HPP

#include <vector>
#include "attributes.hpp"
#include "common.hpp"

using InputInfo = std::pair<std::string, Shape>;

struct CaffeLayer {
	std::string strName;
	std::string strType;
	std::string strParamName;
	std::vector<std::string> inputs;
	std::vector<std::string> outputs;
	Attributes params;
	std::vector<std::vector<float>> blobs;
	Shape inputShape;
};

std::ostream& operator << (std::ostream &os, const CaffeLayer &caffeLayer);

#endif // _CAFFE_LAYER_HPP
