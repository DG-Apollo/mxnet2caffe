#include <algorithm>
#include <functional>
#include <ostream>
#include <numeric>
#include "logging.hpp"
#include "mxnet_parser.hpp"
#include "caffe_layer.hpp"

std::ostream& operator << (std::ostream &os, const CaffeLayer &layer) {
	os << "layer {\n  name: \"" << layer.strName << "\"\n" <<
			"  type: \"" << layer.strType << "\"\n";
	for (auto &strInputName : layer.inputs) {
		os << "  bottom: \"" << strInputName << "\"\n";
	}
	for (auto &strOutputName : layer.outputs) {
		os << "  top: \"" << strOutputName << "\"\n";
	}
	if (!layer.params.empty()) {
		os << "  " << layer.strParamName << " {\n";
		for (auto &attr : layer.params) {
			os << "    " << attr.first << ": " << attr.second << "\n";
		}
		os << "  }\n";
	}
	if (!layer.inputShape.empty()) {
		os << "  input_param {shape: {";
		for (auto s : layer.inputShape) {
			os << " dim: " << s;
		}
		os << "}}\n";
	}
	os << "}\n";
	return os;
}

