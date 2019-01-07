/*
 * converter.cpp
 *
 *  Created on: Jan 5, 2019
 *      Author: devymex
 */

#include "converter.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include "logging.hpp"
#include "istream_helper.hpp"

bool IsEndWith(const std::string &strString, const std::string &strSuffix) {
	if (strString.length() >= strSuffix.length()) {
		return (0 == strString.compare(strString.length() - strSuffix.length(),
				strSuffix.length(), strSuffix));
	}
	return false;
}

std::string Attr_Copy(std::string strSrc) {
	return strSrc;
}

std::string Attr_Mode2Axis(std::string strMode) {
	CHECK(strMode == "instance" || strMode == "channel") <<
			"Unsupported mode: " << strMode;
	return strMode == "instance" ? "1" : "0";
}

std::string Attr_Tuple2Num(std::string strTuple) {
	std::istringstream iss(strTuple);
	size_t nVal1, nVal2;
	iss >> Expect('(') >> nVal1 >> Expect(',') >> nVal2 >> Expect(')');
	CHECK_EQ(nVal1, nVal2);
	return std::to_string(nVal1);
}

std::string Attr_Bool(std::string strBool) {
	std::transform(strBool.begin(), strBool.end(), strBool.begin(), ::tolower);
	bool bValue;
	if (strBool == "0" || strBool == "false") {
		bValue = false;
	} else if (strBool == "1" || strBool == "true") {
		bValue = true;
	} else {
		LOG(INFO) << "Bad boolean value: " << strBool;
	}
	return (bValue) ? "true" : "false";
}

std::string Attr_BoolInv(std::string strBool) {
	std::transform(strBool.begin(), strBool.end(), strBool.begin(), ::tolower);
	bool bValue;
	if (strBool == "0" || strBool == "false") {
		bValue = true;
	} else if (strBool == "1" || strBool == "true") {
		bValue = false;
	} else {
		LOG(INFO) << "Bad boolean value: " << strBool;
	}
	return (bValue) ? "true" : "false";
}

std::string Attr_PoolType(std::string strPool) {
	CHECK(strPool == "max" || strPool == "avg") <<
			"Unsupported pooling method: " << strPool;
	return (strPool == "max") ? "MAX" : "AVE";
}

std::vector<size_t> SortIndicesByDependencies(
		const std::vector<MxnetNode> &nodeAry,
		const std::vector<size_t> &headIndices) {
	std::map<size_t, size_t> nodesLevel;
	for (size_t i = 0; i < nodeAry.size(); ++i) {
		nodesLevel[i] = 0;
	}
	std::function<void(size_t, size_t)> VisitTree;
	VisitTree = [&](size_t iNode, size_t nLevel) {
			CHECK_LT(iNode, nodesLevel.size());
			nodesLevel[iNode] = std::max(nodesLevel[iNode], nLevel);
			for (auto iInput : nodeAry[iNode].inputs) {
				VisitTree(iInput.first, nLevel + 1);
			}
		};
	for (auto iHead : headIndices) {
		VisitTree(iHead, 0);
	}
	std::vector<size_t> results(nodeAry.size());
	std::iota(results.begin(), results.end(), 0);
	std::stable_sort(results.begin(), results.end(),
			[&](size_t i1, size_t i2){
				return nodesLevel[i1] > nodesLevel[i2];
			}
		);
	return std::move(results);
}

struct LayerFlags {
	bool bInPlace;
	size_t nOutNum;
};

std::pair<CaffeLayer, LayerFlags> MxnetNode2CaffeLayer(
		const MxnetNode &mxnetNode) {
	CaffeLayer caffeLayer;
	LayerFlags layerFlags;

	caffeLayer.strName = mxnetNode.strName;

	auto AttrMxnet2Caffe = [&](const std::string &strMxnetKey,
			const std::string &strCaffeKey,
			std::function<std::string(std::string)> converter,
			bool bRequired) {
		std::string strVal = mxnetNode.attrs.GetValue(strMxnetKey, bRequired);
		if (!strVal.empty()) {
			caffeLayer.params.emplace_back(strCaffeKey, converter(strVal));
		}
	};

	layerFlags.bInPlace = false;
	layerFlags.nOutNum = 1;
	if (mxnetNode.strOp == "null") {
		caffeLayer.strType = "Input";
	} else if (mxnetNode.strOp == "Flatten") {
		caffeLayer.strType = "Flatten";
		layerFlags.bInPlace = true;
	} else if (mxnetNode.strOp == "Activation") {
		std::string strActType = mxnetNode.attrs.GetValue("act_type", true);
		if (strActType == "relu") {
			caffeLayer.strType = "ReLU";
		} else {
			LOG(FATAL) << "Activation type " <<
					strActType << " not supported";
		}
		layerFlags.bInPlace = true;
	} else if (mxnetNode.strOp == "SoftmaxActivation") {
		caffeLayer.strType = "Softmax";
		caffeLayer.strParamName = "softmax_param";
		AttrMxnet2Caffe("mode", "axis", Attr_Mode2Axis, false);
	} else if (mxnetNode.strOp == "SliceChannel") {
		caffeLayer.strType = "Slice";
		caffeLayer.strParamName = "slice_param";
		std::string strNumOut = mxnetNode.attrs.GetValue("num_outputs", true);
		layerFlags.nOutNum = std::atoi(strNumOut.c_str());
		CHECK_GT(layerFlags.nOutNum, 1);
		AttrMxnet2Caffe("axis", "axis", Attr_Copy, false);
	} else if (mxnetNode.strOp == "FullyConnected") {
		caffeLayer.strType = "InnerProduct";
		caffeLayer.strParamName = "inner_product_param";
		AttrMxnet2Caffe("num_hidden", "num_output", Attr_Copy, true);
		AttrMxnet2Caffe("no_bias", "bias_term", Attr_BoolInv, true);
	} else if (mxnetNode.strOp == "Convolution") {
		caffeLayer.strType = "Convolution";
		caffeLayer.strParamName = "convolution_param";
		AttrMxnet2Caffe("num_filter", "num_output", Attr_Copy, true);
		AttrMxnet2Caffe("no_bias", "bias_term", Attr_BoolInv, true);
		AttrMxnet2Caffe("kernel", "kernel_size", Attr_Tuple2Num, true);
		AttrMxnet2Caffe("dilate", "dilation", Attr_Tuple2Num, false);
		AttrMxnet2Caffe("pad", "pad", Attr_Tuple2Num, false);
		AttrMxnet2Caffe("stride", "stride", Attr_Tuple2Num, false);
		AttrMxnet2Caffe("num_group", "group", Attr_Copy, false);
	} else if (mxnetNode.strOp == "Pooling") {
		caffeLayer.strType = "Pooling";
		caffeLayer.strParamName = "pooling_param";
		AttrMxnet2Caffe("kernel", "kernel_size", Attr_Tuple2Num, true);
		AttrMxnet2Caffe("stride", "stride", Attr_Tuple2Num, false);
		AttrMxnet2Caffe("pad", "pad", Attr_Tuple2Num, false);
		AttrMxnet2Caffe("global_pool", "global_pooling",
				Attr_Bool, false);
		AttrMxnet2Caffe("pool_type", "pool", Attr_PoolType, false);
		if (caffeLayer.params.GetValue("global_pooling", false) == "true") {
			caffeLayer.params.RemoveValue("kernel_size");
			caffeLayer.params.RemoveValue("stride");
			caffeLayer.params.RemoveValue("pad");
		} else {
			caffeLayer.params.RemoveValue("global_pooling");
		}
		std::string strActType = mxnetNode.attrs.GetValue(
				"pooling_convention", false);
		if (strActType != "full") {
			LOG(WARNING) << "pooling attribute \"pooling_convention\""
					" != \"full\", converted model may not work properly!";
		}
	} else if (mxnetNode.strOp == "elemwise_add") {
		caffeLayer.strType = "Eltwise";
		caffeLayer.strParamName = "eltwise_param";
		caffeLayer.params.emplace_back("operation", "SUM");
	} else if (mxnetNode.strOp == "BatchNorm") {
		caffeLayer.strType = "BatchNorm";
		caffeLayer.strParamName = "batch_norm_param";
		AttrMxnet2Caffe("use_global_stats", "use_global_stats",
				Attr_Bool, false);
	} else {
		LOG(FATAL) << "Unsupported op: " << mxnetNode.strOp;
	}
	return std::make_pair(std::move(caffeLayer), layerFlags);
}

int GuessBlobIDFromInputName(std::string strInputName) {
	using namespace std::placeholders;
	using SuffixBlobID = std::pair<std::string, size_t>;
	using SBAry = std::vector<SuffixBlobID>;
	static SBAry sbAry = {
			{"weight", 0}, {"gamma", 0}, {"moving_mean", 0},
			{"bias", 1}, {"beta", 1}, {"moving_var", 1}
		};
	auto iSuffix = std::find_if(sbAry.begin(), sbAry.end(),
			[&](const SuffixBlobID &sb) {
				return IsEndWith(strInputName, sb.first);
			}
		);
	if (iSuffix == sbAry.end()) {
		return -1;
	}
	return iSuffix->second;
}

void ExpandOrMergeLayers(std::vector<CaffeLayer> &layers) {
	for (auto iLayer = layers.begin(); iLayer != layers.end(); ) {
		if (iLayer->strType == "BatchNorm") {
			CHECK_EQ(iLayer->inputs.size(), 5);
			CHECK_EQ(iLayer->outputs.size(), 1);
			std::string strLayerName = iLayer->strName;
			std::string strOutputName = iLayer->outputs[0];
			std::string strInputGamma = iLayer->inputs[1];
			std::string strInputBeta = iLayer->inputs[2];
			iLayer->inputs.erase(iLayer->inputs.begin() + 1,
					iLayer->inputs.begin() + 3);

			iLayer = layers.insert(++iLayer, CaffeLayer());
			iLayer->strName = strLayerName + "_scale";
			iLayer->strType = "Scale";
			iLayer->strParamName = "scale_param";
			iLayer->inputs.push_back(strOutputName);
			iLayer->inputs.push_back(strInputGamma);
			iLayer->inputs.push_back(strInputBeta);
			iLayer->outputs.push_back(strOutputName);
			iLayer->params.emplace_back("bias_term", "true");

			++iLayer;
		} else if (iLayer->strType == "Flatten" ||
				GuessBlobIDFromInputName(iLayer->strName) >= 0) {
			iLayer = layers.erase(iLayer);
		} else {
			++iLayer;
		}
	}
}

void AssignBlobs(std::vector<CaffeLayer> &layers,
		const std::vector<MxnetParam> &mxnetParams) {
	for (auto &layer : layers) {
		auto &inputs = layer.inputs;
		for (auto iInput = inputs.begin(); iInput != inputs.end(); ) {
			int nBlobId = GuessBlobIDFromInputName(*iInput);
			if (nBlobId >= 0) {
				auto iParam = std::find_if(mxnetParams.begin(),
						mxnetParams.end(), [&](const MxnetParam &param) {
							return IsEndWith(param.strName, *iInput);
						}
					);
				if (iParam != mxnetParams.end()) {
					size_t nNewSize = size_t(nBlobId + 1);
					nNewSize = std::max(nNewSize, layer.blobs.size());
					layer.blobs.resize(nNewSize);
					layer.blobs[nBlobId] = iParam->data;
				}
				iInput = inputs.erase(iInput);
			} else {
				++iInput;
			}
		}
		if (layer.strType == "BatchNorm") {
			layer.blobs.resize(3);
			layer.blobs[2] = {1.0f};
		}
	}
}

std::vector<CaffeLayer> MxnetNodes2CaffeLayers(
		const std::vector<MxnetNode> &mxnetNodes,
		const std::vector<size_t> &headIndices,
		const std::vector<MxnetParam> &mxnetParams,
		const std::vector<InputInfo> &inputInfos) {
	auto sortedIndices = SortIndicesByDependencies(mxnetNodes, headIndices);
	std::vector<CaffeLayer> caffeLayers(mxnetNodes.size());

	std::map<std::string, size_t> typeCnt; // for unamed layers
	for (size_t i = 0; i < sortedIndices.size(); ++i) {
		auto &mxnetNode = mxnetNodes[sortedIndices[i]];
		auto convertResult = MxnetNode2CaffeLayer(mxnetNode);
		auto &caffeLayer = convertResult.first;
		auto &layerFlags = convertResult.second;

		// to give unamed layer a name
		if (caffeLayer.strName.empty()) {
			size_t nTypeCnt = ++typeCnt[caffeLayer.strType];
			caffeLayer.strName = "_unamed_" + caffeLayer.strType +
					std::to_string(nTypeCnt);
		}

		// convert inputs
		for (auto mxnetIdx : mxnetNode.inputs) {
			auto iCaffeIdx = std::find(sortedIndices.begin(),
					sortedIndices.end(), mxnetIdx.first);
			CHECK(iCaffeIdx != sortedIndices.end());
			CHECK_LT(*iCaffeIdx, i);
			auto &prevOutputs = caffeLayers[*iCaffeIdx].outputs;
			CHECK_LT(mxnetIdx.second, prevOutputs.size());
			caffeLayer.inputs.push_back(prevOutputs[mxnetIdx.second]);
		}

		// Convert outputs
		if (layerFlags.bInPlace) {
			CHECK_EQ(layerFlags.nOutNum, 1);
			caffeLayer.outputs.push_back(caffeLayer.inputs.front());
		} else {
			caffeLayer.outputs.resize(layerFlags.nOutNum, caffeLayer.strName);
			if (layerFlags.nOutNum > 1) {
				for (size_t i = 0; i < layerFlags.nOutNum; ++i) {
					caffeLayer.outputs[i] += std::to_string(i);
				}
			}
		}

		// Put this layer to vector
		caffeLayers[i] = std::move(caffeLayer);
	}

	ExpandOrMergeLayers(caffeLayers);

	AssignBlobs(caffeLayers, mxnetParams);
	
	for (auto &layer : caffeLayers) {
		auto iInputInfo = std::find_if(inputInfos.begin(), inputInfos.end(),
				[&](const InputInfo &ii) {
					return (layer.strName == ii.first);
				}
			);
		if (iInputInfo != inputInfos.end()) {
			layer.inputShape = iInputInfo->second;
		}
	}

	return caffeLayers;
}
