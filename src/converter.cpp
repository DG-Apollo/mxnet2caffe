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

std::string Attr_Num2Bool(std::string strBool) {
	CHECK(strBool == "0" || strBool == "1") << "Bad boolean value";
	return (strBool == "1") ? "true" : "false";
}

std::string Attr_Num2BoolInv(std::string strBool) {
	CHECK(strBool == "0" || strBool == "1") << "Bad boolean value";
	return (strBool == "0") ? "true" : "false";
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
		AttrMxnet2Caffe("no_bias", "bias_term", Attr_Num2BoolInv, true);
	} else if (mxnetNode.strOp == "Convolution") {
		caffeLayer.strType = "Convolution";
		caffeLayer.strParamName = "convolution_param";
		AttrMxnet2Caffe("num_filter", "num_output", Attr_Copy, true);
		AttrMxnet2Caffe("no_bias", "bias_term", Attr_Num2BoolInv, true);
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
				Attr_Num2Bool, false);
		AttrMxnet2Caffe("pool_type", "pool", Attr_PoolType, false);
		std::string strActType = mxnetNode.attrs.GetValue(
				"pooling_convention", false);
		if (strActType != "full") {
			LOG(WARNING) << "pooling attribute \"pooling_convention\""
					" != \"full\", converted model may not work properly!";
		}
	} else if (mxnetNode.strOp == "BatchNorm") {
		caffeLayer.strType = "BatchNorm";
		caffeLayer.strParamName = "batch_norm_param";
		AttrMxnet2Caffe("use_global_stats", "use_global_stats",
				Attr_Num2Bool, false);
	} else {
		LOG(FATAL) << "Unsupported op: " << mxnetNode.strOp;
	}
	return std::make_pair(std::move(caffeLayer), layerFlags);
}

int GuessBlobIDFromInputName(std::string strInputName) {
	static std::map<std::string, size_t> suffix2BlodID = {
			{"weight", 0}, {"gamma", 0}, {"mmean", 0},
			{"bias", 1}, {"beta", 1}, {"mvar", 1}};
	auto iSuffixIdx = strInputName.find_last_of('_');
	if (iSuffixIdx == std::string::npos) {
		return -1;
	}
	strInputName.erase(0, iSuffixIdx + 1);
	auto iBlobID = suffix2BlodID.find(strInputName);
	if (iBlobID == suffix2BlodID.end()) {
		return -1;
	}
	int nBlobID = (int)iBlobID->second;
	return nBlobID;
}

void ExpandOrMergeLayers(std::vector<CaffeLayer> &layers) {
	for (auto iLayer = layers.begin(); iLayer != layers.end(); ) {
		if (iLayer->strType == "BatchNorm") {
			auto &inputs = iLayer->inputs;
			CHECK_EQ(inputs.size(), 5);
			CHECK_EQ(iLayer->outputs.size(), 1);
			std::string strLayerName = iLayer->strName;
			std::string strOutputName = iLayer->outputs[0];
			std::string strInputGamma = inputs[0];
			std::string strInputBeta = inputs[1];
			inputs.erase(inputs.begin() + 1, inputs.begin() + 3);

			iLayer = layers.insert(++iLayer, CaffeLayer());
			iLayer->strName = strLayerName + "_scale";
			iLayer->strType = "Scale";
			iLayer->strParamName = "scale_params";
			iLayer->inputs.push_back(strOutputName);
			iLayer->inputs.push_back(strInputGamma);
			iLayer->inputs.push_back(strInputBeta);
			iLayer->outputs.push_back(strOutputName);
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
		const std::vector<MxnetParam> &mxnetParams,
		const std::vector<InputInfo> &inputInfos) {
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
		auto iInputInfo = std::find_if(inputInfos.begin(), inputInfos.end(),
				[&](const InputInfo &ii) {
					return (layer.strName == ii.first);
				}
			);
		if (iInputInfo != inputInfos.end()) {
			layer.inputShape = iInputInfo->second;
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

	AssignBlobs(caffeLayers, mxnetParams, inputInfos);

	return caffeLayers;
}
