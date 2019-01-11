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

#define CPU_ONLY
#include <caffe/caffe.hpp>

#include "logging.hpp"
#include "istream_helper.hpp"

template<typename _Ty>
_Ty Str2Num(std::string str, _Ty _min = std::numeric_limits<_Ty>::min(),
		_Ty _max = std::numeric_limits<_Ty>::max()) {
	CHECK(!str.empty());
	std::istringstream iss(str);
	_Ty val;
	CHECK(iss >> val);
	CHECK_GE(val, _min);
	CHECK_LE(val, _max);
	return val;
}

template<typename _Ty>
std::pair<_Ty, _Ty> Str2Pair(std::string str,
		_Ty _min = std::numeric_limits<_Ty>::min(),
		_Ty _max = std::numeric_limits<_Ty>::max()) {
	CHECK(!str.empty());
	std::istringstream iss(str);
	std::pair<_Ty, _Ty> ret;
	CHECK(iss >> Expect('(') >> ret.first >> Expect(',') >>
			ret.second >> Expect(')'));
	CHECK_GE(ret.first, _min);
	CHECK_LE(ret.first, _max);
	CHECK_GE(ret.second, _min);
	CHECK_LE(ret.second, _max);
}

template<typename _Ty>
_Ty Pair2Num(const std::pair<_Ty, _Ty> &pair) {
	CHECK_EQ(pair.first, pair.second);
	return pair.first;
}

bool Str2Bool(std::string str) {
	if (str == "True" || str == "true" || str == "1") {
		return true;
	}
	CHECK(str == "False" || str == "false" || str == "0");
	return false;
}

struct ConvertInfo {
	bool bInPlace;
	int nOutNum;
};

ConvertInfo MxnetNode2CaffeLayer(MxnetNode mxnetNode,
		caffe::LayerParameter &caffeLayer) {
	caffeLayer.set_name(std::move(mxnetNode.strName));
	ConvertInfo cvtInfo = {false, 1};

	using AttrProc = std::function<void(std::string)>;
	using AttrProcMap = std::map<std::string, AttrProc>;
	AttrProcMap requiredAttrs;
	AttrProcMap optionalAttrs;
	if (mxnetNode.strOp == "null") {
		caffeLayer.set_type("Input");
	} else if (mxnetNode.strOp == "Flatten") {
		caffeLayer.set_type("Flatten");
		cvtInfo.bInPlace = true;
	} else if (mxnetNode.strOp == "Activation") {
		cvtInfo.bInPlace = true;
		requiredAttrs["act_type"] = [&](std::string strVal) {
			if (strVal == "relu") {
				caffeLayer.set_type("ReLU");
			} else {
				LOG(FATAL);
			}
		};
	} else if (mxnetNode.strOp == "SoftmaxActivation") {
		caffeLayer.set_type("Softmax");
		optionalAttrs["mode"] = [&](std::string strVal) {
			if (!strVal.empty()) {
				CHECK(strVal == "instance" || strVal == "channel");
				caffeLayer.mutable_softmax_param()->set_axis(
						(strVal == "instance") ? 1 : 0);
			}
		};
	} else if (mxnetNode.strOp == "SliceChannel") {
		caffeLayer.set_type("Slice");
		requiredAttrs["num_outputs"] = [&](std::string strVal) {
			cvtInfo.nOutNum = Str2Num<int>(strVal, 2);
			for (int i = 0; i < cvtInfo.nOutNum; ++i) {
				caffeLayer.add_top("");
			}
		};
		optionalAttrs["axis"] = [&](std::string strVal) {
			int nAxis = Str2Num<int>(strVal, 0, 4);
			caffeLayer.mutable_slice_param()->set_axis(nAxis);
		};
		optionalAttrs["squeeze_axis"] = [&](std::string strVal) {
			bool bSqueeze = Str2Bool(strVal);
			CHECK(!bSqueeze);
		};
	} else if (mxnetNode.strOp == "FullyConnected") {
		caffeLayer.set_type("Slice");
		requiredAttrs["num_hidden"] = [&](std::string strVal) {
			int nNumHid = Str2Num<int>(strVal, 1);
			caffeLayer.mutable_inner_product_param()->set_num_output(nNumHid);
		};
		optionalAttrs["no_bias"] = [&](std::string strVal) {
			bool bNoBias = Str2Bool(strVal);
			if (!bNoBias) {
				caffeLayer.mutable_inner_product_param()->set_bias_term(false);
			}
		};
		optionalAttrs["flatten"]; // ignored;
	} else if (mxnetNode.strOp == "Convolution") {
		caffeLayer.set_type("Convolution");
		auto &convParam = *caffeLayer.mutable_convolution_param();
		requiredAttrs["num_filter"] = [&](std::string strVal) {
			int nNumChs = Str2Num<int>(strVal, 1);
			convParam.set_num_output(nNumChs);
		};
		requiredAttrs["kernel"] = [&](std::string strVal) {
			auto kernel = Str2Pair<int>(strVal, 1);
			if (kernel.first == kernel.second) {
				convParam.add_kernel_size(kernel.first);
			} else {
				convParam.set_kernel_h(kernel.first);
				convParam.set_kernel_w(kernel.second);
			}
		};
		optionalAttrs["stride"] = [&](std::string strVal) {
			auto stride = Str2Pair<int>(strVal, 1);
			if (stride.first == stride.second) {
				convParam.add_stride(stride.first);
			} else {
				convParam.set_stride_h(stride.first);
				convParam.set_stride_w(stride.second);
			}
		};
		optionalAttrs["pad"] = [&](std::string strVal) {
			auto pad = Str2Pair<int>(strVal, 0);
			if (pad.first == pad.second) {
				convParam.add_pad(pad.first);
			} else {
				convParam.set_pad_h(pad.first);
				convParam.set_pad_w(pad.second);
			}
		};
		optionalAttrs["dilate"] = [&](std::string strVal) {
			int nDilate = Pair2Num(Str2Pair<int>(strVal, 0));
			if (nDilate != 1) {
				convParam.add_dilation(nDilate);
			}
		};
		optionalAttrs["num_group"] = [&](std::string strVal) {
			int nNumChs = convParam.num_output();
			int nNumGroup = Str2Num(strVal, 1, nNumChs);
			CHECK_EQ(nNumChs % nNumGroup, 0);
			if (nNumGroup != 1) {
				int nGroupSize = nNumChs / nNumGroup;
				convParam.set_group(nGroupSize);
			}
		};
		optionalAttrs["no_bias"] = [&](std::string strVal) {
			bool bNoBias = Str2Bool(strVal);
			if (!bNoBias) {
				convParam.set_bias_term(false);
			}
		};
		optionalAttrs["workspace"]; // ignored
		optionalAttrs["cudnn_tune"]; // ignored
		optionalAttrs["cudnn_off"]; // ignored
		//optionalAttrs["layout"]; // unsupported
	} else if (mxnetNode.strOp == "Pooling") {
		caffeLayer.set_type("Pooling");
		auto &poolParam = *caffeLayer.mutable_pooling_param();
		optionalAttrs["pool_type"] = [&](std::string strVal) {
			if (strVal == "avg") {
				poolParam.set_pool(caffe::PoolingParameter_PoolMethod_AVE);
			} else if(strVal == "max") {
				poolParam.set_pool(caffe::PoolingParameter_PoolMethod_MAX);
			}
			LOG(FATAL) << "Unsupported pooling method: " << strVal;
		};
		optionalAttrs["kernel"] = [&](std::string strVal) {
			auto kernel = Str2Pair<int>(strVal, 0);
			if (kernel.first == kernel.second) {
				poolParam.set_kernel_size(kernel.first);
			} else {
				poolParam.set_kernel_h(kernel.first);
				poolParam.set_kernel_w(kernel.second);
			}
		};
		optionalAttrs["stride"] = [&](std::string strVal) {
			auto stride = Str2Pair<int>(strVal, 1);
			if (stride.first == stride.second) {
				poolParam.set_stride(stride.first);
			} else {
				poolParam.set_stride_h(stride.first);
				poolParam.set_stride_w(stride.second);
			}
		};
		optionalAttrs["pad"] = [&](std::string strVal) {
			auto pad = Str2Pair<int>(strVal, 0);
			if (pad.first == pad.second) {
				poolParam.set_pad(pad.first);
			} else {
				poolParam.set_pad_h(pad.first);
				poolParam.set_pad_w(pad.second);
			}
		};
		optionalAttrs["global_pool"] = [&](std::string strVal) {
			bool bGlobalPool = Str2Bool(strVal);
			if (bGlobalPool) {
				poolParam.set_global_pooling(true);
			}
		};
		optionalAttrs["pooling_convention"] = [&](std::string strVal) {
			CHECK(strVal == "full");
		};
		optionalAttrs["p_value"] = [&](std::string strVal) {
			LOG(FATAL) << "Lp pooling is not supported";
		};
		optionalAttrs["count_include_pad"] = [&](std::string strVal) {
			LOG(FATAL) << "count_include_pad is not supported";
		};
		optionalAttrs["cudnn_off"]; // ignored
	} else if (mxnetNode.strOp == "elemwise_add" ||
			mxnetNode.strOp == "_Plus") {
		caffeLayer.set_type("Eltwise");
	} else if (mxnetNode.strOp == "elemwise_mul") {
		caffeLayer.set_type("Eltwise");
		caffeLayer.mutable_eltwise_param()->set_operation(
				caffe::EltwiseParameter_EltwiseOp_PROD);
	} else if (mxnetNode.strOp == "BatchNorm") {
		caffeLayer.set_type("BatchNorm");
		auto &bnParam = *caffeLayer.mutable_batch_norm_param();
		optionalAttrs["eps"] = [&](std::string strVal) {
			double dEpsilon = Str2Num<double>(strVal, 0., 1.);
			bnParam.set_eps((float)dEpsilon);
		};
		optionalAttrs["use_global_stats"] = [&](std::string strVal) {
			bool bUseGlobal = Str2Bool(strVal);
			bnParam.set_use_global_stats(bUseGlobal);
		};
		optionalAttrs["momentum"] = [&](std::string strVal) {
			float fMomentum = Str2Num<float>(strVal);
			bnParam.set_moving_average_fraction(fMomentum);
		};
		optionalAttrs["output_mean_var"]; // ignored
		optionalAttrs["cudnn_off"]; // ignored
		//optionalAttrs["fix_gamma"]; // unsupported
		//optionalAttrs["axis"]; // unsupported
	} else {
		LOG(FATAL) << "Unsupported op: " << mxnetNode.strOp;
	}

	auto ProcAttrs = [&](const AttrProcMap &procMap) {
		for (auto iAttr = mxnetNode.attrs.begin();
				iAttr != mxnetNode.attrs.end(); ) {
			auto iProc = procMap.find(iAttr->first);
			if (iProc != procMap.end()) {
				iProc->second(iAttr->second);
				iAttr = mxnetNode.attrs.erase(iAttr);
			} else {
				iAttr++;
			}
		}
	};

	ProcAttrs(requiredAttrs);
	ProcAttrs(optionalAttrs);
	return cvtInfo;
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

void ExpandOrMergeLayers(std::vector<caffe::LayerParameter> &layers) {
	for (auto iLayer = layers.begin(); iLayer != layers.end(); ) {
		if (iLayer->type() == "BatchNorm") {
			CHECK_EQ(iLayer->bottom_size(), 5);
			CHECK_EQ(iLayer->top_size(), 1);
			std::string strLayerName = iLayer->name();
			std::string strOutputName = iLayer->top(0);
			std::string strInputGamma = iLayer->bottom(1);
			std::string strInputBeta = iLayer->bottom(2);
			auto *pBottom = iLayer->mutable_bottom();
			pBottom->erase(pBottom->begin() + 1, pBottom->begin() + 3);

			iLayer = layers.insert(++iLayer, caffe::LayerParameter());
			iLayer->set_name(strLayerName + "_scale");
			iLayer->set_type("Scale");
			iLayer->add_bottom(strOutputName);
			iLayer->add_bottom(strInputGamma);
			iLayer->add_bottom(strInputBeta);
			iLayer->add_top(strOutputName);
			iLayer->mutable_scale_param()->set_bias_term(true);

			++iLayer;
		} else if (iLayer->type() == "Flatten" ||
				GuessBlobIDFromInputName(iLayer->name()) >= 0) {
			iLayer = layers.erase(iLayer);
		} else {
			++iLayer;
		}
	}
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

caffe::NetParameter MxnetNodes2CaffeNet(
		const std::vector<MxnetNode> &mxnetNodes,
		const std::vector<size_t> &headIndices,
		const std::vector<InputInfo> &inputInfos) {
	auto sortedIndices = SortIndicesByDependencies(mxnetNodes, headIndices);
	std::vector<caffe::LayerParameter> caffeLayers;

	std::map<std::string, size_t> typeCnt; // for unamed layers
	for (size_t i = 0; i < sortedIndices.size(); ++i) {
		auto &mxnetNode = mxnetNodes[sortedIndices[i]];
		caffe::LayerParameter caffeLayer;
		auto cvtInfo = MxnetNode2CaffeLayer(mxnetNode, caffeLayer);

		// to give unamed layer a name
		if (caffeLayer.name().empty()) {
			size_t nTypeCnt = ++typeCnt[caffeLayer.type()];
			caffeLayer.set_name("_unamed_" + caffeLayer.type() +
					std::to_string(nTypeCnt));
		}

		// convert inputs
		for (auto mxnetIdx : mxnetNode.inputs) {
			auto iCaffeIdx = std::find(sortedIndices.begin(),
					sortedIndices.end(), mxnetIdx.first);
			CHECK(iCaffeIdx != sortedIndices.end());
			CHECK_LT(*iCaffeIdx, i);
			auto &prevOutputs = caffeLayers[*iCaffeIdx].top();
			CHECK_LT(mxnetIdx.second, prevOutputs.size());
			caffeLayer.add_bottom(prevOutputs.Get(mxnetIdx.second));
		}

		// Convert outputs
		if (cvtInfo.bInPlace) {
			CHECK_EQ(cvtInfo.nOutNum, 1);
			caffeLayer.add_top(caffeLayer.bottom().Get(0));
		} else {
			for (size_t i = 0; i < cvtInfo.nOutNum; ++i) {
				caffeLayer.add_top(caffeLayer.name());
			}
			if (cvtInfo.nOutNum > 1) {
				for (size_t i = 0; i < cvtInfo.nOutNum; ++i) {
					std::string &strTop = *caffeLayer.mutable_top(i);
					strTop += std::to_string(i);
				}
			}
		}

		// Put this layer to vector
		caffeLayers.emplace_back(std::move(caffeLayer));
	}

	ExpandOrMergeLayers(caffeLayers);
	
	caffe::NetParameter net;
	for (auto &layer : caffeLayers) {
		auto iInputInfo = std::find_if(inputInfos.begin(), inputInfos.end(),
				[&](const InputInfo &ii) {
					return (layer.name() == ii.first);
				}
			);
		if (iInputInfo != inputInfos.end()) {
			auto *pShape = layer.mutable_input_param()->add_shape();
			for (auto d : iInputInfo->second) {
				pShape->add_dim((int)d);
			}
		}
		net.add_layer()->CopyFrom(layer);
	}

	return net;
}

bool IsEndWith(const std::string &strString, const std::string &strSuffix) {
	if (strString.length() >= strSuffix.length()) {
		return (0 == strString.compare(strString.length() - strSuffix.length(),
				strSuffix.length(), strSuffix));
	}
	return false;
}

