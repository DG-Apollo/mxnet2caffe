/**
* Copyright (C) DeepGlint, Inc - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
*
* Conversion from structurized MxNet nodes to a caffe::NetParameter
*
* Written by Devymex <yumengwang@deepglint.com>, Jan. 2019
*/


#include "converter.hpp"

#include <algorithm>
#include <functional>
#include <numeric>

#define CPU_ONLY
#include <caffe/caffe.hpp>
#include <glog/logging.h>

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
	return ret;
}

template<typename _Ty>
std::vector<_Ty> Str2Tuple(std::string str) {
	CHECK(!str.empty());
	std::string::size_type beg = str.find('(');
	std::string::size_type end = str.rfind(')');
	CHECK_NE(beg, std::string::npos);
	CHECK_NE(end, std::string::npos);
	CHECK_GT(end - beg, 1);
	str.erase(end, -1);
	str.erase(0, beg + 1);
	std::istringstream iss(str);
	std::vector<_Ty> ret;
	for (std::string strVal; std::getline(iss, strVal, ','); ) {
		std::istringstream isv(strVal);
		_Ty val;
		isv >> val;
		ret.push_back(val);
	}
	return ret;
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
	AttrProcMap reqAttrProcs;
	AttrProcMap optAttrProcs;
	if (mxnetNode.strOp == "null") {
		caffeLayer.set_type("Input");
	} else if (mxnetNode.strOp == "Flatten") {
		caffeLayer.set_type("Flatten");
		cvtInfo.bInPlace = true;
	} else if (mxnetNode.strOp == "Activation") {
		cvtInfo.bInPlace = true;
		reqAttrProcs["act_type"] = [&](std::string strVal) {
			if (strVal == "relu") {
				caffeLayer.set_type("ReLU");
			} else if (strVal == "Sigmoid" || strVal == "sigmoid") {
				caffeLayer.set_type("Sigmoid");
			} else if (strVal == "tanh") {
				caffeLayer.set_type("TanH");
			} else {
				LOG(FATAL);
			}
		};
	} else if (mxnetNode.strOp == "LeakyReLU") {
		std::string strActType = mxnetNode.attrs.GetValue("act_type", false);
		if (strActType.empty()) {
			strActType = "leaky";
		}
		if (strActType == "leaky") {
			caffeLayer.set_type("ReLU");
			reqAttrProcs["slope"] = [&](std::string strArg) {
				float fSlope = Str2Num<float>(strArg, 0.f, 1.f);
				caffeLayer.mutable_relu_param()->set_negative_slope(fSlope);
			};
		} else if (strActType == "elu") {
			caffeLayer.set_type("ELU");
			reqAttrProcs["slope"] = [&](std::string strArg) {
				float fAlpha = Str2Num<float>(strArg, 0.f, 1.f);
				caffeLayer.mutable_elu_param()->set_alpha(fAlpha);
			};
		} else if (strActType == "prelu") {
			caffeLayer.set_type("PReLU");
		} else {
			LOG(FATAL) << "Unsupported act_type \"" << strActType <<
					"\" of LeakyRelu";
		}
		optAttrProcs["gamma"];
		optAttrProcs["act_type"];
		optAttrProcs["lower_bound"];
		optAttrProcs["upper_bound"];
	} else if (mxnetNode.strOp == "abs") {
		caffeLayer.set_type("AbsVal");
	} else if (mxnetNode.strOp == "SoftmaxActivation") {
		caffeLayer.set_type("Softmax");
		optAttrProcs["mode"] = [&](std::string strVal) {
			if (!strVal.empty()) {
				if (strVal == "channel") {
					caffeLayer.mutable_softmax_param()->set_axis(0);
				} else {
					CHECK(strVal == "instance");
				}
			}
		};
	} else if (mxnetNode.strOp == "softmax") {
		caffeLayer.set_type("Softmax");
		optAttrProcs["axis"] = [&](std::string strVal) {
			int nAxis = Str2Num<int>(strVal, -1, 4);
			CHECK(nAxis == -1);
		};
		optAttrProcs["temperature"] = [&](std::string strVal) {
			double dTemp = Str2Num<double>(strVal);
			CHECK (dTemp == 1.0);
		};
	} else if (mxnetNode.strOp == "SliceChannel") {
		caffeLayer.set_type("Slice");
		reqAttrProcs["num_outputs"] = [&](std::string strVal) {
			cvtInfo.nOutNum = Str2Num<int>(strVal, 2);
		};
		optAttrProcs["axis"] = [&](std::string strVal) {
			int nAxis = Str2Num<int>(strVal, 0, 4);
			if (nAxis != 1) {
				caffeLayer.mutable_slice_param()->set_axis(nAxis);
			}
		};
		optAttrProcs["squeeze_axis"] = [&](std::string strVal) {
			bool bSqueeze = Str2Bool(strVal);
			CHECK(!bSqueeze);
		};
	} else if (mxnetNode.strOp == "concat" || mxnetNode.strOp == "Concat") {
		caffeLayer.set_type("Concat");
		optAttrProcs["dim"] = [&](std::string strVal) {
			int nDim = Str2Num<int>(strVal, 1);
			if (nDim != 1) {
				caffeLayer.mutable_concat_param()->set_axis(nDim);
			}
		};
		optAttrProcs["num_args"]; // ignored
	} else if (mxnetNode.strOp == "Dropout") {
		caffeLayer.set_type("Dropout");
		optAttrProcs["p"] = [&](std::string strVal) {
			float fRatio = Str2Num<float>(strVal, 0.f, 1.f);
			if (fRatio != 0.5f) {
				caffeLayer.mutable_dropout_param()->set_dropout_ratio(fRatio);
			}
		};
		optAttrProcs["axes"]; // ignored
		optAttrProcs["mode"]; // ignored
	} else if (mxnetNode.strOp == "FullyConnected") {
		caffeLayer.set_type("InnerProduct");
		reqAttrProcs["num_hidden"] = [&](std::string strVal) {
			int nNumHid = Str2Num<int>(strVal, 1);
			caffeLayer.mutable_inner_product_param()->set_num_output(nNumHid);
		};
		optAttrProcs["no_bias"] = [&](std::string strVal) {
			bool bNoBias = Str2Bool(strVal);
			if (bNoBias) {
				caffeLayer.mutable_inner_product_param()->set_bias_term(false);
			}
		};
		optAttrProcs["flatten"]; // ignored;
	} else if (mxnetNode.strOp == "Convolution") {
		caffeLayer.set_type("Convolution");
		auto &convParam = *caffeLayer.mutable_convolution_param();
		reqAttrProcs["num_filter"] = [&](std::string strVal) {
			int nNumChs = Str2Num<int>(strVal, 1);
			convParam.set_num_output(nNumChs);
		};
		reqAttrProcs["kernel"] = [&](std::string strVal) {
			auto kernel = Str2Pair<int>(strVal, 1);
			if (kernel.first == kernel.second) {
				convParam.add_kernel_size(kernel.first);
			} else {
				convParam.set_kernel_h(kernel.first);
				convParam.set_kernel_w(kernel.second);
			}
		};
		optAttrProcs["stride"] = [&](std::string strVal) {
			auto stride = Str2Pair<int>(strVal, 1);
			if (stride.first == stride.second) {
				if (stride.first != 1) {
					convParam.add_stride(stride.first);
				}
			} else {
				convParam.set_stride_h(stride.first);
				convParam.set_stride_w(stride.second);
			}
		};
		optAttrProcs["pad"] = [&](std::string strVal) {
			auto pad = Str2Pair<int>(strVal, 0);
			if (pad.first == pad.second) {
				if (pad.first != 0) {
					convParam.add_pad(pad.first);
				}
			} else {
				convParam.set_pad_h(pad.first);
				convParam.set_pad_w(pad.second);
			}
		};
		optAttrProcs["dilate"] = [&](std::string strVal) {
			int nDilate = Pair2Num(Str2Pair<int>(strVal, 0));
			if (nDilate != 1) {
				convParam.add_dilation(nDilate);
			}
		};
		optAttrProcs["num_group"] = [&](std::string strVal) {
			int nNumChs = convParam.num_output();
			int nNumGroup = Str2Num(strVal, 1, nNumChs);
			CHECK_EQ(nNumChs % nNumGroup, 0);
			if (nNumGroup != 1) {
				int nGroupSize = nNumGroup;
				convParam.set_group(nGroupSize);
			}
		};
		optAttrProcs["no_bias"] = [&](std::string strVal) {
			bool bNoBias = Str2Bool(strVal);
			if (bNoBias) {
				convParam.set_bias_term(false);
			}
		};
		optAttrProcs["layout"] = [&](std::string strVal) {
			CHECK(strVal == "None");
		};
		optAttrProcs["workspace"]; // ignored
		optAttrProcs["cudnn_tune"]; // ignored
		optAttrProcs["cudnn_off"]; // ignored
	} else if (mxnetNode.strOp == "Pooling") {
		caffeLayer.set_type("Pooling");
		auto &poolParam = *caffeLayer.mutable_pooling_param();
		optAttrProcs["pool_type"] = [&](std::string strVal) {
			if (strVal == "avg") {
				poolParam.set_pool(caffe::PoolingParameter_PoolMethod_AVE);
			} else if(strVal != "max") {
				LOG(FATAL) << "Unsupported pooling method: " << strVal;
			}
		};
		optAttrProcs["kernel"] = [&](std::string strVal) {
			if (!poolParam.global_pooling()) {
				auto kernel = Str2Pair<int>(strVal, 0);
				if (kernel.first == kernel.second) {
					poolParam.set_kernel_size(kernel.first);
				} else {
					poolParam.set_kernel_h(kernel.first);
					poolParam.set_kernel_w(kernel.second);
				}
			}
		};
		optAttrProcs["stride"] = [&](std::string strVal) {
			auto stride = Str2Pair<int>(strVal, 1);
			if (stride.first == stride.second) {
				if (stride.first != 1) {
					poolParam.set_stride(stride.first);
				}
			} else {
				poolParam.set_stride_h(stride.first);
				poolParam.set_stride_w(stride.second);
			}
		};
		optAttrProcs["pad"] = [&](std::string strVal) {
			auto pad = Str2Pair<int>(strVal, 0);
			if (pad.first == pad.second) {
				if (pad.first != 0) {
					poolParam.set_pad(pad.first);
				}
			} else {
				poolParam.set_pad_h(pad.first);
				poolParam.set_pad_w(pad.second);
			}
		};
		optAttrProcs["global_pool"] = [&](std::string strVal) {
			bool bGlobalPool = Str2Bool(strVal);
			if (bGlobalPool) {
				poolParam.set_global_pooling(true);
				poolParam.clear_kernel_size();
			}
		};
		optAttrProcs["pooling_convention"] = [&](std::string strVal) {
			//CHECK(strVal == "full");
		};
		optAttrProcs["p_value"] = [&](std::string strVal) {
			LOG(FATAL) << "Lp pooling is not supported";
		};
		optAttrProcs["count_include_pad"] = [&](std::string strVal) {
			LOG(FATAL) << "count_include_pad is not supported";
		};
		optAttrProcs["cudnn_off"]; // ignored
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
		optAttrProcs["eps"] = [&](std::string strVal) {
			double dEpsilon = Str2Num<double>(strVal, 0., 1.);
			bnParam.set_eps((float)dEpsilon);
		};
		optAttrProcs["use_global_stats"] = [&](std::string strVal) {
			bool bUseGlobal = Str2Bool(strVal);
			bnParam.set_use_global_stats(bUseGlobal);
		};
		optAttrProcs["momentum"] = [&](std::string strVal) {
			float fMomentum = Str2Num<float>(strVal);
			bnParam.set_moving_average_fraction(fMomentum);
		};
		optAttrProcs["fix_gamma"] = [&](std::string strVal) {
			if (strVal == "True" || strVal == "true" || strVal == "1" ) {
				caffeLayer.add_param(); // just a tag for fix_gamma
			}
		};
		optAttrProcs["axis"] = [&](std::string strVal) {
			int nAxis = Str2Num<int>(strVal);
			CHECK(nAxis == 1);
		};
		optAttrProcs["output_mean_var"]; // ignored
		optAttrProcs["cudnn_off"]; // ignored
		//optAttrProcs["axis"]; // unsupported
	} else if (mxnetNode.strOp == "SoftmaxOutput") {
		caffeLayer.set_type("SoftmaxWithLoss");
		optAttrProcs["grad_scale"] = [&](std::string strVal) {
			float fGradScale = Str2Num<float>(strVal);
			CHECK_EQ(fGradScale, 1.0f) << "grad_scale is not supported";
		};
		optAttrProcs["ignore_label"] = [&](std::string strVal) {
			int nIgnoreLabel = Str2Num<int>(strVal, -1);
			if (nIgnoreLabel != -1) {
				caffeLayer.mutable_loss_param()->set_ignore_label(nIgnoreLabel);
			}
		};
		optAttrProcs["multi_output"] = [&](std::string strVal) {
			bool bMultiOut = Str2Bool(strVal);
			if (bMultiOut) {
				//TODO:
			}
		};
		optAttrProcs["normalization"] = [&](std::string strVal) {
			if (strVal == "batch") {
				caffeLayer.mutable_loss_param()->set_normalization(
						caffe::LossParameter_NormalizationMode_BATCH_SIZE);
			} else if (strVal == "valid") {
				caffeLayer.mutable_loss_param()->set_normalization(
						caffe::LossParameter_NormalizationMode_VALID);
			} else {
				CHECK(strVal == "null");
			}
		};
		optAttrProcs["out_grad"] = [&](std::string strVal) {
			CHECK(strVal == "False" || strVal == "0");
		};
		optAttrProcs["smooth_alpha"] = [&](std::string strVal) {
			CHECK(strVal == "False" || strVal == "0");
		};
		optAttrProcs["preserve_shape"]; // ignored
		optAttrProcs["use_ignore"]; // ignored
	} else if (mxnetNode.strOp == "reshape" || mxnetNode.strOp == "Reshape") {
		caffeLayer.set_type("Reshape");
		optAttrProcs["shape"] = [&](std::string strVal) {
			auto shape = Str2Tuple<int>(strVal);
			CHECK_GT(shape.size(), 0);
			CHECK_LE(shape.size(), 4);
			auto *pShape = caffeLayer.mutable_reshape_param()->mutable_shape();
			for (auto s : shape) {
				//CHECK_GT(s, -2);
				pShape->add_dim(s);
			}
		};
	} else if (mxnetNode.strOp == "L2Normalization") {
		caffeLayer.set_type("Normalization");
		optAttrProcs["mode"] = [&](std::string strVal) {
			CHECK(strVal == "instance");
		};
	} else if (mxnetNode.strOp == "broadcast_mul") {
		caffeLayer.set_type("BroadcastMul");
	} else if (mxnetNode.strOp == "_mul_scalar") {
		caffeLayer.set_type("Power");
		optAttrProcs["scalar"] = [&](std::string strVal) {
			float fScalar = Str2Num<float>(strVal);
			caffeLayer.mutable_power_param()->set_scale(fScalar);
		};
	} else {
		LOG(FATAL) << "Unsupported op: " << mxnetNode.strOp;
	}

	auto ProcAttrs = [&](const AttrProcMap &procMap, bool bRequired) {
			for (auto attrProc : procMap) {
				std::string strVal = mxnetNode.attrs.GetValue(
						attrProc.first, bRequired);
				if (!strVal.empty() && attrProc.second != nullptr) {
					attrProc.second(strVal);
				}
				if (mxnetNode.attrs.HasValue(attrProc.first)) {
					mxnetNode.attrs.RemoveValue(attrProc.first);
				}
			}
		};

	ProcAttrs(reqAttrProcs, true);
	ProcAttrs(optAttrProcs, false);
	std::set<std::string> hiddenKeys = {
			"__ctx_group__",
			"__lr_mult__",
			"__wd_mult__",
			"__force_mirroring__",
			"__mirror_stage__"
		};
	if (mxnetNode.strOp != "null") {
		for (auto &attr : mxnetNode.attrs) {
			if (hiddenKeys.find(attr.first) == hiddenKeys.end()) {
				LOG(FATAL) << "Unknown attr \"" << attr.first <<
						"\" found in node \"" << mxnetNode.strName <<
						"\" (" <<mxnetNode.strOp << ")";
			}
		}
	}
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
			CHECK_GE(iLayer->bottom_size(), 3);
			CHECK_EQ(iLayer->top_size(), 1);
			std::string strLayerName = iLayer->name();
			std::string strOutputName = iLayer->top(0);
			std::string strInputGamma = iLayer->bottom(1);
			std::string strInputBeta = iLayer->bottom(2);
			auto *pBottom = iLayer->mutable_bottom();
			pBottom->DeleteSubrange(1, 2);
			//pBottom->erase(pBottom->begin() + 1, pBottom->begin() + 3);
			if (iLayer->bottom_size() == 1) {
				CHECK(IsEndWith(strInputGamma, "_gamma"));
				CHECK(IsEndWith(strInputBeta, "_beta"));
				std::string strMean = strInputGamma.substr(0,
						strInputGamma.size() - 6) + "_moving_mean";
				std::string strVar = strInputGamma.substr(0,
						strInputGamma.size() - 6) + "_moving_var";
				iLayer->add_bottom(strMean);
				iLayer->add_bottom(strVar);
			} else {
				CHECK(iLayer->bottom_size() == 3);
			}
			bool bFixedGamma = false;
			// If fix_gamma is set, a "param" should be added to the layer before
			if (iLayer->param_size() > 0) {
				CHECK_EQ(iLayer->param_size(), 1);
				bFixedGamma = true;
				iLayer->clear_param();
			}
			if (bFixedGamma) {
				auto *pParam = iLayer->add_param();
				pParam->set_decay_mult(100.);
				pParam->set_lr_mult(0.f);
				//iLayer->mutable_relu_param();
				auto *pFiller = iLayer->mutable_scale_param()->mutable_filler();
				pFiller->set_type("constant");
				pFiller->set_value(1.0f);
			}
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
		const std::vector<InputInfo> &inputInfos,
		std::map<std::string, std::vector<std::string>> &blobMapping) {
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
			for (int i = 0; i < cvtInfo.nOutNum; ++i) {
				caffeLayer.add_top(caffeLayer.name());
			}
			if (cvtInfo.nOutNum > 1) {
				for (int i = 0; i < cvtInfo.nOutNum; ++i) {
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
		} else {
			CHECK_GT(layer.bottom_size(), 0) << "Unmarked input node: " <<
					layer.name();
		}
		auto &bottoms = *layer.mutable_bottom();
		for (auto iBottom = bottoms.begin(); iBottom != bottoms.end(); ) {
			int nBlobID = GuessBlobIDFromInputName(*iBottom);
			if (nBlobID >= 0) {
				auto &blobVec = blobMapping[layer.name()];
				blobVec.resize(nBlobID + 1);
				blobVec[nBlobID] = std::move(*iBottom);
				bottoms.DeleteSubrange(iBottom - bottoms.begin(), 1);
			} else {
				++iBottom;
			}
		}
		net.add_layer()->CopyFrom(layer);
	}
	LOG(INFO) << caffeLayers[1].name();

	return net;
}

bool IsEndWith(const std::string &strString, const std::string &strSuffix) {
	if (strString.length() >= strSuffix.length()) {
		return (0 == strString.compare(strString.length() - strSuffix.length(),
				strSuffix.length(), strSuffix));
	}
	return false;
}
