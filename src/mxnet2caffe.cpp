
#include <functional>
#include <iostream>
#include <fstream>
#include <map>

#include "logging.hpp"
#include "common.hpp"
#include "json_helper.hpp"
#include "mxnet_parser.hpp"
#include "caffe_layer.hpp"
#include "converter.hpp"

namespace proto = google::protobuf;
using InputInfo = std::pair<std::string, Shape>;

struct ProgramOptions {
	std::string strMxnetJson;
	std::string strMxnetParams;
	std::string strCaffeProto;
	std::string strCaffeModel;
	std::vector<InputInfo> inputInfos;
};

StringPair SplitString(std::string str, size_t nPos) {
	if (nPos == 0) {
		return std::make_pair(std::string(), std::move(str));
	} else if (nPos >= str.size()) {
		return std::make_pair(std::move(str), std::string());
	}
	return std::make_pair(str.substr(0, nPos),
			str.substr(nPos, str.size() - nPos));
}

bool ParseArgument(int nArgCnt, char **ppArgs, ProgramOptions &po) {
	if (nArgCnt < 2) {
		return false;
	}
	std::string strConfFn = ppArgs[1];
	std::string strWorkPath;
	auto pathAndName = SplitString(strConfFn,
			strConfFn.find_last_of("\\/") + 1);
	if (!pathAndName.second.empty()) {
		strWorkPath = pathAndName.first;
	}

	std::ifstream configFile(strConfFn);
	CHECK(configFile.is_open());
	Json jConfig;
	configFile >> jConfig;
	configFile.close();

	po.strMxnetJson = strWorkPath + std::string(jConfig["mxnet_json"]);
	po.strMxnetParams = strWorkPath + std::string(jConfig["mxnet_params"]);
	po.strCaffeProto = strWorkPath + std::string(jConfig["caffe_prototxt"]);
	po.strCaffeModel = strWorkPath + std::string(jConfig["caffe_caffemodel"]);

	Json::iterator jInputs = jConfig.find("inputs");
	po.inputInfos = ParseArray<InputInfo>(jInputs,
			[](Json::iterator iInput) -> InputInfo {
				std::string strName = (*iInput)["name"];
				CHECK(!strName.empty());
				Json::iterator jShape = iInput->find("shape");
				CHECK(jShape != iInput->end());
				auto intShape = ParseArray<int>(jShape);
				CHECK(!intShape.empty());
				Shape shape;
				for (auto s : intShape) {
					CHECK_GT(s, 0);
					shape.emplace_back(s);
				}
				return std::make_pair(strName, shape);
			}
		);
	CHECK(!po.inputInfos.empty());
	return true;
}

std::string GenerateModelName(std::string strProtoFn) {
	auto pathAndName = SplitString(strProtoFn,
			strProtoFn.find_last_of("\\/") + 1);
	if (pathAndName.second.empty()) {
		std::swap(pathAndName.first, pathAndName.second);
	}
	strProtoFn = pathAndName.second;
	auto nameAndExt = SplitString(strProtoFn, strProtoFn.find_last_of("."));
	if (nameAndExt.first.empty()) {
		std::swap(nameAndExt.first, nameAndExt.second);
	}
	return nameAndExt.first;
}

/*
void AssignBlobs(std::vector<CaffeLayer> &layers,
		const std::vector<MxnetParam> &mxnetParams) {
	for (auto &layer : layers) {
		auto &inputs = layer.mutable_bottom();
		for (auto iInput = inputs.begin(); iInput != inputs.end(); ) {
			std::string &strInputName = *iInput;
			int nBlobId = GuessBlobIDFromInputName(strInputName);
			if (nBlobId >= 0) {
				auto iParam = std::find_if(mxnetParams.begin(),
						mxnetParams.end(), [&](const MxnetParam &param) {
							return IsEndWith(param.strName, strInputName);
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
*/
int main(int nArgCnt, char *ppArgs[]) {
	ProgramOptions po;
	if (!ParseArgument(nArgCnt, ppArgs, po)) {
		return -1;
	}

	auto mxnetParseResult = ParseMxnetJson(po.strMxnetJson);
	auto mxnetParams = LoadMxnetParam(po.strMxnetParams);
	auto protoNet = MxnetNodes2CaffeLayers(mxnetParseResult.first,
			mxnetParseResult.second, mxnetParams, po.inputInfos);
	protoNet.set_name(GenerateModelName(po.strCaffeProto));

	std::string strProtoBuf;
	proto::TextFormat::PrintToString(protoNet, &strProtoBuf);

	std::ofstream protoFile(po.strCaffeProto);
	protoFile.write(strProtoBuf.data(), strProtoBuf.size());
	protoFile.close();

	caffe::Net<float> net(protoNet);
	auto &layers = net.layers();
	for (auto &netLayer : layers) {
		std::string strLayerName = netLayer->layer_param().name();
		auto iProtoLayer = std::find_if(protoNet.layer.begin(),
				protoNet.layer.end(),
				[&strLayerName](const caffe::LayerParameter &protoLayer) {
					std::string strCompose = protoLayer.name() + "_";
					strCompose = strCompose + strCompose + "0_split";
					return protoLayer.name() == strLayerName) ||
							strCompose == strLayerName;
				}
			);
		CHECK(iProtoLayer != protoNet.layer.end());
		auto &inputs = iProtoLayer->bottom();
		for (auto &strInputName : inputs) {
			auto iParam = std::find_if(mxnetParams.begin(),
					mxnetParams.end(), [&](const MxnetParam &param) {
						return IsEndWith(param.strName, strInputName);
					}
				);
			if (iParam != mxnetParam.end()) {
				int nBlobId = GuessBlobIDFromInputName(strInputName);
				CHECK_GE(nBlobId, 0);
				CHECK_LT(nBlobId, netLayer.blobs.size());
				auto &pNetBlob = iParam->blobs()[nBlobId];
				CHECK_EQ(pNetBlob->count(), iParam->data.size());
				memcpy(pNetBlob->mutable_cpu_data(), iParam->data.size(),
						pNetBlob->count() * sizeof(float));
			}
		}
	}

	net.ToProto(&protoNet, false);
	caffe::WriteProtoToBinaryFile(protoNet, po.strCaffeModel.c_str());

	return 0;
}

