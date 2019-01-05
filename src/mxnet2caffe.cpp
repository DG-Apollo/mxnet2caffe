
#define CPU_ONLY

#include <functional>
#include <iostream>
#include <fstream>
#include <caffe/caffe.hpp>
#include <map>

#include "logging.hpp"
#include "common.hpp"
#include "json_helper.hpp"
#include "mxnet_parser.hpp"
#include "caffe_layer.hpp"
#include "converter.hpp"

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

int main(int nArgCnt, char *ppArgs[]) {
	ProgramOptions po;
	if (!ParseArgument(nArgCnt, ppArgs, po)) {
		return -1;
	}

	auto mxnetParseResult = ParseMxnetJson(po.strMxnetJson);
	auto mxnetParams = LoadMxnetParam(po.strMxnetParams);
	auto caffeLayers = MxnetNodes2CaffeLayers(mxnetParseResult.first,
			mxnetParseResult.second, mxnetParams, po.inputInfos);

	std::ofstream protoFile(po.strCaffeProto);
	protoFile << "name: \"" << GenerateModelName(po.strCaffeProto) << "\"\n";
	for (auto &l : caffeLayers) {
		protoFile << l;
	}
	protoFile.close();

	caffe::NetParameter netParams;
	caffe::ReadProtoFromTextFile(po.strCaffeProto.c_str(), &netParams);
	caffe::Net<float> net(netParams);
	auto layers = net.layers();
	for (auto &netLayer : layers) {
		std::string strLayerName = netLayer->layer_param().name();
		auto iCaffeLayer = std::find_if(caffeLayers.begin(), caffeLayers.end(),
				[&strLayerName](const CaffeLayer &layer) {
					return layer.strName == strLayerName;
				}
			);
		CHECK(iCaffeLayer != caffeLayers.end()) << strLayerName;
		auto blobs = netLayer->blobs();
		if (!blobs.empty()) {
			if (iCaffeLayer->blobs.size() != blobs.size()) {
				LOG(WARNING) << "NetLayer has " << blobs.size() <<
						" blobs, but caffeLayer has " <<
						iCaffeLayer->blobs.size() << " blobs";
			} else {
				for (size_t i = 0; i < blobs.size(); ++i) {
					CHECK_EQ(blobs[i]->count(), iCaffeLayer->blobs[i].size());
					memcpy(blobs[i]->mutable_cpu_data(),
							iCaffeLayer->blobs[i].data(),
							blobs[i]->count() * sizeof(float));
				}
			}
		}
	}

	net.ToProto(&netParams, false);
	caffe::WriteProtoToBinaryFile(netParams, po.strCaffeModel.c_str());

	return 0;
}

