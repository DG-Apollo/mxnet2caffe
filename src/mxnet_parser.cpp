
#include "mxnet_parser.hpp"
#include <fstream>
#include "logging.hpp"
#include "json_helper.hpp"

MxnetNode ParseMxnetNode(Json::iterator jNode) {
	MxnetNode node;
	for (Json::iterator jField = jNode->begin();
			jField != jNode->end(); ++jField) {
		if (jField.key() == "op") {
			node.strOp = jField.value();
		} else if (jField.key() == "name") {
			node.strName = jField.value();
		} else if (jField.key() == "attrs") {
			node.attrs = ParseArray<StringPair>(jField,
					[](Json::iterator jAttr) {
						return std::make_pair(jAttr.key(), jAttr.value());
					});
		} else if (jField.key() == "inputs") {
			node.inputs = ParseArray<MxnetInput>(jField,
					[](Json::iterator jInput) {
						CHECK(jInput->is_array());
						auto inputIndices = ParseArray<size_t>(jInput);
						CHECK_EQ(inputIndices.size(), 3U);
						return std::make_pair(inputIndices[0], inputIndices[1]);
					});
		}
	}
	return std::move(node);
}

std::pair<std::vector<MxnetNode>, std::vector<size_t>> ParseMxnetJson(
		const std::string &strFile) {
	std::ifstream jsonFile(strFile);
	CHECK(jsonFile.is_open()) << strFile;
	Json jModel;
	jsonFile >> jModel;
	jsonFile.close();

	std::vector<MxnetNode> nodes;
	std::vector<size_t> headIndices;
	std::vector<size_t> argIndices;
	for (Json::iterator jField = jModel.begin();
			jField != jModel.end(); ++jField) {
		if (jField.key() == "nodes") {
			nodes = ParseArray<MxnetNode>(jField, ParseMxnetNode);
		} else if (jField.key() == "headIndices") {
			headIndices = ParseArray<size_t>(jField);
		} else if (jField.key() == "arg_nodes") {
			argIndices = ParseArray<size_t>(jField);
		} else if (jField.key() == "attrs") {
		} else if (jField.key() == "node_row_ptr") {
		}
	}
	for (auto iArgIdx : argIndices) {
		CHECK(nodes[iArgIdx].strOp == "null");
	}

	return std::make_pair(std::move(nodes), std::move(headIndices));
}


std::vector<MxnetParam> LoadMxnetParam(std::string strModelFn) {
	FILE* fp = fopen(strModelFn.c_str(), "rb");
	CHECK(fp);

	std::vector<MxnetParam> params;

	uint64_t header;
	uint64_t reserved;
	fread(&header, 1, sizeof(uint64_t), fp);
	fread(&reserved, 1, sizeof(uint64_t), fp);

	uint64_t data_count;
	fread(&data_count, 1, sizeof(uint64_t), fp);

	for (int i = 0; i < (int)data_count; i++) {
		uint32_t magic;// 0xF993FAC9
		fread(&magic, 1, sizeof(uint32_t), fp);
		// shape
		uint32_t ndim;
		std::vector<int64_t> shape;
		if (magic == 0xF993FAC9) {
			int32_t stype;
			fread(&stype, 1, sizeof(int32_t), fp);
			fread(&ndim, 1, sizeof(uint32_t), fp);
			shape.resize(ndim);
			fread(&shape[0], 1, ndim * sizeof(int64_t), fp);
		} else if (magic == 0xF993FAC8)	{
			fread(&ndim, 1, sizeof(uint32_t), fp);
			shape.resize(ndim);
			fread(&shape[0], 1, ndim * sizeof(int64_t), fp);
		} else {
			ndim = magic;
			shape.resize(ndim);
			std::vector<uint32_t> shape32;
			shape32.resize(ndim);
			fread(&shape32[0], 1, ndim * sizeof(uint32_t), fp);
			for (int j=0; j<(int)ndim; j++) {
				shape[j] = shape32[j];
			}
		}

		// context
		int32_t dev_type;
		int32_t dev_id;
		fread(&dev_type, 1, sizeof(int32_t), fp);
		fread(&dev_id, 1, sizeof(int32_t), fp);

		int32_t type_flag;
		fread(&type_flag, 1, sizeof(int32_t), fp);

		// data
		size_t len = 0;
		if (shape.size() == 1) len = shape[0];
		if (shape.size() == 2) len = shape[0] * shape[1];
		if (shape.size() == 3) len = shape[0] * shape[1] * shape[2];
		if (shape.size() == 4) len = shape[0] * shape[1] * shape[2] * shape[3];

		MxnetParam p;
		p.data.resize(len);
		fread(&p.data[0], 1, len * sizeof(float), fp);
		params.push_back(p);
	}
	uint64_t name_count;
	fread(&name_count, 1, sizeof(uint64_t), fp);
	for (int i = 0; i < (int)name_count; i++) {
		uint64_t len;
		fread(&len, 1, sizeof(uint64_t), fp);
		MxnetParam& p = params[i];
		p.strName.resize(len);
		fread((char*)p.strName.data(), 1, len, fp);
		if (memcmp(p.strName.c_str(), "arg:", 4) == 0) {
			p.strName = std::string(p.strName.c_str() + 4);
		}
		if (memcmp(p.strName.c_str(), "aux:", 4) == 0) {
			p.strName = std::string(p.strName.c_str() + 4);
		}
	}

	fclose(fp);
	return std::move(params);
}


