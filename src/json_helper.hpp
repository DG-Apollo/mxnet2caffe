/**
* Copyright (C) DeepGlint, Inc - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
*
* Json helper: Load vector from a json
*
* Written by Devymex <yumengwang@deepglint.com>, Jan. 2019
*/

#ifndef JSON_HELPER_HPP_
#define JSON_HELPER_HPP_

#include "json.hpp"

using Json = nlohmann::json;

template<typename _Ty>
std::vector<_Ty> ParseArray(Json::iterator &jAry) {
	std::vector<_Ty> ary;
	for (Json::iterator i = jAry->begin(); i != jAry->end(); ++i) {
		ary.push_back(i->get<_Ty>());
	}
	return std::move(ary);
}

template<typename _Ty>
std::vector<_Ty> ParseArray(Json::iterator jAry,
		std::function<_Ty(Json::iterator)> elemProc) {
	std::vector<_Ty> ary;
	for (Json::iterator i = jAry->begin(); i != jAry->end(); ++i) {
		ary.push_back(elemProc(i));
	}
	return std::move(ary);
}



#endif /* JSON_HELPER_HPP_ */
