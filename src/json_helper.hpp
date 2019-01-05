/*
 * json_helper.hpp
 *
 *  Created on: Jan 5, 2019
 *      Author: devymex
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
