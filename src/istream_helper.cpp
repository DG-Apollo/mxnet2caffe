/**
* Copyright (C) DeepGlint, Inc - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
*
* Istream helper: parsing formatted strings.
*
* Written by Devymex <yumengwang@deepglint.com>, Jan. 2019
*/

#include "istream_helper.hpp"

std::istream& operator >> (std::istream& is, Expect const& e);

Expect::Expect(char expected) : expected(expected) {
}

std::istream& operator>>(std::istream& is, Expect const& e) {
	char actual;
	if ((is >> actual) && (actual != e.expected)) {
		is.setstate(std::ios::failbit);
	}
	return is;
}

