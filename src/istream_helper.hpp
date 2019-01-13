/**
* Copyright (C) DeepGlint, Inc - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
*
* Istream helper: parsing formatted strings.
*
* Written by Devymex <yumengwang@deepglint.com>, Jan. 2019
*/

#ifndef ISTREAM_HELPER_HPP_
#define ISTREAM_HELPER_HPP_

#include <istream>

struct Expect {
	char expected;
	Expect(char expected);
};

std::istream& operator >> (std::istream& is, Expect const& e);

#endif /* ISTREAM_HELPER_HPP_ */
