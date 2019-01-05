/*
 * istream_helper.hpp
 *
 *  Created on: Jan 5, 2019
 *      Author: devymex
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
