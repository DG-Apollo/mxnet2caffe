/*
 * istream_helper.cpp
 *
 *  Created on: Jan 5, 2019
 *      Author: devymex
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

