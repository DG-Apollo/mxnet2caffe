# FindCaffe.cmake
# ------------
#	You can specify the path to caffe files in CAFFE_HOME
#
#	This will define the following variables:
#	CAFFE_FOUND			- True if the system has the Inference Engine library
#	CAFFE_INCLUDE_DIRS	- TensorRT include directories
#	CAFFE_LIBRARIES		- TensorRT libraries

INCLUDE(FindPackageHandleStandardArgs)

FIND_PATH(CAFFE_INCLUDE_DIR
	NAMES
		caffe/caffe.hpp
		caffe/common.hpp
		caffe/net.hpp
		caffe/util/io.hpp
		caffe/proto/caffe.pb.h
	PATH_SUFFIXES include
	HINTS 
		"${CAFFE_HOME}/distribute"
		"${CAFFE_HOME}/build/install"
		"$ENV{CAFFE_HOME}/distribute"
		"$ENV{CAFFE_HOME}/build/install"
	)

FIND_LIBRARY(CAFFE_LIBRARIES
	NAMES caffe
	PATH_SUFFIXES lib
	HINTS 
		"${CAFFE_HOME}/distribute"
		"${CAFFE_HOME}/build/install"
		"$ENV{CAFFE_HOME}/distribute"
		"$ENV{CAFFE_HOME}/build/install"
	)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Caffe
	FOUND_VAR CAFFE_FOUND
	REQUIRED_VARS CAFFE_INCLUDE_DIR
	REQUIRED_VARS CAFFE_LIBRARIES
	)

