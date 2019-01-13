
# Mxnet2Caffe: A tool to convert model from MXNet to Caffe
- Modern C++ Practice
- High performance
- No dependency execpt for Caffe

## Required Packages
[BVLC/Caffe](https://github.com/BVLC/caffe)

You must clearly know the absolute path where your caffe installed. If your caffe is believed to have installed in `YOUR_CAFFE_INSTALL_DIR`, like `/usr/local`, try to verify the caffe files do really exsist:
```
ls YOUR_CAFFE_INSTALL_DIR/include/caffe/caffe.hpp
ls YOUR_CAFFE_INSTALL_DIR/lib/libcaffe.so
```
Please DO NOT complain any building failure before you did all above confirmations yet.

## Build
```
mkdir build && cd build
cmake -DCAFFE_INSTALL_DIR=<YOUR_CAFFE_INSTALL_PATH> ..
make # or make -j8
```

## To Prepare a MxNet Model
A MxNet model consist of a symbol file (\*.json) and a parameters file (\*.params). In a more general setting, the two files should be named like:
 - `<model_name>-symbol.json` and
 - `<model_name>-0000.params`.

Mxnet2Caffe use an independent config json file to describe your model and obtain configurations of conversion from it. Following is an example of the config json file:
```
{
	"mxnet_json" : "squeezenet_v1.1-symbol.json",
	"mxnet_params" : "squeezenet_v1.1-0000.params",
	"caffe_prototxt" : "squeezenet_v1.1.prototxt",
	"caffe_caffemodel" : "squeezenet_v1.1.caffemodel",
	"inputs" : [
			{"name" : "data", "shape" : [1, 3, 224, 224]},
			{"name" : "prob_label", "shape" : [1]}
		]
}
```
Find more examples in sub-path `./examples`.

### Properties used by the config json:
It should be very clear in the above example.

### Running the conversion:
Simply run command `./mxnet2caffe config.json` and a Caffe model will be presented by your configurations.
