
# Mxnet2Caffe: A tool to convert model from MXNet to Caffe
- Modern C++ Practice
- High performance
- No dependency execpt for Caffe

## Required Packages
[BVLC/Caffe](https://github.com/BVLC/caffe)

You must clearly know the absolute path where your caffe installed. If your caffe is believed to have installed in `YOUR_CAFFE_HOME`, please try to verify those caffe files do really exsist:
```
ls YOUR_CAFFE_HOME/distribute/include/caffe/caffe.hpp
ls YOUR_CAFFE_HOME/distribute/lib/libcaffe.so
```
or
```
ls YOUR_CAFFE_HOME/build/install/include/caffe/caffe.hpp
ls YOUR_CAFFE_HOME/build/install/lib/libcaffe.so
```
Please DO NOT complains any building failure before you really did all above confirmations.

## Build
```
mkdir build && cd build
cmake -DCAFFE_HOME=<YOUR_CAFFE_HOME> ..
make # or make -j8
```

## To Prepare a MxNet Model
A MxNet model consist of a symbol file (\*.json) and a parameters file (\*.params). In general settings, the two files should have name like:
 - `<model_name>-symbol.json` and
 - `<model_name>-0000.params`.

Mxnet2Caffe use an independent config json file to describe your model and to configure conversion settings. Following is an example of the config json file:
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
Simply run command `./mxnet2caffe config.json` and a Caffe model will be presented after conversion by your configurations.

