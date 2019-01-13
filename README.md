
# Mxnet2Caffe: A tool to convert model from MXNet to Caffe
- Modern C++ Practice
- High performance
- No dependency execpt for Caffe

## Required Packages
[BVLC/Caffe](https://github.com/BVLC/caffe)
You must clearly know the absolute path where the caffe installed. If you believe your caffe has installed in `YOUR_CAFFE_INSTALL_DIR`, like `/usr/local`, try to verify the caffe files do really exsist:
```
ls YOUR_CAFFE_INSTALL_DIR/include/caffe/caffe.hpp
ls YOUR_CAFFE_INSTALL_DIR/lib/libcaffe.so
```
Please DO NOT complain any building failure before you did above confirmation yet.

## Build
```
mkdir build && cd build
#If your want to speicfy a path Caffe installed, you shold defined
#  the variable CAFFE_INSTALL_DIR from where cmake to find Caffe.
cmake .. # or cmake -DCAFFE_INSTALL_DIR=<YOUR_CAFFE_INSTALL_PATH> ..
make # or make -j8
```

## To Prepare a MxNet Model
A MxNet model consist of a symbol file (\*.json) and a parameters file (\*.params). In a more general setting, the two files should be named
 - `<model_name>-symbol.json` and
 - `<model_name>-0000.params`.

Mxnet2Caffe use an independent config json file to describe your model and tell it how to do the conversion. Following is a example for the config json file:
```
{
  "mxnet_json" : "resnet-18-symbol.json",
  "mxnet_params" : "resnet-18-0000.params",
  "caffe_prototxt" : "resenet-18.prototxt",
  "caffe_caffemodel" : "resenet-18.caffemodel",
  "inputs" : [
      {"name" : "data", "shape" : [1, 3, 224, 224]},
      {"name" : "softmax_label", "shape" : [1]}
    ]
}
```
Find more examples in sub-path `./examples`.

### Properties used by the config json:
It should be very clear in the above example.

### Running the conversion:
Simply `./mxnet2caffe config.json` and the Caffe model will be presented by your configurations.

