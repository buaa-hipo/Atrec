This is a repository for code of "AtRec: Accelerating Recommendation Model Training on CPUs".

## ENV
- Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
- 64 GiB Memory
- Docker

## Start docker container using existing images

Atrec uses the following docker container:
```
docker run -it --net=host --ipc=host alideeprec/deeprec-build:deeprec-dev-cpu-py36-ubuntu18.04
```

DeepRec uses the same image as Atrec. For TensorFlow and PyTorch, the official images can be used. Take TensorFlow for example:
```
docker run -it --net=host --ipc=host tensorflow/tensorflow:2.11.0 # For TensorFlow
docker run -it --net=host --ipc=host tensorflow/tensorflow:2.11.0-gpu # For TensorFlow gpu
```

## Install Atrec

We recommend use the docker image `alideeprec/deeprec-build:deeprec-dev-cpu-py36-ubuntu18.04` and install the pre-built wheel directly. Use the following command in the repo root directory if you are using the recommended docker image:

```
pip install wheel/tensorflow-1.15.5+deeprec2208-cp36-cp36m-linux_x86_64.whl
```

DeepRec, TensorFlow and PyTorch can be fetched and installed from PyPI using pip directly.

## Build from Source (Skip if you are using the docker image given above)

Atrec can also be built from source. Use the following command to build and install Atrec if needed:

```
cd DeepRec
./configure # Keep pressing Enter along the configuration process to use all default options
bazel build --config=noaws --config=nogcp --config=nohdfs --config=nokafka --config=noignite --config=nonccl -c opt --config=opt --config=mkl_threadpool --define build_with_mkl_dnn_v1_only=true --copt=-O3 //tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

pip install /tmp/tensorflow_pkg/tensorflow-1.15.5+deeprec2208-cp36-cp36m-linux_x86_64.whl
```

## Extract The Dataset

Due to the size limitation, only the amz dataset is bundled in this repo. It can be extracted as follows:

```
cd dataset
tar xvf amz_book.tar.zst
```

Make sure git lfs support is correctly configured to get the dataset archive.

The other datasets used can be obtained through the references in the paper.


## Prepare The Dataset

The default data location of all training scripts is the `data` directory in `pwd`. Here is an example for preparing amz dataset for amz-DIEN:

```
mkdir -p evaluation/amz-DIEN/data
cp -r dataset/amz_book evaluation/amz-DIEN/data/ # the dataset is assumed to be already extracted
```

## Start Training

Enter the specific model dir and run the script (amz-DIEN for example):

```
cd evaluation/amz-DIEN
python train_deeprec.py
```

The end-to-end time and final AUC achieved would be printed after the training is finished.

