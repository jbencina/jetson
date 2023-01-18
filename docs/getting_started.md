# Getting Started
## Prepare Environment
### Get Nvidia Jetpack
> Note: Nvidia has discontinued support for the Jetson Nano beyond Jetpack SDK v4.6.3. If you try flashing the 5.x image, the device will not boot.

Download JetPack 4.6.3 SDK from https://developer.nvidia.com/jetpack-sdk-463. This will include:
- CUDA 10.2
- CUDNN 8.2.1
- Ubuntu 18.04

Flash SD card using balenaEtcher https://www.balena.io/etcher/. Boot up Jetson nano and follow on-screen prompts for network, keyboard, and user setup.

**Optional**: If you've installed a wifi module, apply this fix wifi low-power setting ([tutorial](https://github.com/robwaat/Tutorial/blob/master/Jetson%20Disable%20Wifi%20Power%20Management.md)). This addresses extreme latency over wifi SSH due to power mode. The linked guide will apply this fix on each boot cycle.

Update installed packages
```
sudo apt update && apt upgrade
```

## Prepare Python
>Note: The Tensorflow & PyTorch builds provided by NVIDIA are limited to Python `3.6` for the Jetson Nano. if you intend to use OpenCV with TensorFlow or PyTorch, you should build this using a `3.6` environment rather than `3.8`

Install Python 3.8
```bash
sudo apt install python3.8 python3.8-dev python3.8-venv

# Or via Python 3.6
# sudo apt install python3 python3-dev python3-venv
```

This will be accessed via `python3.8` command since `python3` is linked to 3.6.

## Build OpenCV
We'll need to build our own version of OpenCV since the version on PyPI does not have Gstreamer enabled. Gstreamer is needed to capture video on the Jetson.

Install gStreamer packages - these should already be installed with the SDK
```bash
sudo apt install gstreamer1.0* libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```
Create clean Python environment for build
```bash
python3.8 -m venv envs/opencv
source envs/opencv/bin/activate
pip install -U pip wheel
```
**Optional**: If building over SSH, it's possible that your connection drops and requires you to restart the build process. We can use tmux to create a persistent session which we can re-join if needed:
```bash
sudo apt install tmux

tmux # Starts new session
tmux attach # Joins last session
```

Clone repo and build. We use the opencv-python version as a base since it shortcuts many steps we'd have to do if building from the original.
> This step will take about 2 hours. Ensure your CPUs are running in performance mode with `sudo jetson_clocks`. Run these commands in a `tmux` session if performing over SSH
```python
git clone --recursive https://github.com/opencv/opencv-python.git
cd opencv-python
export CMAKE_ARGS="-DWITH_GSTREAMER=ON"
pip wheel . --verbose
```

Install the resulting .whl file in the environment of choice
```bash
pip install opencv-*.whl
```

## Fix Paths
```
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}:/home/jlee/bin
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

## Install Pytorch && Tensorflow
My experience is that the pre-built wheels are somewhat flaky and required adaptation not documented in any of the guides. You might be better off building TensorFlow and PyTorch from scratch for the best compatability as the existing wheels become more stale.

You can also leverage the Docker containers such as https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch

### Tensorflow
Steps based on official guide: https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

>Note: Due to `4.6.3` Jetpack limitation on Nano, the latest available Tensorflow version provided by NVIDIA is `2.7.0` with the `22.01` TensorFlow container. Additionally, the version built by NVIDIA only supports Python `3.6` https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html#tf-jetson-rel

Install system libraries
```bash
sudo apt install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
```

Create new Python environment for TensorFlow. Note this is is run as Python 3.6 so you may need to install `sudo apt install python3-venv`.
```bash
python3 -m venv venv/tensorflow
source venv/tensorflow/bin/activate
pip install -U pip testresources setuptools wheel
```

One of the libraries (xlocale.h) required by Numpy was changed which causes the build to fail because it no longer exists. To work around this, a symlink can be created:
```bash
sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
```

Install TensorFlow. This should be the `22.01` version per the NVIDIA compatibility table, but they only seem to provide `22.1`. Ensure you're using a Python `3.6` environment for this install.
```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow==2.7.0+nv22.1
```
I was getting Python core dump failures when attempting to import this version of TensorFlow, but found that adding this environment setting resolves the issue
```bash
export OPENBLAS_CORETYPE=ARMV8
```
We can validate the install by starting a Python instance and checking if there are reported GPUs
```python
>>> import tensorflow as tf
>>> tf.config.list_physical_devices('GPU')

[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## PyTorch
Steps based on official guide: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

Install system libraries
```bash
sudo apt install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev
```

Create new Python environment for TensorFlow. Note this is is run as Python 3.6 so you may need to install `sudo apt install python3-venv`.
```bash
python3 -m venv venv/tensorflow
source venv/tensorflow/bin/activate
pip install -U pip testresources setuptools wheel numpy
```

You'll also need to install the following packages before installing PyTorch
```bash
sudo apt-get install libopenblas-base libopenmpi-dev 
```

Install PyTorch from the pre-compiled wheel available in this thread for 1.10.0 https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048. The one provided in the NVIDIA docs did not work for me.
```bash
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl torch-1.10.0-cp36-cp36m-linux_aarch64.whl

pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

Validate that PyTorch is correctly using GPU
```python
>>> import torch
>>> torch.tensor([1.0, 2.0]).cuda()
tensor([1., 2.], device='cuda:0')
```

## Build PyTorch
The benefit of building PyTorch is that you can use more recent versions without
worrying about the exact versions of external libraries installed on the Jetson or
being stuck with Python 3.6. The downside is mostly time as the build can take several
hours.

Create a new Python environment for the PyTorch build. Here I use Python 3.8.
```bash
python3.8 -m venv envs/pytorch-build
source envs/pytorch-build/bin/activate
pip install -U pip wheel setuptools
```

Install some required system packages
```bash
sudo apt install cmake libopenblas-dev libopenmpi-dev 
```

Create a 2GB swap file following this guide
https://docs.rackspace.com/support/how-to/create-a-linux-swap-file/. The instructions
here are for 1GB, but I ran out of memory compiling PyTorch.

Clone PyTorch & install the required libraries
```bash
git clone --recursive --branch <version> http://github.com/pytorch/pytorch
cd pytorch
pip install -r requirements.txt
pip install scikit-build ninja numpy
```

You'll need to set the following environment variables in preparation for the build.
I lowered `MAX_JOBS` to 1, but the build time took very long. I did this because I
had run into memory issues. I think the 2GB swap file may have allowed for more
max jobs.

```bash
export USE_NCCL=0
export USE_DISTRIBUTED=0
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0
export TORCH_CUDA_ARCH_LIST="5.3"
export MAX_JOBS=1
export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
export PYTORCH_BUILD_VERSION=1.12.1
export PYTORCH_BUILD_NUMBER=1
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Run the build process. I suggest creating this in a tmux session in case the SSH
connection drops

```bash
python setup.py bdist_wheel
```

You'll have a file like `torch-1.12.1-cp38-cp38-linux_aarch64.whl` in your `/dist`
directory in the cloned PyTorch repo. You can now install this wheel in whatever
Python 3.8 environment you'd like

```bash
pip install torch-1.12.1-cp38-cp38-linux_aarch64.whl
```

We can test and ensure it works
```python
>>> import torch
>>> torch.rand(10).cuda()
tensor([1.7768e-04, 7.1322e-01, 7.0025e-01, 2.5537e-01, 9.5965e-01, 8.2763e-01,
        9.4872e-01, 4.0117e-01, 2.5573e-01, 1.3752e-01], device='cuda:0')
```
