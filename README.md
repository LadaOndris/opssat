# AI on the edge: "the OPS-SAT case" - Starter kit
<!--
*** Based on https://github.com/othneildrew/Best-README-Template
-->



<!-- PROJECT LOGO -->
<br />
<p align="center">
    Scripts to clean and prepare the OPSSAT competition dataset.
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#directory-content">Directory content</a></li>
    <li><a href="#getting-started">Getting started</a></li>
    <li><a href="#competition-data">Competition data</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This is the starting kit for the competition: `AI on the edge: "the OPS-SAT case" - Starter kit`. 

## Directory content

The `competition_toolkit` includes:

* `efficientnet_lite.py`, this is a slightly edited version of the [Keras](https://keras.io/api/models/model/)-based implementation of the `EfficientNet-lite-0` deep neural network included in     [efficientnet-lite-keras](https://github.com/sebastian-sz/efficientnet-lite-keras). This is the Deep Neural Network model that **shall** be used for the competition. **Submissions exploiting different architecture will be discarded**. 
* `starter_kit_notebook.ipynb`: this notebook shows how to load the competition dataset, instatiate and train an `EfficientNet-lite-0` model, included in `efficientnet_lite.py` and evaluate it by using the metrics described in [here](https://kelvins.esa.int/opssat/scoring/).
* `serverside_evaluation.ipynb`: this script shows the pipeline used to evaluate a submission. Such a pipeline includes, `HDF5` weights loading and model checking, conversion from `TF (Keras)` model to `tflite` by using `float16` quantization format (please, refer to [post-training float16 - quantization](https://www.tensorflow.org/lite/performance/post_training_float16_quant)) and score calculation by using the metrics described [here](https://kelvins.esa.int/opssat/scoring/).
* `pytorch_to_tf_conversion.ipynb`: this notebook provides a utility to converts `EfficientNet-lite-0` models trained in [PyTorch](https://pytorch.org/) by using the [efficientnet-lite-pytorch](https://pypi.org/project/efficientnet-lite-pytorch/) and [efficientnet_lite0_pytorch_model](https://pypi.org/project/efficientnet-lite0-pytorch-model/) python packages to the `TF (Keras)` model included in `efficientnet_lite.py`.
* `model_pytorch_baseline.pth`: This is an example of a valid [PyTorch](https://pytorch.org/) model (same as the competition baseline) which is used to showcase its translation to a valid submission file.

## Getting started

Clone this repository: 
```
git clone https://gitlab.com/EuropeanSpaceAgency/the_opssat_case_starter_kit.git
cd the_opssat_case_starter_kit
```

In our servers, where the submissions are evaluated, the following version of core packages are running:

* `python` **3.9**
* `tensorflow` **2.7.0**
* `numpy` **1.21.1**
* `sciki_learn` **1.0.2**
* `cudatoolkit` **11.2** 
* `cudnn` **8.1.0**

To create a similar environment on your machine, assuming you are in an updated conda base environment, you can run: 

```
 conda create --name opssat python=3.9 numpy jupyter scikit_learn
 conda activate opssat
 python3 -m pip install tensorflow==2.7
 ```

If you have GPUs (not strictly needed fot he notebooks, but highly useful for the competition) you will also need to install the correct versions of `cudatoolkit` and `cudnn` before the tensorflow installation.

Note that we pinned the tensorflow version and python version: the notebooks in this starter kit may break if different versions are used, since the tensorflow/keras API is not stable.

 Eventually, to test the correct installation of `tensorflow`, you can run:

 ```
 python
 import tensorflow
 ```

 If you get a `PROTOBUF` error, you can fix it by running: 

 ```
 conda install protobuf=3.20.1
 ```

If you want to use the `pytorch_to_tf_conversion.ipynb` notebook to convert a model trained on `Pytorch` and based on [efficientnet-lite-pytorch](https://pypi.org/project/efficientnet-lite-pytorch/) and [efficientnet_lite0_pytorch_model](https://pypi.org/project/efficientnet-lite0-pytorch-model/), you need to run the following commands to install `efficientnet_lite0_pytorch_model` and `efficientnet_lite0_pytorch_model`:

```
pip install efficientnet_lite_pytorch
pip install efficientnet_lite0_pytorch_model
```
	
## Competition data

The competition dataset can be downloaded from the following `Zenodo` [link](https://zenodo.org/record/6524750). Consecutively to the download, proceed as follows:

1. Create a directory `ops_sat_competiton_official`.
2. Unzip `ops_sat_competiton_official.zip` in the `ops_sat_competiton_official`. 

Once unzipped, the `ops_sat_competiton_official` will contain:

 * a subdirectory for each of the 8 competition class: `Agricultural`, `Cloud`, `Mountain`, `Natural`,`River`, `Sea_ice`, `Snow`, and `Water`. Each of this subdirectory includes 10 200x200x3-patches of that class (labeled traing data).

 * a subdirectory `images` containing 26 2048x1944x3 or 2048x1942x3 raw OPS-SAT satellite images. The competitors can dispose of these images to implement a training strategy to improve their models.
 **N.B.**: `images` **DOES NOT** contain data of one of the competition class. Therefore, to run the notebooks correctly, the directory `images` **shall be removed** from the `ops_sat_competiton_official` directory to run the previous notebooks correctly. 
