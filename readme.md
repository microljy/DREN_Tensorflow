# DREN_TensorFlow
Paper Link:[Deep Rotation Equivirant Network](https://arxiv.org/abs/1705.08623)
[DREN Caffe version](https://github.com/microljy/DREN)
## Usage
### installation TensorFlow
Note that this code work in TensorFlow_1.0.0. TensorFlow 0.12.0 is not supported.

	export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0-cp27-none-linux_x86_64.whl

	pip install --upgrade $TF_BINARY_URL
### rmnist
#### data
Down load data, and transform it into npy.

	cd DREN_ROOT/data/rmnist
	sh get_data.sh
	matlab < data_preprocess.m
	python generate_npy.py
	rm data.mat

#### train model

	python train_mnist.py

#### results
This is the results of this implementation. It can be boosted through fine tuning training params.

|model|error|
|-
|Z2CNN|5.94%|
|DREN_Z2CNN|4.61%|
|DREN_Z2CNN_x4|2.86%|

## Discussion
DREN can be used to boost the performance of classification of aerial image, microscope images, CT images and so on.

## Citation
Please cite DREN in your publications if it helps your research:

	@article{Li2017Deep,
	  title={Deep Rotation Equivariant Network},
	  Journal = {arXiv preprint arXiv:1705.08623},
	  author={Li, Junying and Yang, Zichen and Liu, Haifeng and Cai, Deng},
	  year={2017},
	}