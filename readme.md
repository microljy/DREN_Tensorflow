# DREN_TensorFlow
Paper Link:[Deep Rotation Equivirant Network](https://arxiv.org/abs/1705.08623)
[DREN Caffe version](https://github.com/microljy/DREN)
## Usage

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