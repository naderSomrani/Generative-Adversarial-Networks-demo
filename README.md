<img src='imgs/horse2zebra.gif' align="right" width=384>

<br><br><br>

# CycleGAN and pix2pix in PyTorch

This is PyTorch implementations for both unpaired and paired image-to-image translation.

The code was written by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung), and supported by [Tongzhou Wang](https://ssnl.github.io/).

Original repository by [junyanz](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="800"/>

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" width="800px"/>

<img src='imgs/edges2cats.jpg' width="400px"/>

## [CycleGAN notebook](CycleGAN.ipynb) | [Pix2Pix notebook](pix2pix.ipynb)


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

- Install [PyTorch](http://pytorch.org and) 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, we provide a installation script `./scripts/conda_deps.sh`. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.

### CycleGAN train/test
- Download a CycleGAN dataset (e.g. maps):
```bash
bash ./datasets/download_cyclegan_dataset.sh maps
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Train a model:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`.
- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
- The test results will be saved to a html file here: `./results/maps_cyclegan/latest_test/index.html`.

### pix2pix train/test
- Download a pix2pix dataset (e.g.[facades](http://cmp.felk.cvut.cz/~tylecr1/facade/)):
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Train a model:
```bash
#!./scripts/train_pix2pix.sh
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
To see more intermediate results, check out  `./checkpoints/facades_pix2pix/web/index.html`.

- Test the model (`bash ./scripts/test_pix2pix.sh`):
```bash
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
- The test results will be saved to a html file here: `./results/facades_pix2pix/test_latest/index.html`. You can find more scripts at `scripts` directory.
- To train and test pix2pix-based colorization models, please add `--model colorization` and `--dataset_mode colorization`. See our training [tips](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#notes-on-colorization) for more details.

### Apply a pre-trained model (CycleGAN)
- You can download a pretrained model (e.g. horse2zebra) with the following script:
```bash
bash ./scripts/download_cyclegan_model.sh horse2zebra
```
- The pretrained model is saved at `./checkpoints/{name}_pretrained/latest_net_G.pth`. Check [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_cyclegan_model.sh#L3) for all the available CycleGAN models.
- To test the model, you also need to download the  horse2zebra dataset:
```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

- Then generate the results using
```bash
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
```
- The option `--model test` is used for generating results of CycleGAN only for one side. This option will automatically set `--dataset_mode single`, which only loads the images from one set. On the contrary, using `--model cycle_gan` requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at `./results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.

- For your own experiments, you might want to specify `--netG`, `--norm`, `--no_dropout` to match the generator architecture of the trained model.

### Apply a pre-trained model (pix2pix)
Download a pre-trained model with `./scripts/download_pix2pix_model.sh`.

- Check [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_pix2pix_model.sh#L3) for all the available pix2pix models. For example, if you would like to download label2photo model on the Facades dataset,
```bash
bash ./scripts/download_pix2pix_model.sh facades_label2photo
```
- Download the pix2pix facades datasets:
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- Then generate the results using
```bash
python test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained
```
- Note that we specified `--direction BtoA` as Facades dataset's A to B direction is photos to labels.

- If you would like to apply a pre-trained model to a collection of input images (rather than image pairs), please use `--model test` option. See `./scripts/test_single.sh` for how to apply a model to Facade label maps (stored in the directory `facades/testB`).

- See a list of currently available models at `./scripts/download_pix2pix_model.sh`

## [Datasets](docs/datasets.md)
Download pix2pix/CycleGAN datasets and create your own datasets.

## Custom Model and Dataset
If you plan to implement custom models and dataset for your new applications, we provide a dataset [template](data/template_dataset.py) and a model [template](models/template_model.py) as a starting point.

## Overview of Code Structure
To help users better understand and use our codebase, we briefly overview the functionality and implementation of each package and each module. Please see the documentation in each file for more details. If you have questions, you may find useful information in [training/test tips](tips.md) and [frequently asked questions](qa.md).

[train.py](../train.py) is a general-purpose training script. It works for various models (with option `--model`: e.g., `pix2pix`, `cyclegan`, `colorization`) and different datasets (with option `--dataset_mode`: e.g., `aligned`, `unaligned`, `single`, `colorization`). See the main [README](.../README.md) and [training/test  tips](tips.md) for more details.

[test.py](../test.py) is a general-purpose test script. Once you have trained your model with `train.py`, you can use this script to test the model. It will load a saved model from `--checkpoints_dir` and save the results to `--results_dir`. See the main [README](.../README.md) and [training/test tips](tips.md) for more details.


[data](../data) directory contains all the modules related to data loading and preprocessing. To add a custom dataset class called `dummy`, you need to add a file called `dummy_dataset.py` and define a subclass `DummyDataset` inherited from `BaseDataset`. You need to implement four functions: `__init__` (initialize the class, you need to first call `BaseDataset.__init__(self, opt)`), `__len__` (return the size of dataset), `__getitem__`　(get a data point), and optionally `modify_commandline_options` (add dataset-specific options and set default options). Now you can use the dataset class by specifying flag `--dataset_mode dummy`. See our template dataset [class](../data/template_dataset.py) for an example.   Below we explain each file in details.

* [\_\_init\_\_.py](../data/__init__.py) implements the interface between this package and training and test scripts. `train.py` and `test.py` call `from data import create_dataset` and `dataset = create_dataset(opt)` to create a dataset given the option `opt`.
* [base_dataset.py](../data/base_dataset.py) implements an abstract base class ([ABC](https://docs.python.org/3/library/abc.html)) for datasets. It also includes common transformation functions (e.g., `get_transform`, `__scale_width`), which can be later used in subclasses.
* [image_folder.py](../data/image_folder.py) implements an image folder class. We modify the official PyTorch image folder [code](https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py) so that this class can load images from both the current directory and its subdirectories.
* [template_dataset.py](../data/template_dataset.py) provides a dataset template with detailed documentation. Check out this file if you plan to implement your own dataset.
* [aligned_dataset.py](../data/aligned_dataset.py) includes a dataset class that can load image pairs. It assumes a single image directory `/path/to/data/train`, which contains image pairs in the form of {A,B}. See [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#prepare-your-own-datasets-for-pix2pix) on how to prepare aligned datasets. During test time, you need to prepare a directory `/path/to/data/test` as test data.
* [unaligned_dataset.py](../data/unaligned_dataset.py) includes a dataset class that can load unaligned/unpaired datasets. It assumes that two directories to host training images from domain A `/path/to/data/trainA` and from domain B `/path/to/data/trainB` respectively. Then you can train the model with the dataset flag `--dataroot /path/to/data`. Similarly, you need to prepare two directories `/path/to/data/testA` and `/path/to/data/testB` during test time.
* [single_dataset.py](../data/single_dataset.py) includes a dataset class that can load a set of single images specified by the path `--dataroot /path/to/data`. It can be used for generating CycleGAN results only for one side with the model option `-model test`.
* [colorization_dataset.py](../data/colorization_dataset.py) implements a dataset class that can load a set of nature images in RGB, and convert RGB format into (L, ab) pairs in [Lab](https://en.wikipedia.org/wiki/CIELAB_color_space) color space. It is required by pix2pix-based colorization model (`--model colorization`).


[models](../models) directory contains modules related to objective functions, optimizations, and network architectures. To add a custom model class called `dummy`, you need to add a file called `dummy_model.py` and define a subclass `DummyModel` inherited from `BaseModel`. You need to implement four functions: `__init__` (initialize the class; you need to first call `BaseModel.__init__(self, opt)`), `set_input` (unpack data from dataset and apply preprocessing), `forward` (generate intermediate results), `optimize_parameters` (calculate loss, gradients, and update network weights), and optionally `modify_commandline_options` (add model-specific options and set default options). Now you can use the model class by specifying flag `--model dummy`. See our template model [class](../models/template_model.py) for an example.  Below we explain each file in details.

* [\_\_init\_\_.py](../models/__init__.py)  implements the interface between this package and training and test scripts.  `train.py` and `test.py` call `from models import create_model` and `model = create_model(opt)` to create a model given the option `opt`. You also need to call `model.setup(opt)` to properly initialize the model.
* [base_model.py](../models/base_model.py) implements an abstract base class ([ABC](https://docs.python.org/3/library/abc.html)) for models. It also includes commonly used helper functions (e.g., `setup`, `test`, `update_learning_rate`, `save_networks`, `load_networks`), which can be later used in subclasses.
* [template_model.py](../models/template_model.py) provides a model template with detailed documentation. Check out this file if you plan to implement your own model.
* [pix2pix_model.py](../models/pix2pix_model.py) implements the pix2pix [model](https://phillipi.github.io/pix2pix/), for learning a mapping from input images to output images given paired data. The model training requires `--dataset_mode aligned` dataset. By default, it uses a `--netG unet256` [U-Net](https://arxiv.org/pdf/1505.04597.pdf) generator, a `--netD basic` discriminator (PatchGAN), and  a `--gan_mode vanilla` GAN loss (standard cross-entropy objective).
* [colorization_model.py](../models/colorization_model.py) implements a subclass of `Pix2PixModel` for image colorization (black & white image to colorful image). The model training requires `-dataset_model colorization` dataset. It trains a pix2pix model, mapping from L channel to ab channels in [Lab](https://en.wikipedia.org/wiki/CIELAB_color_space) color space. By default, the `colorization` dataset will automatically set `--input_nc 1` and `--output_nc 2`.
* [cycle_gan_model.py](../models/cycle_gan_model.py) implements the CycleGAN [model](https://junyanz.github.io/CycleGAN/), for learning image-to-image translation  without paired data.  The model training requires `--dataset_mode unaligned` dataset. By default, it uses a `--netG resnet_9blocks` ResNet generator, a `--netD basic` discriminator (PatchGAN  introduced by pix2pix), and a least-square GANs [objective](https://arxiv.org/abs/1611.04076) (`--gan_mode lsgan`).
* [networks.py](../models/networks.py) module implements network architectures (both generators and discriminators), as well as normalization layers, initialization methods, optimization scheduler (i.e., learning rate policy), and GAN objective function (`vanilla`, `lsgan`, `wgangp`).
* [test_model.py](../models/test_model.py) implements a model that can be used to generate CycleGAN results for only one direction. This model will automatically set `--dataset_mode single`, which only loads the images from one set. See the test [instruction](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#apply-a-pre-trained-model-cyclegan) for more details.

[options](../options) directory includes our option modules: training options, test options, and basic options (used in both training and test). `TrainOptions` and `TestOptions` are both subclasses of `BaseOptions`. They will reuse the options defined in `BaseOptions`.
* [\_\_init\_\_.py](../options/__init__.py)  is required to make Python treat the directory `options` as containing packages,
* [base_options.py](../options/base_options.py) includes options that are used in both training and test. It also implements a few helper functions such as parsing, printing, and saving the options. It also gathers additional options defined in `modify_commandline_options` functions in both dataset class and model class.
* [train_options.py](../options/train_options.py) includes options that are only used during training time.
* [test_options.py](../options/test_options.py) includes options that are only used during test time.


[util](../util) directory includes a miscellaneous collection of useful helper functions.
  * [\_\_init\_\_.py](../util/__init__.py) is required to make Python treat the directory `util` as containing packages,
  * [get_data.py](../util/get_data.py) provides a Python script for downloading CycleGAN and pix2pix datasets.  Alternatively, You can also use bash scripts such as [download_pix2pix_model.sh](../scripts/download_pix2pix_model.sh) and [download_cyclegan_model.sh](../scripts/download_cyclegan_model.sh).
  * [html.py](../util/html.py) implements a module that saves images into a single HTML file.  It consists of functions such as `add_header` (add a text header to the HTML file), `add_images` (add a row of images to the HTML file), `save` (save the HTML to the disk). It is based on Python library `dominate`, a Python library for creating and manipulating HTML documents using a DOM API.
  * [image_pool.py](../util/image_pool.py) implements an image buffer that stores previously generated images. This buffer enables us to update discriminators using a history of generated images rather than the ones produced by the latest generators. The original idea was discussed in this [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Shrivastava_Learning_From_Simulated_CVPR_2017_paper.pdf). The size of the buffer is controlled by the flag `--pool_size`.
  * [visualizer.py](../util/visualizer.py) includes several functions that can display/save images and print/save logging information. It uses a Python library `visdom` for display and a Python library `dominate` (wrapped in `HTML`) for creating HTML files with images.
  * [util.py](../util/util.py) consists of simple helper functions such as `tensor2im` (convert a tensor array to a numpy image array), `diagnose_network` (calculate and print the mean of average absolute value of gradients), and `mkdirs` (create multiple directories).

