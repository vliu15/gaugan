# GauGAN
Unofficial Pytorch implementation of GauGAN, from [Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) (Park et al. 2019). Implementation for [Generative Adversarial Networks (GANs) Specialization](https://www.coursera.org/specializations/generative-adversarial-networks-gans) course material.

## Usage
1. Download the [Cityscapes dataset](https://www.cityscapes-dataset.com/), unzip the `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip` folders and move them to `data` directory.
2. All Python requirements can be found in `requirements.txt`. Support for Python>=3.7.
3. Default config for can be found in `config.yml`. All defaults are as per the configurations described in the original paper and code.

### Training
By default, all checkpoints will be stored in `logs/YYYY-MM-DD_hh_mm_ss`, but this can be edited via the `train.log_dir` field in the config files.

1. To train GauGAN, run `python train.py`.

### Inference
1. Edit the `resume_checkpoint` field `config.yml` to reflect the desired checkpoint from training and run `python infer.py --encode`. The `--encode` flag generates Gaussian statistics from the input image via the encoder. If not specified, noise will be sampled from a standard Gaussian.
> You can edit the number of test images to show with the flag `--n_show`. Defaults to 5.
