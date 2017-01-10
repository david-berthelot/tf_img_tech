# tf_img_tech

This contains various machine learning experiments in tensorflow such Generative Adversarial Networks (GANs).

## Setup code

This code is written in Python3, I typically use virtualenv to have a local installation of the required packages.

    virtualenv -p python3 --system-site-packages env
    . env/bin/activate
    pip install -r requirements.txt

Install TensorFlow in virtualenv:

Latest instructions: https://www.tensorflow.org/get_started/os_setup

(Note: no need for *sudo* when installing in virtualenv)


## Setup datasets

Go to Train/celeba and follow the README instructions.

## Train

    CUDA_VISIBLE_DEVICES=0 python -i tf_gan.py --opts=incr=1,carry_decay=5,batch=16,epoch=16,n=64,state=64,scales=4,hier=1,W=64,diversity=10.,history=1

Notes:
- By default the data is kept on disk, if your disk is slow and you have a lot of memory add the option in_memory=1 to --opts
- Preview folder: History will save intermediate training images so you can visualize how convergence affects the images. Every 5 iteration a preview image is generated.
- Models folder: The final model is saved there.

## Models

There are several models available in tf_gan.py, just uncomment the one you want to try out.
