# Celeba dataset

Information about this dataset can be found here: 
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## Download

1. Download the attributes file *list_attr_celeba.txt* from: https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs
2. Download the partition between train and eval *list_eval_partition.txt* from: https://drive.google.com/drive/folders/0B7EVK8r0v71pdjI3dmwtNm5jRkE
3. Download the celeba images files *img_align_celeba.zip* (1.3GB) from: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg

## Setup

Unzip the images file:

    unzip img_align_celeba.zip
    
Build the dataset:

    . env/bin/activate  # If you didn't already do so
    python dataset.py --opts=celeba=1
