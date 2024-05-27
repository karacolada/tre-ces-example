# -*- coding: utf-8 -*-
"""
download_images

Script to retrieve images for the 2024 FathomNet out-of-sample challenge as part of FGVC 10. 

Assumes COCO formatted annotation file has been downloaded from http://www.kaggle.com/competitions/fathomnet-out-of-sample-detection
"""
# Authors: 
# Eric Orenstein (eorenstein@mbari.org)
# Lukas Picek (lukaspicek@gmail.com)


import os
import json
import logging
import argparse
import requests
from shutil import copyfileobj
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def download_img(args):
    """
    Download a single image.

    :param args: Tuple of (name, url, outdir)
    """
    name, url, outdir = args
    file_name = os.path.join(outdir, name)

    # Only download if the image does not exist in the outdir
    if not os.path.exists(file_name):
        resp = requests.get(url, stream=True)
        resp.raw.decode_content = True
        with open(file_name, 'wb') as f:
            copyfileobj(resp.raw, f)
        return 1  # Indicate that image was downloaded
    else:
        return 0  # Indicate that image already exists


def download_imgs(imgs, outdir=None):
    """
    Download images to an output dir

    :param imgs: list of tuples (name, url)
    :param outdir: desired directory [default to working directory]
    """

    # Set the out directory to default if not specified
    if not outdir:
        outdir = os.path.join(os.getcwd(), 'images')

    # Make the directory if it does not exist
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        logging.info(f"Created directory {outdir}")

    num_processes = cpu_count() * 2  # Use twice the number of CPU cores for multiprocessing
    pool = Pool(processes=num_processes)

    # Prepare arguments for multiprocessing
    args_list = [(name, url, outdir) for name, url in imgs]

    # Use tqdm for progress bar
    with tqdm(total=len(imgs)) as pbar:
        for _ in pool.imap_unordered(download_img, args_list):
            pbar.update(1)

    pool.close()
    pool.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download images from a COCO annotation file")
    parser.add_argument('dataset', type=str, help='Path to json COCO annotation file')
    parser.add_argument('--outpath', type=str, default=None, help='Path to desired output folder')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info(f'opening {args.dataset}')
    with open(args.dataset, 'r') as ff:
        dataset = json.load(ff)

    ims = dataset['images']

    logging.info(f'retrieving {len(ims)} images')

    ims = [(im['file_name'], im['coco_url']) for im in ims]

    # Download images
    download_imgs(ims, outdir=args.outpath)