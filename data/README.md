# FathomNet2024 Kaggle competition

See https://www.kaggle.com/competitions/fathomnet2024

- `eval` took 7 minutes to download, and stands at 5.4G (2686 images)
- `train` took 18 minutes to download, and stands at 17G (8058 images)

## Dataset Description

The training and test images for the competition were all collected in the Monterey Bay Area between the surface and 1300 meters depth. The images contain bounding box annotations of 18 morphotaxonomic (i.e. semantic) categories of bottom dwelling animals. The training and test data come from the same 0-1300 meter range. Training images only have annotations for the 18 base supercategories. Test images contain those 18 categories and two new ones. Critically, there might be unannotated examples of the new classes in the training data. This is a common scenario in ocean research: many scientists annotate images with a particular target organism in mind and ignore everything else.

The competition goal is to label and count the instances of the known supercategories in the test images while also counting unknown objects. Some test images may not contain only known categories, only unknown objects, or a combination of both.
Data Format

The training dataset is provided with bounding box annotations object detection adhering to the COCO Object Detection standard. Every training image contains at least one annotation corresponding to a supercategory from 1 to 18.

The 18 semantic supercategories:

['Anemone', 'Barnacle', 'Black coral', 'Crab', 'Eel', 'Feather star', 'Fish', 'Gastropod', 'Glass sponge', 'Sea cucumber', 'Sea fan', 'Sea pen', 'Sea spider', 'Sea star', 'Squat lobster', 'Stony coral', 'Urchin', 'Worm']

We are not able to provide images as a single downloadable archive due to FathomNet's Terms of Use. Images should be downloaded using the indicated coco_url field in each of the annotation files. Participants can either use the provided download_images.py python script or write their own.
Files

    train.json - the training images, annotations, and categories in COCO formatted json
    eval.json - the evaluation images in COCO formatted json
    sample_submission.csv - a sample submission file in the correct format
    download_images.py - python script to download imagery from FathomNet

## 📜 Terms of Use

By downloading and using this dataset you are agreeing to FathomNet's data use policy. In particular:

    The annotations are licensed under a Creative Commons Attribution-No Derivatives 4.0 International License.
    The images are licensed under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0 International License. Notwithstanding the foregoing, all of the images may be used for training and development of machine learning algorithms for commercial, academic, and government purposes.
    Images and annotations are provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.
    Please acknowledge FathomNet and MBARI when using images for publication or communication purposes regarding this competition. For all other uses of the images, users should contact the original copyright holder.

For more details please see the FathomNet Data Use Policy and Terms of Use.

## 🚀 Download

Images are made available for download by the unique URLs in the COCO-formatted object detection annotation files. The download script can be run from the command line. To install the requirements, create a virtual environment running python 3.9:

```sh
$ conda create -n fgvc_test python=3.9 pip
$ conda activate fgvc_test
$ pip install -r requirements.txt
```

To download the images, run the script from the command line:

```sh
$ python download_images.py [PATH/TO/DATASET.json] --outpath [PATH/TO/IMAGE/DIRECTORY]
```

If no `--outpath` directory is specified, the script by default downloads images to the directory the command is executed from. 