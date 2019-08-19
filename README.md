# CNN_Source_of_Light
This is a project, which consists of convolutional neural networks and finding the source of light from the 2D image.

## Overview
The code in this project is used for Master of Science thesis of Matias Ijäs.

The project is still in state: ***In progress***

## Usage
The software can be used as it is for non-profitable use. The creator of the software does not take any responsibility for possible misuses done with the software or bugs in the software.

It is highly recommended to ask permission to use code: matias.ijas2@gmail.com

### Reference info
If the code is used in some scientific study, please refer to it as:
**In text:**
> (Ijäs, 2019)

**In reference list (Harvard style):**
> Ijäs, M. 2019, *CNN Source of Light (source code)*, Github, Available: https://github.com/NemoHostem/CNN_Source_of_Light

## Requirements
Mandatory and optional softwares and Python packages are listed below.

### Software environment
- [Python 3.7](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)

### Python packages
- [Keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](https://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [imageio](https://pypi.org/project/imageio/)

### Optional
- [Spyder](https://www.spyder-ide.org/)
- [Face3D](https://github.com/YadiraF/face3d)

## Phases / Updates
Sprints of 2-3 weeks were used during the period of development phase. These subsection are only about development of code, excluding theoretical parts that are studied and written in M.Sc thesis.

### Sprint 1
- Created exif-extractor for possible gathering of image GPS data
- Created a working Face3D model used for creating training and testing data for CNN. (not pushed)
- Created color_space_checker to analyze RGB color format intensities and mask image

### Sprint 2
- Began to implement (classification) CNN for the problem
- Created compute_angles.py to compute real angles for training and test sets.
- 3D_rotator.js is used to visualize and understand light source in a 3D environment.

### Sprint 3
- Changed the CNN to regression, as the classification had issues with single values, which had high probability.
- Improved the CNN structure and made minor optimizations, such as changed input size to 64x64x3.

### Sprint 4
- Created a filtered dataset, which masks images with color intensities
- Overall testing different saved models
- Created a visualization part for testing, which visualizes ground truth and estimated value with image

### Sprint 5
- Created a few grayscale models and improved performance a bit
- Created a test_CNN.py file for large-scale testing
- Created read_pgm_files.py for making Yale (.pgm) dataset readable

### Sprint 6
- Commented most important code blocks and removed redundant code
- Created a predict tool for only predicting estimated angle of given image
- Implemented a face detection tool into my environment

