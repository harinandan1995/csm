# Canonical Surface Mapping (CSM)

### Installation

To setup a conda environment run

`conda env create -f resources/env.yml`
> The packages will be updated as we progress through the project

To update an already existing environment run
 
 `conda env update -f resources/env.yml`

### Data

Download the annotations data

```wget https://syncandshare.lrz.de/dl/fi7mmNz9FxDGTNdSbyYDkmBS/cachedir.tar.gz & tar -xf cachedir.tar.gz```

Download the images for cub data

```wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz && tar -xf CUB_200_2011.tgz```