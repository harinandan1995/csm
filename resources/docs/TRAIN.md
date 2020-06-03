## Training
Data is available for the following categories
For the categories from imagenet the sysnetid is also provided

* bird - CUB
* horse - imagenet (n02381460)
* zebra - imagenet (n02391049)
* bear - imagenet (n02131653)
* sheep - imagenet (n10588074)
* cow - imagenet (n01887787)

### Data
#### Annotations
* Download the annotations for all the categories
```
wget https://syncandshare.lrz.de/dl/fi7mmNz9FxDGTNdSbyYDkmBS/cachedir.tar.gz & tar -xf cachedir.tar.gz
```
#### CUB Data
* Download the images for bird category
```
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz && tar -xf CUB_200_2011.tgz
```
#### Imagenet Data
* Download the images for the categories from imagenet [here](http://image-net.org/download)

#### Pretrained model

Coming soon ......

### Training
* To start the training without modifying the data paths follow the directory structure
```
* csm
    * config/   
    * datasets/
        * cachedir/
        * CUB_200_2011/
        * IMAGNET #TODO - I am not sure about the directory name will update once we get access
    * resources/
    * src/
    * ...
    * ...
    * run.py
```

#### Bird
```python 
python run.py --config config/bird_train.yml --device cuda:0 train
``` 
> Use device cpu if you want to use the CPU

**NOT READY - WILL BE READY ONCE WE GET IMAGENET DATA ACCESS**
#### Imagenet
```python 
python run.py --config config/imnet_train.yml --device cuda:0 train
``` 
> Use device cpu if you want to use the CPU

#### Config

Most of the parameters required for the training are passed via a config file

For more details on the parameters that are used check the config files in [config](../../config)

##### Command line update
You can update few config parameters via command line as well. 

For example to change the batch size to 32 and learning rate to 0.005 use the following command
```python 
python run.py --config config/bird_train.yml --device cuda:0 train --train.batch_size 32 --train.optim.lr 0.005
```  
Same level of nesting as in the config file are used in command line updates (eg. train.optim.lr)
> Run ``python run.py train --help`` for more information
 