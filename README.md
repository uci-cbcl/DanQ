README for DanQ
===============
DanQ is a hybrid convolutional and recurrent neural network model for predicting the function of DNA *de novo* from sequence. 

Please excuse how incomplete everything is. It's Winter break now, and it's hard for me to write a paper, spend quality family, catch up on Netflix, etc. all at the same time. But, it will be done!

Citing DanQ
===========
Quang, D. and Xie, X. ``DanQ: a hybrid convolutional and recurrent neural network for predicting the function of DNA sequences'', *Under review*, 2015.

INSTALL
=======

DanQ uses a lot of bleeding edge software packages, and very often these software packages are not backwards compatible when they are updated. Therefore, I have included the most recent version numbers of the software packages for the configuration that worked for me. For the record, I am using Ubuntu Linux 14.04 LTS with an NVIDIA Titan Z GPU.

Required
--------
* [Python] (https://www.python.org) (2.7.10). The easiest way to install Python and all of the necessary dependencies is to download and install [Anaconda] (https://www.continuum.io) (2.3.0). I listed the versions of Python and Anaconda I used, but the latest versions should be fine. If you're curious as to what packages in Anaconda are used, they are: [numpy] (http://www.numpy.org/) (1.10.1), [scipy] (http://www.scipy.org/) (0.16.0), and [h5py] (http://www.h5py.org) (2.5.0). 
* [Theano] (https://github.com/Theano/Theano) (latest). At the time I wrote this, Theano 0.7.0 is already included in Anaconda. However, it is missing some crucial helper functions. You need to git clone the latest bleeding edge version since there isn't a version number for it:

```
$ git clone git://github.com/Theano/Theano.git
$ cd Theano
$ python setup.py develop
```

* [keras] (https://github.com/fchollet/keras/releases/tag/0.2.0) (0.2.0). Deep learning package that uses Theano backend. I'm in the process of upgrading to version 0.3.0 with the Tensorflow backend.

* [seya] (https://github.com/EderSantana/seya) (???). I had to modify the source code of this package a little bit. You can try getting the latest version from Github, but for your convenience I've uploaded my copy of the package. You can install it as follows:

```
cd DanQ_seya
python setup.py install
``` 

I will likely improve DanQ soon and drop the dependency on seya.

Optional
--------
* [CUDA] (https://developer.nvidia.com/cuda-toolkit-65) (6.5). Theano can use either CPU or GPU, but using a GPU is almost entirely necessary for a network and dataset this large.

* [cuDNN] (https://developer.nvidia.com/cudnn) (2). Significantly speeds up convolution operations. 

USAGE
=====

If you have everything installed, you can train a model as follows:

```
$ python DanQ_train.py
```

On my system, each epoch took about 6 hours. Whenever the validation loss is reaches a new minimum at the end of a training epoch, the best weights are stored in [DanQ_bestmodel.hdf5] (https://cbcl.ics.uci.edu/public_data/DanQ/DanQ_bestmodel.hdf5). I've already uploaded the fully trained model in the hyperlink. You can see motif results, including visualizations and TOMTOM comparisons to known motifs, in the motifs/ folder. Likewise, you can also train a much larger model where about half of the motifs are initialized with JASPAR motifs:

```
$ python DanQ-JASPAR_train.py
```

Weights are saved to the fight [DanQ-JASPAR_bestmodel.hdf5] (https://cbcl.ics.uci.edu/public_data/DanQ/DanQ-JASPAR_bestmodel.hdf5) whenever the validation loss is lowered. Motif results for this model are also stored in the motifs/ folder.

To-Do
=====

* Predict on FASTA sequences
* Annotate genetic variation


