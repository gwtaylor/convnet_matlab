Simple Matlab demo: training a 2-layer convolutional net on the Small NORB dataset
Author: Graham Taylor

Originally written April 12, 2010
Updated and made public August 31, 2012

We provide 32x32 downsampled data for the Small NORB dataset.

http://dl.dropbox.com/u/13294233/smallnorb/smallnorb-5x01235x9x18x6x2x32x32-testing-dat-matlab-bicubic.mat
http://dl.dropbox.com/u/13294233/smallnorb/smallnorb-5x01235x9x18x6x2x96x96-testing-cat-matlab.mat
http://dl.dropbox.com/u/13294233/smallnorb/smallnorb-5x01235x9x18x6x2x96x96-testing-info-matlab.mat
http://dl.dropbox.com/u/13294233/smallnorb/smallnorb-5x46789x9x18x6x2x32x32-training-dat-matlab-bicubic.mat
http://dl.dropbox.com/u/13294233/smallnorb/smallnorb-5x46789x9x18x6x2x96x96-training-cat-matlab.mat
http://dl.dropbox.com/u/13294233/smallnorb/smallnorb-5x46789x9x18x6x2x96x96-training-info-matlab.mat

Original dataset: http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/

Note that we only train on image 1 of the stereo pair.

You will need to change the path defined at the top of smallnorb_makebatches.m to reflect the actual location of the NORB data.

e.g.
datasetpath = '~/Dropbox/Public/smallnorb';

The convolutional net in this example has the following architecture:

Data; connected to
Convolutional Layer 1; connected to
Subsampling Layer 1; connected to
Convolutional Layer 2; connected to
Subsampling Layer 2

The output of Subsampling L2 is vectorized and fully-connected to the
output layer which is a multinomial (i.e. 1 of "K")

Here, the outputs correspond to object categories (K=5). 

The data is connected to each map of Convolutional L1.  We build a
random connectivity map to determine which maps of Subsampling L1 are
connected to which maps of Convolutional L2.

We use Carl Rasmussen's "minimize" conjugate gradient code to train the
network. Therefore, we define a function which returns:
1) The value (at the current setting of the parameters) of the error 
function to be minimized; and 
2) The gradient of the error function with respect to the parameters

The benefit of this is that we can use Carl's "checkgrad" function to
check the gradients of the cross-entropy error with respect to the parameters of the convnet using the method of finite differences.

Note that for the first 6 epochs, only the topmost (fully-connected)
weights are updated while the other parameters are held constant.

Running the demo

There are three entry points:

1) norbbackpropc2.m : "Main script"; set up data and train network

It has been written to be understandable and therefore is not
optimized (i.e. it loops over feature maps and cases)
We recommend going through this code first, since everything is explicit

2) norbbackprobc2_fast.m : Fast version of the above; it requires ipp_mt_conv2
(IPP-optimized 2-D convolution)

We've linked to a compiled mex for 64-bit linux which should work on
most modern Linux systems.  Otherwise you will need to grab the source
from Rob Fergus' webpage and attempt to build it for other
architectures.

http://www.cs.nyu.edu/~fergus/code/ipp_mt_conv2.cpp

3) demo_checkgrad.m : Shows how to use the method of finite
differences to debug your backprop code

We find checkgrad to be an extremely helpful tool; not just for
convolutional nets!

The bulk of the code is in:

convnet_forward2.m : Forward pass only (does not include the topmost layer)
convnet_probsm.m : Topmost (fully-connected) layer
fn_2layer_convnet_classify.m : Backprop (written for minimize, checkgrad)

And the optimized counterparts :
convnet_forward2_fast.m, fn_2layer_convnet_classify_fast.m