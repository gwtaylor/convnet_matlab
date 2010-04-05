Demo: training a 2-layer convolutional net on NORB 

The norb data has been downsampled to 32x32 pixels. 
Note that we only train on image 1 of the stereo pair.
You may need to change this line:

datasetpath = '/misc/FergusGroup/gwtaylor/smallnorb';

in smallnorb_makebatches.m to reflect the location to the RAID.

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

I use Carl Rasmussen's "minimize" conjugate gradient code to train the
network. Therefore, I define a function which returns:
1) The value (at the current setting of the parameters) of the error 
function to be minimized; and 
2) The gradient of the error function with respect to the parameters

The benefit of this is that I can use Carl's "checkgrad" function to
check my gradients using the method of finite differences.

Note that for the first 6 epochs, only the topmost (fully-connected)
weights are updated while the other parameters are held constant.

Running the demo

There are three entry points:

1) norbbackpropc2.m : "Main script"; set up data and train network

It has been written to be understandable and therefore is not
optimized (i.e. it loops over feature maps and cases)
I would recommend going through this code first, since everything is explicit

2) norbbackprobc2f.m : Fast version of the above; it requires ipp_mt_conv2
(IPP-optimized 2-D convolution)

I've included the compiled mex for 64-bit linux which should work on
all of the intel server machines If you're feeling adventurous, you
can grab the source from Rob Fergus' webpage and attempt to build it
for other architectures

3) demo_checkgrad.m : Shows how to use the method of finite
differences to debug your backprop code

I find checkgrad to be an extremely helpful tool; not just for
convolutional nets.

The bulk of the code is in:

convnet_forward2.m : Forward pass only (does not include the topmost layer)
convnet_probsm.m : Topmost (fully-connected) layer
CG_SMALLNORB_CLASSIFY_C2.m : Backprop (written for minimize, checkgrad)

And the optimized counterparts :
convnet_forward2f.m, CG_SMALLNORB_CLASSIFY_C2f.m
