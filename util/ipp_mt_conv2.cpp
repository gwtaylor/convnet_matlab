// MEX wrapper for 2-D image convolution using Intel's Integrated ...
// Performance Primitive (IPP) Libraries and multi-threading
// Written by Rob Fergus (fergus@cs.nyu.edu) 11/14/07.
//
// Usage:
//
// out = ipp_mt_conv2( image , kernel , mode );
//
// Inputs:
//   1. image - The stack of images you want to convolve: n x m x c ...
//    matrix of single precision floating point numbers.
//   2. kernel - i x j matrix of single precision floating point numbers.
//   3. mode -  string: either "full" or "valid". Unless string is ...
//    "full", code will default to "valid". Note that this code does ...
//    NOT support the "same" option that Matlab's conv2 does. For reasons
//     that aren't clear to me, it is much faster (than you'd expect) to use 
//     the 'valid' option, rather than the 'full' one.
//
// Output:
//   1. out - Convolved stack of images. If in valid mode, this will be (n-i+1) x (m-j+1) x c  matrix of single ...
//  precision floating point numbers. If in full mode, this will be  (n+i-1) x (m+j-1) x c  matrix of single ...
//  precision floating point numbers. Each of the c images will have ...
//  been independently convolved by the kernel.
//
//
// Requirements:
//   1. A system with one or more multi-core Intel 64-bit CPU's
//   2. Up to date installations of:
//       - Intel 64-bit C compiler (tested on version 10.0.023)
//       - Intel Integrated Performance Primitive (IPP) libraries (tested on version 5.3)
//   3. Matlab - to actually use the MEX file (tested on 7.5.0.338 (R2007b))
//
//
// Points to note:
//
// 1. These environment variables that need to be set in bash before running the mex file:
//    export PATH="/opt/intel/cce/10.0.023/bin:${PATH}";
//    export LD_LIBRARY_PATH="/opt/intel/cce/10.0.023/lib:${LD_LIBRARY_PATH}";
// 
// 2.  The IPP libraries will automatically swap to using Fourier domain ...
//    multiplication once the size of the kernel is above 15x15 or so. 
//
// 3. I've no idea why the valid convolution is so much faster than the full one. 
//
// 4. Typical speedup when running on stacks of images (using same machine as 3.) is at least 5x or so. 
//    On a machine with two dualcore Xeons, the relative performance is (using 'valid'):
//
//     a. Large grayscale image - 2000x2000 pixels, kernel size = 5x5
//     Results --- Matlab (conv2): 0.68 secs. IPP: 0.11 secs. Speedup: 6.2
//
//     b. Large grayscale image - 2000x2000 pixels, kernel size = 30x30
//     Results --- Matlab (fft2): 1.28 secs. IPP: 0.29 secs. Speedup: 4.5
//
//     c. Large color image - 2000x2000x3 pixels, kernel size = 5x5
//     Results --- Matlab (conv2): 3.91 secs. IPP: 0.14 secs. Speedup: 27.0
//
//     d. Large color image - 2000x2000x3 pixels, kernel size = 30x30
//     Results --- Matlab (fft2): 3.92 secs. IPP: 0.34 secs. Speedup: 11.6
//
//     e. Multiple tiny images - 32 x 32 x 10000, kernel size = 5x5
//     Results --- Matlab (conv2): 1.83 secs. IPP: 0.06 secs. Speedup: 32.6
//
// 
//
// *************************
//
// This is an enhanced version of Matlab's conv2 command. It can be used in the same manner as conv2 but has additional features. 
// Instead of convolving a single 2D image matrix with another 2D kernel matrix, ...
// the image can be a 3D matrix, i.e. a stack of images. For example ...
// if you have a 1024x768 color image, this is a 1028x768x3 matrix ...
// which can be directly passed to the routine. The multi-threading ...
// will ensure that a different processor core will run on each ...
// color channel. Note that the kernel is still 2D, so it is not ...
// doing a 3D convolution. 

// But the stack can be any size, for example a pile ...
// of 1000 400x300 images can be convolved with kernel by passing ...
// them in a 400x300x1000 matrix. 
//
//
// Example 1:
// 
// a = single(rand(400,300));
// b = single(rand(5,8));
//
// out = ipp_mt_conv2(a,b,'valid');
// out2 = conv2(a,b,'valid');
// % out and out2 are identical
//
//
// Example 2:
// 
// a = single(rand(400,300,1000));
// b = single(rand(5,8));
//
// out = ipp_mt_conv2(a,b,'full');
// % Now compare to Matlab's command
// for i=1:1000,
//   out2(:,:,i) = conv2(a(:,:,i),b,'full');
// end
// % out and out2 are identical. But Matlab's command will be
// % much slower
//
//
// *************************************************************************************************************************
// How to compile:
//
// Type in Matlab to compile:
// >> mex -f /home/fergus/matlab/mexopts.sh -I/opt/intel/ipp/current/em64t/include -L/opt/intel/ipp/current/em64t/lib  -L/opt/intel/cce/10.0.023/lib -lguide  -lippiemergedem64t -lippimergedem64t  -lippcoreem64t  -lippsemergedem64t   -lippsmergedem64t  -lstdc++ ipp_mt_conv2.cpp        
//
// Normal output:
// ipp_mt_conv2.cpp(369): (col. 7) remark: OpenMP DEFINED LOOP WAS PARALLELIZED.
// ipp_mt_conv2.cpp(367): (col. 5) remark: OpenMP DEFINED REGION WAS PARALLELIZED.
// ipp_mt_conv2.cpp(419): (col. 5) remark: OpenMP DEFINED LOOP WAS PARALLELIZED.
// ipp_mt_conv2.cpp(417): (col. 5) remark: OpenMP DEFINED REGION WAS PARALLELIZED.
// >>
//
// Note that you will need to change the paths in the command to find (i) the IPP librariesl (ii) Intel compiler on your system and (iii) the customized mexopts.sh file - see below.

// ********************************************************************************************************************************************************************

// More detailed output of compile command in case you are have ...
// difficulties compiling/linking:
//
// >> mex -v -f /home/fergus/matlab/mexopts.sh -I/opt/intel/ipp/current/em64t/include -L/opt/intel/ipp/current/em64t/lib  -L/opt/intel/cce/10.0.023/lib -lguide  -lippiemergedem64t -lippimergedem64t  -lippcoreem64t  -lippsemergedem64t   -lippsmergedem64t  -lstdc++ ipp_mt_conv2.cpp
// ----------------------------------------------------------------
// -> options file specified on command line:
//    FILE = /home/fergus/matlab/mexopts.sh
// ----------------------------------------------------------------
// ->    MATLAB                = /misc/linux/64/opt/matlab/R2007b
// ->    CC                    = /opt/intel/cce/10.0.023/bin/icc
// ->    CC flags:
//          CFLAGS             = -ansi -D_GNU_SOURCE -fexceptions -fPIC -fno-omit-frame-pointer -openmp -pthread
//          CDEBUGFLAGS        = -g
//          COPTIMFLAGS        = -O -DNDEBUG
//          CLIBS              = -Wl,-rpath-link,/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -L/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -lmx -lmex -lmat -lm
//          arguments          =  -DMX_COMPAT_32
// ->    CXX                   = /opt/intel/cce/10.0.023/bin/icc
// ->    CXX flags:
//          CXXFLAGS           = -ansi -D_GNU_SOURCE -openmp -fPIC -fno-omit-frame-pointer -pthread
//          CXXDEBUGFLAGS      = -g
//          CXXOPTIMFLAGS      = -O3 -DNDEBUG
//          CXXLIBS            = -Wl,-rpath-link,/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -L/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -lmx -lmex -lmat -lm
//          arguments          =  -DMX_COMPAT_32
// ->    FC                    = g95
// ->    FC flags:
//          FFLAGS             = -fexceptions -fPIC -fno-omit-frame-pointer
//          FDEBUGFLAGS        = -g
//          FOPTIMFLAGS        = -O
//          FLIBS              = -Wl,-rpath-link,/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -L/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -lmx -lmex -lmat -lm
//          arguments          =  -DMX_COMPAT_32
// ->    LD                    = /opt/intel/cce/10.0.023/bin/icc
// ->    Link flags:
//          LDFLAGS            = -pthread -shared -Wl,--version-script,/misc/linux/64/opt/matlab/R2007b/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined
//          LDDEBUGFLAGS       = -g
//          LDOPTIMFLAGS       = -O
//          LDEXTENSION        = .mexa64
//          arguments          =  -L/opt/intel/ipp/current/em64t/lib -L/opt/intel/cce/10.0.023/lib -lguide -lippiemergedem64t -lippimergedem64t -lippcoreem64t -lippsemergedem64t -lippsmergedem64t -lstdc++
// ->    LDCXX                 = 
// ->    Link flags:
//          LDCXXFLAGS         = 
//          LDCXXDEBUGFLAGS    = 
//          LDCXXOPTIMFLAGS    = 
//          LDCXXEXTENSION     = 
//          arguments          =  -L/opt/intel/ipp/current/em64t/lib -L/opt/intel/cce/10.0.023/lib -lguide -lippiemergedem64t -lippimergedem64t -lippcoreem64t -lippsemergedem64t -lippsmergedem64t -lstdc++
// ----------------------------------------------------------------
//
// -> /opt/intel/cce/10.0.023/bin/icc -c  -I/opt/intel/ipp/current/em64t/include -I/misc/linux/64/opt/matlab/R2007b/extern/include -I/misc/linux/64/opt/matlab/R2007b/simulink/include -DMATLAB_MEX_FILE -ansi -D_GNU_SOURCE -openmp -fPIC -fno-omit-frame-pointer -pthread  -DMX_COMPAT_32 -O3 -DNDEBUG ipp_mt_conv2.cpp
//
// ipp_mt_conv2.cpp(130): (col. 7) remark: OpenMP DEFINED LOOP WAS PARALLELIZED.
// ipp_mt_conv2.cpp(128): (col. 7) remark: OpenMP DEFINED REGION WAS PARALLELIZED.
// ipp_mt_conv2.cpp(169): (col. 7) remark: OpenMP DEFINED LOOP WAS PARALLELIZED.
// ipp_mt_conv2.cpp(167): (col. 7) remark: OpenMP DEFINED REGION WAS PARALLELIZED.
// -> /opt/intel/cce/10.0.023/bin/icc -c  -I/opt/intel/ipp/current/em64t/include -I/misc/linux/64/opt/matlab/R2007b/extern/include -I/misc/linux/64/opt/matlab/R2007b/simulink/include -DMATLAB_MEX_FILE -ansi -D_GNU_SOURCE -fexceptions -fPIC -fno-omit-frame-pointer -openmp -pthread  -DMX_COMPAT_32 -O -DNDEBUG /misc/linux/64/opt/matlab/R2007b/extern/src/mexversion.c
//
// -> /opt/intel/cce/10.0.023/bin/icc -O -pthread -shared -Wl,--version-script,/misc/linux/64/opt/matlab/R2007b/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined -o ipp_mt_conv2.mexa64  ipp_mt_conv2.o mexversion.o  -L/opt/intel/ipp/current/em64t/lib -L/opt/intel/cce/10.0.023/lib -lguide -lippiemergedem64t -lippimergedem64t -lippcoreem64t -lippsemergedem64t -lippsmergedem64t -lstdc++ -Wl,-rpath-link,/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -L/misc/linux/64/opt/matlab/R2007b/bin/glnxa64 -lmx -lmex -lmat -lm
//
// >> 
//
// You will need to alter the mexopts.sh file that Matlab uses. Make ...
// a copy in your home matlab directory and edit the glnxa64 portion ...
// as per the example below. The alter the -f option in the mex ...
//  command above to all it.
// Here is a copy of the glnxa64 portion of my mexopts.sh file:
//
// #----------------------------------------------------------------------------
//             ;;
//         glnxa64)
// #----------------------------------------------------------------------------
// # CC and CXX should be path to Intel's 64-bit compiler 
// # note that cce is the 64-bit version and cc is the 32-bit version
//             RPATH="-Wl,-rpath-link,$TMW_ROOT/bin/$Arch"
//             CC='/opt/intel/cce/10.0.023/bin/icc'
//             CFLAGS='-ansi -D_GNU_SOURCE -fexceptions'
//             CFLAGS="$CFLAGS -fPIC -fno-omit-frame-pointer -openmp -pthread"
//             CLIBS="$RPATH $MLIBS -lm"
//             COPTIMFLAGS='-O -DNDEBUG'
//             CDEBUGFLAGS='-g'
// #
//             CXX='/opt/intel/cce/10.0.023/bin/icc'
// # Ensure we have OpenMP in here too
//             CXXFLAGS='-ansi -D_GNU_SOURCE -openmp'
//             CXXFLAGS="$CXXFLAGS -fPIC -fno-omit-frame-pointer -pthread"
//             CXXLIBS="$RPATH $MLIBS -lm"
// # Use -O3 (agressive) or -O (which is -O2) is more conservative
//             CXXOPTIMFLAGS='-O3 -DNDEBUG'
//             CXXDEBUGFLAGS='-g'
// #
// #
//             FC='g95'
//             FFLAGS='-fexceptions'
//             FFLAGS="$FFLAGS -fPIC -fno-omit-frame-pointer"
//             FLIBS="$RPATH $MLIBS -lm"
//             FOPTIMFLAGS='-O'
//             FDEBUGFLAGS='-g'
// #
//             LD="$COMPILER"
//             LDEXTENSION='.mexa64'
//             LDFLAGS="-pthread -shared -Wl,--version-script,$TMW_ROOT/extern/lib/$Arch/$MAPFILE -Wl,--no-undefined"
//             LDOPTIMFLAGS='-O'
//             LDDEBUGFLAGS='-g'
// #
//             POSTLINK_CMDS=':'
// #----------------------------------------------------------------------------
//
// Of course, you will also need to alter the path to the IPP ...
// libraries and the Intel Compiler their locations on your system.

// **************************************************************************************************************************************************

#include <mex.h> // Mex header 
#include <stdio.h>
#include <ipp.h> // Intel IPP header
#include <math.h>
#include <string.h>

#ifdef MULTITHREADING_OMP
#include <omp.h> // OpenMP header
#endif

#define MAX_NUM_THREADS      4 // Max number of parallel threads that the ...
//code will try to use (Integer). Set to 1 if you want to use a single core ...
//    (typical speedup over Matlab's conv2/fft2 is 3.5x). Set >1 for ...
//    multithreading. This number should be less than the #cpus per ...
//    machine x #cores/cpu. i.e. a machine which two quad-core cpu's ...
//    could have upto 8 threads. Note that if there are fewer images than threads 
//    then the code will automatically turn down the number of threads (since the extra ones do nothing except waste
//    resources.

// Input Arguments 
#define	IMAGE   	prhs[0] // The stack of images you want to convolve: n x m x c matrix of single precision floating point numbers.
#define KERNEL          prhs[1] // The kernel: i x j matrix of single precision floating point numbers.
#define MODE            prhs[2] // String: either "full" or "valid". Unless string is "full", code will default to "valid".

// Output Arguments 
#define	OUTPUT   	plhs[0] // Convolved stack of images. If in valid mode, this will be (n-i+1) x (m-j+1) x c  matrix of single ...
//  precision floating point numbers. If in full mode, this will be  (n+i-1) x (m+j-1) x c  matrix of single ...
//  precision floating point numbers.


void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{ 
  unsigned int image_x, image_y, kernel_x, kernel_y, output_x, output_y;
  int status,buflen,num_images,i,num_dims,number_threads;
  char *modep;
  float *kernelp, *imagep, *outputp, *kernelp_base, *imagep_base, *outputp_base;
  char default_mode[] = "full";

  IppStatus retval;
  IppiSize output_size, kernel_size, image_size;
  const mwSize *imagedims;
  int outputdims[3];  
    
  // Check for proper number of arguments 
  if (nrhs != 3) { 
    mexErrMsgTxt("Three input arguments required."); 
  } else if (nlhs > 1) {
    mexErrMsgTxt("Too many output arguments."); 
  } 
  
  // IMAGE must be a single. 
  if (mxIsSingle(IMAGE) != 1)
    mexErrMsgTxt("Image must be a single precision (float), not double.");
  
  // IMAGE must be 3-D stack of images 
  // Uncomment if you only want this to run on stacks of images
  //if (mxGetNumberOfDimensions(IMAGE) != 3)
  //  mexErrMsgTxt("Image must be a 3D array of images.");
  
  // Input must be a single. 
  if (mxIsSingle(KERNEL) != 1)
    mexErrMsgTxt("Kernel must be a single precision (float), not double.");
  
  // Get dimensions of image and kernel 
  num_dims = mxGetNumberOfDimensions(IMAGE);
  imagedims = (mwSize*) mxCalloc(num_dims, sizeof(mwSize));
  imagedims = mxGetDimensions(IMAGE);
  
  
  image_size.width = imagedims[0];
  image_size.height = imagedims[1];
  
  if (num_dims == 2)
    num_images = 1;
  else
    num_images = imagedims[2];
  
  //mexPrintf("%d images at %d by %d\n",num_images,image_size.width,image_size.height);

  kernel_size.width = round(mxGetM(KERNEL));
  kernel_size.height = round(mxGetN(KERNEL));
  
  // Get pointers to IMAGE and KERNEL
  imagep_base = (float*) mxGetData(IMAGE);
  kernelp = (float*) mxGetData(KERNEL);

   
  // MODE must be a string. 
  if (mxIsChar(MODE) != 1)
    mexErrMsgTxt("Mode must be a string.");
  
  // Input must be a row vector. 
  if (mxGetM(MODE) != 1)
    mexErrMsgTxt("Mode must be a row vector.");
  
  // Get the length of the input string. 
  buflen = (mxGetM(MODE) * mxGetN(MODE)) + 1;
   
  // Allocate memory for input and output strings. 
  modep = (char*) mxCalloc(buflen, sizeof(char));

  // Copy the string output from FILENAME into a C string input_buf. 
  status = mxGetString(MODE, modep, buflen);
  if (status != 0) 
    mexWarnMsgTxt("Not enough space. String is truncated.");
   
  // *****************************************************************************************************
  // Main part of code
  
  //********************************************************************************
  // Decide Full or valid. Default is valid 
  if (!strcmp(modep,default_mode)){ // Full convolution

    // Create output matrix of appropriate size
    output_size.width  = image_size.width  + kernel_size.width - 1;
    output_size.height = image_size.height + kernel_size.height - 1;
    outputdims[1] = output_size.height;
    outputdims[0] = output_size.width;
    outputdims[2] = num_images;
    OUTPUT = mxCreateNumericArray(3,outputdims, mxSINGLE_CLASS, mxREAL);
    
    // Check matrix generated OK
    if (OUTPUT==NULL)
      mexErrMsgTxt("Could not allocate output array");
    
    // Get pointer to output matrix
    outputp_base = (float*) mxGetData(OUTPUT);

    if (num_images<MAX_NUM_THREADS)
      number_threads= num_images;
    else
      number_threads = MAX_NUM_THREADS;
    
    // Setup openMP for core inner loop
    #pragma omp parallel num_threads(number_threads)
    {
      #pragma omp for
      // Main loop over all images in stack. Stuff inside this loop ...
      // will be multi-threaded out to different cores.
      for (i=0;i<num_images;i++){

	// Don't put any Matlab functions (e.g. mx...) in here - I ...
	// think it kill the multithreading.

	// set pointer offset for input
	imagep = imagep_base + (i*image_size.width*image_size.height);

	// set pointer offset for output
        outputp = outputp_base + (i*output_size.width*output_size.height);

	//mexPrintf("Image: %d imagep: %p %f outputp: %p %f kernelp: %p %f\n",i,imagep,*imagep,outputp,*outputp,kernelp,*kernelp);	

	// call IPP full convolution routine for 32-bit floating point matrices
	retval = ippiConvFull_32f_C1R(imagep,sizeof(float)*image_size.width,image_size,kernelp,sizeof(float)*kernel_size.width,kernel_size,outputp,sizeof(float)*output_size.width);

      }
      
    }
    
  }
  
  else{ //Valid convolution
 
    // Create output matrix of appropriate size
    output_size.width  = image_size.width  - kernel_size.width + 1;
    output_size.height = image_size.height - kernel_size.height + 1;
    outputdims[1] = output_size.height;
    outputdims[0] = output_size.width;
    outputdims[2] = num_images;
    OUTPUT = mxCreateNumericArray(3,outputdims, mxSINGLE_CLASS, mxREAL);
     
    // Check matrix generated OK
    if (OUTPUT==NULL)
      mexErrMsgTxt("Could not allocate output array");
    
    // Get pointer to output matrix
    outputp_base = (float*) mxGetData(OUTPUT);
     
    if (num_images<MAX_NUM_THREADS)
      number_threads= num_images;
    else
      number_threads = MAX_NUM_THREADS;

    // Setup openMP for core inner loop
#pragma omp parallel num_threads(number_threads)
    {
    #pragma omp for

      // Main loop over all images in stack. Stuff inside this loop ...
      // will be multi-threaded out to different cores.
      for (i=0;i<num_images;i++){
    	  
	  // set pointer offset for input
	  imagep = imagep_base + (i*image_size.width*image_size.height);
	  
	  // set pointer offset for output
	  outputp = outputp_base + (i*output_size.width*output_size.height);
 	  
	  //mexPrintf("Image: %d imagep: %p %f outputp: %p %f kernelp: %p %f\n",i,imagep,*imagep,outputp,*outputp,kernelp,*kernelp);	
	  
	  // call IPP valid convolution routine for 32-bit floating point matrices
	  retval = ippiConvValid_32f_C1R(imagep,sizeof(float)*image_size.width,image_size,kernelp,sizeof(float)*kernel_size.width,kernel_size,outputp,sizeof(float)*output_size.width);
 
	}
	
      }
    }
    

    // Parse any error (can't put inside inner loop as it will stop ...
    //  multithreading)
    if (retval!=ippStsNoErr){
      mexPrintf("Error performing convolution\n");
      
      if (retval==ippStsNullPtrErr)
	mexErrMsgTxt("Pointers are NULL\n");
      
      if (retval==ippStsSizeErr)
	mexErrMsgTxt("Sizes negative or zero\n");
      
      if (retval==ippStsStepErr)
	mexErrMsgTxt("Steps negative or zero\n");
      
      if (retval==ippStsMemAllocErr)
	mexErrMsgTxt("Memory allocation error\n");
      
    }


    // Free up
    mxFree(modep); 
 

   return;
    
}

