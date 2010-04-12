% Similar to norbbackpropc1.m but multi-layer architecture:
%   convolutional layer 1
%   sub-sampling layer 1
%   convolutional layer 2
%   sub-sampling layer 2
%   1 layer of multi-class logistic regression
% No pre-training
%
% Uses a faster forward-pass & backprop
% (relies on IPP libraries)
% Note that since everything is "single" type; it no longer checkgrads

preprocessing_type = 1; %use Local Contrast Normalization

maxepoch=200;

fprintf('Loading & preprocessing data\n');
smallnorb_makebatches %preprocess 24,300 training & 24,300 test cases
smallnorb_reshape %make 4-d data

%convert everything to singles
batchdata = single(batchdata);
testbatchdata = single(testbatchdata);
batchtargets = single(batchtargets);
testbatchtargets = single(testbatchtargets);

%Uncomment these two lines to only use the first 20 batches of data
%It runs much faster; useful for debugging
%batchdata = batchdata(:,:,:,1:20);
%testbatchdata = testbatchdata(:,:,:,1:20);

[nr nc numcases numbatches] = size(batchdata);

nummaps1=6;    %number of output feature maps
filtersize1=5; %size of a filter in the filter bank (currently locked to
               %the same for both rows, columns)
downsample1=2; %downsampling ratio (currently locked to the same for both
               %rows,columns)

nummaps2=16;   %number of output feature maps
filtersize2=5; %size of a filter in the filter bank (currently locked to
               %the same for both rows, columns)
downsample2=2; %downsampling ratio (currently locked to the same for both
               %rows,columns)
num_connect = 4; %each map in convolution layer 2 looks at this many 
                 %randomly selected inputs from the prev layer

numlabels = size(batchtargets,2);

outr1=nr-filtersize1+1;
outc1=nc-filtersize1+1;
nr1 = outr1/downsample1; nc1 = outc1/downsample1; %downsampled dimensions

outr2=nr1-filtersize2+1;
outc2=nc1-filtersize2+1;
nr2 = outr2/downsample2; nc2 = outc2/downsample2;

%%%% INITIALIZE PARAMETERS

%convolution layer
filters1 = 0.01*randn(filtersize1,filtersize1,nummaps1,'single');
convcoeff1 = ones(nummaps1,1,'single') + 0.01*randn(nummaps1,1,'single'); %trainable scalar coeff 1 per map
%3rd dim: first num_connect entries are for output map 1, next num_connect 
%entries are for output map 2, etc.
filters2 = 0.01*randn(filtersize2,filtersize2,num_connect*nummaps2,'single');
convcoeff2 = ones(nummaps2,1,'single') + 0.01*randn(nummaps2,1,'single'); %trainable scalar coeff 1 per map

%logistic regression (classifier)
outsize=nr2*nc2*nummaps2; %classification weights are connected to all
                          %maps in the second subsampling layer
w_class = 0.01*randn(outsize+1,numlabels,'single'); %+1 is for bias

% CONNECTIVITY
% Each map of second convolutional layer is randomly connected to 4 maps
% from the first subsampling output map
% each row in connections corresponds to a feature map in the second
% convolution layer
connections = zeros(nummaps2,num_connect);
rand('state',0);
for ii=1:nummaps2
  %slightly ugly because it uses the stats toolbox to sample without
  %replacement
  connections(ii,:) = randsample(nummaps1,num_connect); 
end

%%%% END INITIALIZATION

%the internal function needs to know about dimensions
l1=filtersize1;
l2=nummaps1;
l3=downsample1;
l4=filtersize2;
l5=nummaps2;
l6=downsample2;
l7=outsize; %output maps of subsampling layer flattened to vector
l8=numlabels; %hard-coded

test_err=[];
train_err=[];

for epoch = 1:maxepoch

%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0; 
err_cr=0;
counter=0;

for batch = 1:numbatches
  data = [batchdata(:,:,:,batch)];
  target = [batchtargets(:,:,batch)];
  
  %forward pass
  yy = convnet_forward2f(data,filters1,convcoeff1,downsample1,filters2, ...
    convcoeff2,downsample2,connections);
  yy = [yy ones(numcases,1,'single')]; %extra dimension (for bias)
  
  %go through classifier
  targetout = convnet_probs(yy,w_class);
  
  [I J]=max(targetout,[],2);
  [I1 J1]=max(target,[],2);
  counter=counter+length(find(J==J1));
  %compute cross-entropy error
  err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
end
 train_err(epoch)=(numcases*numbatches-counter);
 train_crerr(epoch)=err_cr/numbatches;

%%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0;
err_cr=0;
counter=0;

[nr nc testnumcases testnumbatches] = size(testbatchdata);

for batch = 1:testnumbatches
  data = [testbatchdata(:,:,:,batch)];
  target = [testbatchtargets(:,:,batch)];
  
  %forward pass
  yy = convnet_forward2f(data,filters1,convcoeff1,downsample1,filters2, ...
    convcoeff2,downsample2,connections);
  yy = [yy ones(numcases,1,'single')]; %extra dimension (for bias)
  
  %go through classifier
  targetout = convnet_probs(yy,w_class);
  
  [I J]=max(targetout,[],2);
  [I1 J1]=max(target,[],2);
  counter=counter+length(find(J==J1));
  %compute cross-entropy error
  err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
end
 test_err(epoch)=(testnumcases*testnumbatches-counter);
 test_crerr(epoch)=err_cr/testnumbatches;
 fprintf(1,'Before epoch %d Train # misclassified: (%d/%d : %6.4f).\n Test # misclassified: (%d/%d : %6.4f) \t \t \n',...
            epoch,train_err(epoch),numcases*numbatches, ...
            train_err(epoch)/(numcases*numbatches),test_err(epoch), ...
            testnumcases*testnumbatches, ...
            test_err(epoch)/(testnumcases*testnumbatches));

%%%%%%%%%%%%%% END OF COMPUTING TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 tt=0;
 for batch = 1:numbatches/10
 fprintf(1,'epoch %d batch %d\r',epoch,batch);

%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 tt=tt+1; 
 data=[];
 target=[]; 
 for kk=1:10
    %since batchdata is 4-d
    %need to use the general form of cat
    data = cat(3,data,batchdata(:,:,:,(tt-1)*10+kk));
    target=[target
        batchtargets(:,:,(tt-1)*10+kk)];
 end 

%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  max_iter=3;

  if epoch<6  % First update top-level weights holding other weights fixed. 

    %perform forward pass to compute input to classifier
    %but do not add extra bias dimension (added inside
    %CG_SMALLNORB_CLASSIFY_CINIT)

    %forward pass
    yy = convnet_forward2f(data,filters1,convcoeff1,downsample1,filters2, ...
      convcoeff2,downsample2,connections);
  
    VV = w_class(:);
    Dim = [l7;l8];
    
    [X, fX] = minimize(VV,'CG_SMALLNORB_CLASSIFY_CINIT',max_iter,Dim,yy,target);
    w_class = reshape(X,l7+1,l8);

  else
      
    VV = [filters1(:);convcoeff1(:);filters2(:);convcoeff2(:);w_class(:)];
    Dim = [l1; l2; l3; l4; l5; l6; l7; l8];

    [X, fX] = minimize(VV,'CG_SMALLNORB_CLASSIFY_C2f',max_iter,Dim, ...
      data,target,connections);
    
    filters1 = reshape(X(1:l1*l1*l2),[l1 l1 l2]);
    xxx = l1*l1*l2;
    convcoeff1 = reshape(X(xxx+1:xxx+l2),l2,1);
    xxx = xxx+l2;
    filters2 = reshape(X(xxx+1:xxx+l4*l4*(num_connect*l5)),[l4 l4 num_connect*l5]);
    xxx = xxx+l4*l4*(num_connect*l5);
    convcoeff2 = reshape(X(xxx+1:xxx+l5),l5,1);
    xxx = xxx+l5;
    w_class = reshape(X(xxx+1:xxx+(l7+1)*l8),l7+1,l8);


  end
%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 end

 %save smallnorbclassifyconv2f_weights filters1 convcoeff1 filters2 convcoeff2 w_class connections
 %save smallnorbclassifyconv2f_error test_err test_crerr train_err train_crerr;

end



