%Demonstrates the use of checkgrad (finite difference method)
%To ensure that the gradients for a 2-layer convolutional net (2-D) are ok
%It checks the function fn_2layer_convnet_classify which is used in backprop
filtersize1=3;
nummaps1=2;
downsample1=2;

filtersize2=2;
nummaps2=2;
downsample2=2;

%connectivity map only for the 2nd layer
%each feature map in layer 1 is connected to the input
num_connect = 2;

%data properties
nr = 12;
nc = 12;
numcases = 3;

numlabels = 2;

%dimensions of convolutional layer 1
outr1=nr-filtersize1+1;
outc1=nc-filtersize1+1;
%dimensions of subsampling layer 1
nr1 = outr1/downsample1; nc1 = outc1/downsample1; %downsampled dimensions

%dimensions of convolutional layer 2
outr2=nr1-filtersize2+1;
outc2=nc1-filtersize2+1;

%dimensions of subsampling layer 2
nr2 = outr2/downsample2; nc2 = outc2/downsample2;

%Generate random "training data"
XX = randn(nr,nc,numcases); 

%generate random labels to match data
r = rand(1,numlabels);
r = r/sum(r,2);
targets = zeros(numcases,numlabels);

[junk,ind] = max(r,[],2);
targets(:,ind) =1; 
for c=1:numcases  
  targets(c,:)=multirnd(r,1);
end

%model parameters
filters1 = 0.01*randn(filtersize1,filtersize1,nummaps1);
convcoeff1 = ones(nummaps1,1) + 0.01*randn(nummaps1,1); %trainable scalar coeff 1 per map
filters2 = 0.01*randn(filtersize2,filtersize2,num_connect*nummaps2);
convcoeff2 = ones(nummaps2,1)+ 0.01*randn(nummaps2,1); %trainable scalar coeff 1 per map
outsize=nr2*nc2*nummaps2; %classification weights are connected to all
                          %maps in the second subsampling layer
w_class = 0.01*randn(outsize+1,numlabels); %+1 is for bias

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
  connections(ii,:) = randSample(nummaps1,num_connect); 
end

%vectorize the parameters
VV = [filters1(:);convcoeff1(:);filters2(:);convcoeff2(:);w_class(:)];

Dim(1)=filtersize1;
Dim(2)=nummaps1;
Dim(3)=downsample1;
Dim(4)=filtersize2;
Dim(5)=nummaps2;
Dim(6)=downsample2;
Dim(7)=outsize;
Dim(8)=numlabels;

%First col - grads reported by function (dy)
%Second col - grads by finite differences (dh)
%Returned value (ans) - error measure: norm(dh-dy)/norm(dh+dy)
checkgrad('fn_2layer_convnet_classify',VV,1e-8,Dim,XX,targets,connections)
