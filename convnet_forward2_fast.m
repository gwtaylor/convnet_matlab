function [yy,map1,y1,map2] = convnet_forward2_fast(data,filters1,convcoeff1, ...
                                      downsample1,filters2,convcoeff2, ...
                                      downsample2,connections)

%takes a mini-batch of data and does a forward pass through the
%convolutional net
%data and filters should be singles
%downsample : downsampling factor
%returns: yy: outputs of final subsampling layer in vector form
%         map2: output of nonlinearity at 2nd convolution layer
%         y1:   output of 1st subsampling layer
%         map1: output of nonlinearity at 1st convolution layer

%dimensions
[nr,nc,numcases]=size(data);
outr1=nr-size(filters1,1)+1;
outc1=nc-size(filters1,2)+1;
nr1 = outr1/downsample1; nc1 = outc1/downsample1; %downsampled dimensions
nummaps1 = size(filters1,3);
num_connect = size(connections,2);
nummaps2 = size(filters2,3)/num_connect;

outr2=nr1-size(filters2,1)+1;
outc2=nc1-size(filters2,2)+1;
nr2 = outr2/downsample2; nc2 = outc2/downsample2;

%preallocate
resp1 = zeros(outr1,outc1,numcases,nummaps1,'single'); %filter responses
map1 = resp1; %output maps
resp2 = zeros(outr2,outc2,numcases,nummaps2,'single'); %filter responses
map2 = resp2; %output maps

%output of subsampling layers
y1 = zeros(nr1,nc1,numcases,nummaps1,'single');
y2 = zeros(nr2,nc2,numcases,nummaps2,'single');

%dsf = ones(downsample,'single'); %downsampling filter
dsf1 = ones(downsample1,'single');
dsf2 = ones(downsample2,'single');

%convolution layer 1
for jj=1:nummaps1
  resp1(:,:,:,jj)=ipp_mt_conv2(data,filters1(:,:,jj),'valid');
end

%nonlinearity
%map1 = 1./(1+exp(-resp1));
map1 = tanh(resp1); %element-wise

%subsampling layer 1
for jj=1:nummaps1
  a = ipp_mt_conv2(map1(:,:,:,jj),dsf1,'valid'); %average
  y1(:,:,:,jj)=convcoeff1(jj)*a(1:downsample1:end,1:downsample1:end,:);
end

%connections matrix tells us which subsampling layer 1 feature maps are
%combined by each feature map in convolution layer 2
for jj=1:nummaps2
  for kk=1:num_connect; %iterate through prev layer feature
        input_map = connections(jj,kk);
        filteridx=num_connect*(jj-1)+kk; %index to 3rd dim in filters2
        resp2(:,:,:,jj)=resp2(:,:,:,jj)+ipp_mt_conv2(y1(:,:,:,input_map), ...
          filters2(:,:,filteridx),'valid');
  end
end
%nonlinearity
%map2 = 1./(1+exp(-resp2));
map2 = tanh(resp2); %element-wise

%subsampling layer 2
for jj=1:nummaps2
  a = ipp_mt_conv2(map2(:,:,:,jj),dsf2,'valid'); %average
  y2(:,:,:,jj)=convcoeff2(jj)*a(1:downsample2:end,1:downsample2:end,:);
end
%now we feed the output (i.e. all maps) to a classifier
%basically we want to flatten here (except for cases)
yy = permute(y2,[1 2 4 3]); %cases are last dimension
yy = transpose(reshape(yy,nr2*nc2*nummaps2,numcases)); %transpose so that
                                                      %cases are the
                                                      %first dimension
