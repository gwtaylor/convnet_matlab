function [f, df] = CG_SMALLNORB_CLASSIFY_C2f(VV,Dim,XX,target, ...
  connections)

l1 = Dim(1);
l2 = Dim(2);
l3 = Dim(3);
l4 = Dim(4);
l5 = Dim(5);
l6 = Dim(6);
l7 = Dim(7);
l8 = Dim(8);

[nr,nc,numcases] = size(XX);
num_connect = size(connections,2);

filtersize1=l1;
nummaps1=l2;
downsample1=l3;
filtersize2=l4;
nummaps2=l5;
downsample2=l6;
outsize=l7;
numlabels=l8;

outr1=nr-filtersize1+1;
outc1=nc-filtersize1+1;
nr1 = outr1/downsample1; nc1 = outc1/downsample1; %downsampled dimensions

outr2=nr1-filtersize2+1;
outc2=nc1-filtersize2+1;
nr2 = outr2/downsample2; nc2 = outc2/downsample2;

%deconversion of vectorized parameters
filters1 = reshape(VV(1:l1*l1*l2),[l1 l1 l2]);
xxx = l1*l1*l2;
convcoeff1 = reshape(VV(xxx+1:xxx+l2),l2,1);
xxx = xxx+l2;
filters2 = reshape(VV(xxx+1:xxx+l4*l4*(num_connect*l5)),[l4 l4 num_connect*l5]);
xxx = xxx+l4*l4*(num_connect*l5);
convcoeff2 = reshape(VV(xxx+1:xxx+l5),l5,1);
xxx = xxx+l5;
w_class = reshape(VV(xxx+1:xxx+(l7+1)*l8),l7+1,l8);

%forward pass
%returns output map of convolution layer (used in backprop)
[yy,map1,y1,map2] = convnet_forward2f(XX,filters1,convcoeff1,downsample1,filters2, ...
    convcoeff2,downsample2,connections);
yy = [yy ones(numcases,1,'single')]; %extra dimension (for bias)

%go through classifier
targetout = convnet_probs(yy,w_class);  

%cross-entropy error generalized to multi-class
f = -sum(sum( target.*log(targetout)));

delta5 = (targetout-target); 
dw_class =  yy'*delta5; 

%subsampling layer has no parameters
%if the layer following the subsampling layer is a fully connected layer
%then the sensitivity maps can be computed by vanilla backprop
delta4 = (delta5*w_class');
delta4 = delta4(:,1:end-1);

%delta4 is in vector form
%but now we need to reshape it so we can do the element wise multiplication
%This is simply reversing the reshaping operations performed in the forward
%pass
delta4r=transpose(delta4);
delta4r=reshape(delta4r,[nr2 nc2 nummaps2 numcases]); 
delta4r=permute(delta4r,[1 2 4 3]); %last dimension is filters

%need to upsample the downsampling layer's sensitivity map to make it the
%same size as the convolutional layer's map
%repeat for each map in the convolutional layer
delta3 = zeros(outr2,outc2,numcases,nummaps2,'single');
dfilters2=zeros(filtersize2,filtersize2,numcases,num_connect*nummaps2,'single');
dconvcoeff2=zeros(nummaps2,1,'single');
for jj=1:nummaps2
  %upsample the deltas to make them compatible size
  %up = reshape(kron(delta4r(:,:,:,jj),ones(downsample2,'single')), ...
  %  [outr2 outc2 numcases]);
  up = expand(delta4r(:,:,:,jj),[downsample2 downsample2 1]);
  %for sigmoid nonlinearity
  %delta3(:,:,:,jj)=convcoeff2(jj)*map2(:,:,:,jj).*(1-map2(:,:,:,jj)).*up;
  %for tanh nonlinearity
  delta3(:,:,:,jj)=convcoeff2(jj)*(1-map2(:,:,:,jj).^2).*up;
  
  tmp=map2(:,:,:,jj).*up; %dconvcoeff not summed
  dconvcoeff2(jj)=sum(tmp(:)); %summing over pixels and cases
end

%filter gradients - this is where it is different for the 2-layer model
%note that filters in the second convolutional layer correspond to both
%inputs and outputs (for first layer there was just a single input)

for jj=1:nummaps2 %iterate through output
  for kk=1:num_connect; %iterate through prev layer feature maps (input)
    input_map = connections(jj,kk);
    filteridx=num_connect*(jj-1)+kk; %index to 3rd dim in filters2
    for cc=1:numcases
      %here we perform cross-correlation by rotating the kernel 180 deg
      %where the kernel is the sensitivity maps and applying convolution
      krnl = rot180(delta3(:,:,cc,jj));
%       dfilters2(:,:,cc,filteridx)=rot180(conv2(y1(:,:,cc,input_map), ...
%         krnl,'valid'));
      %unfortunately, the kernel here (rotated sensitivity at one layer up)
      %is case-specific and so I do not think we can do this in parallel
      dfilters2(:,:,cc,filteridx)=rot180(ipp_mt_conv2(y1(:,:,cc,input_map), ...
        krnl,'valid'));
    end
  end
end
dfilters2=squeeze(sum(dfilters2,3)); %sum over cases

%subsampling layer 1
%layer l+1 was not a fully-connected layer, it was convolutional
%so this is slighly harder than subsampling layer 2 (calculation of delta4)
delta2 = zeros(nr1,nc1,numcases,nummaps1,'single');
for jj=1:nummaps1
    %only add the contribution if subsampling layer 1 is connected to
    %convolutional layer 2
    %rows gives the output maps we need to consider
    %for each row, cols gives the connection index (we need this for the
    %filter number .. i.e. 3rd dimension of filters2)
    [rows,cols] = find(connections==jj);
    for kk=1:length(rows)
      map_out = rows(kk);
      filteridx=num_connect*(map_out-1)+cols(kk); %3rd index in filters2
      krnl = rot180(filters2(:,:,filteridx));
      delta2(:,:,:,jj) = delta2(:,:,:,jj)+ ...
        ipp_mt_conv2(delta3(:,:,:,map_out),krnl, 'full');
    end
end

%convolutional layer 1
%need to upsample the downsampling layer's sensitivity map to make it the
%same size as the convolutional layer's map
%repeat for each map in the convolutional layer
delta1 = zeros(outr1,outc1,numcases,nummaps1,'single');
dfilters1=zeros(filtersize1,filtersize1,numcases,nummaps1,'single');
dconvcoeff1=zeros(nummaps1,1,'single');
for jj=1:nummaps1
  %upsample the deltas to make them compatible size
  %up = reshape(kron(delta2(:,:,:,jj),ones(downsample1,'single')),[outr1 ...
  %                    outc1 numcases]);
  up = expand(delta2(:,:,:,jj),[downsample1 downsample1 1]);
  %for sigmoid nonlinearity
  %delta3(:,:,:,jj)=convcoeff2(jj)*map2(:,:,:,jj).*(1-map2(:,:,:,jj)).*up;
  %for tanh nonlinearity
  delta1(:,:,:,jj)=convcoeff1(jj)*(1-map1(:,:,:,jj).^2).*up;
  
  tmp=map1(:,:,:,jj).*up; %dconvcoeff not summed
  dconvcoeff1(jj)=sum(tmp(:)); %summing over pixels and cases
end

%filter gradients - simple for convolutional layer 1
%since there is a single input (the data)
for jj=1:nummaps1
    %filter gradients
  for cc=1:numcases
    %here we perform cross-correlation by rotating the kernel 180 deg
    %where the kernel is the sensitivity maps and applying convolution
    krnl = rot180(delta1(:,:,cc,jj));
    %dfilters1(:,:,cc,jj)=rot180(conv2(XX(:,:,cc),krnl,'valid'));
    %unfortunately, the kernel here (rotated sensitivity at one layer up)
    %is case-specific and so I do not think we can do this in parallel
    dfilters1(:,:,cc,jj)=rot180(ipp_mt_conv2(XX(:,:,cc),krnl,'valid'));
  end
end

dfilters1=squeeze(sum(dfilters1,3)); %sum over cases

df = [dfilters1(:);dconvcoeff1(:);dfilters2(:);dconvcoeff2(:);dw_class(:)];