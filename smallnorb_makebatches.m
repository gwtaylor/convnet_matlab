%first, set flag
%preprocessing_type = 0; %Vinod's suggestion
%preprocessing_type = 1; %diCarlo method (Koray's implementation)


datasetpath = '/misc/FergusGroup/gwtaylor/smallnorb';

load(fullfile(datasetpath,['smallnorb-5x46789x9x18x6x2x32x32-training-' ...
                    'dat-matlab-bicubic.mat']));

%labels are the same as the 96x96 dataset
load(fullfile(datasetpath,['smallnorb-5x46789x9x18x6x2x96x96-training-' ...
                    'cat-matlab.mat']));
                
                
load(fullfile(datasetpath,['smallnorb-5x01235x9x18x6x2x32x32-testing-' ...
                    'dat-matlab-bicubic.mat']));
                
%labels are the same as the 96x96 dataset
load(fullfile(datasetpath,['smallnorb-5x01235x9x18x6x2x96x96-testing-' ...
                    'cat-matlab.mat']));

% validdata = traindata(:,1,20001:end); %use the last 4300 examples (only
%                                       %1/2 of stereo pair) as the
%                                       %validation set
% 
% validlabels = trainlabels(20001:end);
% 
% traindata = traindata(:,1,1:20000);
% 
% trainlabels = trainlabels(1:20000);

traindata = traindata(:,1,:);
testdata = testdata(:,1,:);

%get rid of singleton dimension
%validdata = double(squeeze(validdata));
traindata = double(squeeze(traindata));
testdata = double(squeeze(testdata));

%Before anything else; take care of the transpose issue: row-major vs
%col-major
traindata = transpose_dataset(traindata);
testdata = transpose_dataset(testdata);

%Normalize images

if preprocessing_type ==0

  %Vinod suggests multiplying each image by a scalar value
  %Such that all images have the same average pixel value
  %Just divide each image by its average pixel value?
  traindata = traindata/255; %work in [0 1] space
  %validdata = validdata/255;
  testdata = testdata/255;
  
  % $$$ validdata = bsxfun(@rdivide,validdata,mean(validdata,1));
  % $$$ traindata = bsxfun(@rdivide,traindata,mean(traindata,1));
  % $$$ testdata = bsxfun(@rdivide,testdata,mean(testdata,1));
  m = mean(traindata',1);
  s = mean(std(traindata',1));
  traindata=bsxfun(@minus,traindata,m')/s;
  % m = mean(validdata',1);
  % s = mean(std(validdata',1));
  % validdata=bsxfun(@minus,validdata,m')/s;
  m = mean(testdata',1);
  s = mean(std(testdata',1));
  testdata=bsxfun(@minus,testdata,m')/s;
  
elseif preprocessing_type ==1
  
  %diCarlo method
  %done image-by-image
  %assume square images
  nr = sqrt(size(traindata,1)); nc=nr;
  
  %load 9x9 Gaussian kernel (saved as "ker")
  %load('/home/gwtaylor/matlab/koray/randomc101/data/params.mat','ker')
  
  %5x5 Gaussian kernel
  ker=fspecial('Gaussian',5,1.591);
  
  %3x3 Gaussian kernel
  %ker=fspecial('Gaussian',3,1.591);
  
  numcases = size(traindata,2);
  traindata_pp = zeros((nr-size(ker,1)+1)*(nc-size(ker,2)+1),numcases);
  
  for ii=1:size(traindata,2)
    im = traindata(:,ii);
    im = reshape(im,nr,nc);
    
    pim = imPreProcess(im,ker);
    
    %     figure(30);
    %     subplot(1,2,1);
    %     imagesc(im); colormap gray; axis off; axis equal;
    %     subplot(1,2,2);
    %     imagesc(pim); colormap gray; axis off; axis equal;
    
    if mod(ii,1000)==0
      fprintf('done train %d/24300\r',ii);
    end
    
    traindata_pp(:,ii) = ...
      reshape(pim,[(nr-size(ker,1)+1)*(nc-size(ker,2)+1) 1]);
    
  end
  
  traindata = traindata_pp;
  
  numcases = size(testdata,2);
  testdata_pp = zeros((nr-size(ker,1)+1)*(nc-size(ker,2)+1),numcases);
  
  for ii=1:size(testdata,2)
    im = testdata(:,ii);
    im = reshape(im,nr,nc);
    
    pim = imPreProcess(im,ker);
    
    %     figure(30);
    %     subplot(1,2,1);
    %     imagesc(im); colormap gray; axis off; axis equal;
    %     subplot(1,2,2);
    %     imagesc(pim); colormap gray; axis off; axis equal;
    
    if mod(ii,1000)==0
      fprintf('done test %d/24300\r',ii);
    end
    
    testdata_pp(:,ii) = ...
      reshape(pim,[(nr-size(ker,1)+1)*(nc-size(ker,2)+1) 1]);
    
  end
  
  testdata = testdata_pp;

clear traindata_pp testdata_pp
else error('Unknown preprocessing type')
  
end

ytrain = zeros(5,24300); %holds 1-of-K encoded labels
for kk=1:5
  ytrain(kk,trainlabels==kk-1)=1;
end

% yvalid = zeros(5,4300); %holds 1-of-K encoded labels
% for kk=1:5
%   yvalid(kk,validlabels==kk-1)=1;
% end

ytest = zeros(5,24300); %holds 1-of-K encoded labels
for kk=1:5
  ytest(kk,testlabels==kk-1)=1;
end

clear trainlabels testlabels

%randomly permute the order of the training set
rand('state',0); %keep track
numcases = size(traindata,2);
randomorder=randperm(numcases);

batchsize = 100;
numbatches=numcases/batchsize;
numdims  =  size(traindata,1);

%Use Russ' convention
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, 5, numbatches);

for b=1:numbatches
  batchdata(:,:,b) = traindata(:,randomorder(1+(b-1)*batchsize:b*batchsize))';
  batchtargets(:,:,b) = ytrain(:, randomorder(1+(b-1)*batchsize:b*batchsize))';
end;
clear traindata ytrain;

%randomly permute the order of the test set
rand('state',1); %keep track
numcases = size(testdata,2);
randomorder=randperm(numcases);

batchsize = 100;
numbatches=numcases/batchsize;
numdims  =  size(testdata,1);

%Use Russ' convention
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, 5, numbatches);

for b=1:numbatches
  testbatchdata(:,:,b) = testdata(:,randomorder(1+(b-1)*batchsize:b*batchsize))';
  testbatchtargets(:,:,b) = ytest(:, randomorder(1+(b-1)*batchsize:b*batchsize))';
end;
clear testdata ytest;


%%% Reset random seeds 
rand('state',sum(100*clock)); 
randn('state',sum(100*clock)); 



