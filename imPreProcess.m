function pim = imPreProcess(img,k)
% 
% Processes a given image using a specific scheme described in Pinto et al
% img is supposed to be a grayscale image
% k is the weighting kernel that will be used in local neighborhoods
%

dim = double(img);

%
% 1. subtract the mean and divide by the standard deviation
%
mn = mean(dim(:));
sd = std(dim(:));

dim = dim - mn;
dim = dim / sd;

%
% 2. calculate local mean and std
% divide each pixel by local std if std>1
%
lmn = conv2(dim,k,'valid');
lmnsq = conv2(dim.^2,k,'valid');
lvar = lmnsq - lmn.^2;
lvar(lvar<0) = 0; % avoid numerical problems
lstd = sqrt(lvar);
lstd(lstd<1) = 1;

shifti = floor(size(k,1)/2)+1;
shiftj = floor(size(k,2)/2)+1;

% since we do valid convolutions
dim = dim(shifti:shifti+size(lstd,1)-1,shiftj:shiftj+size(lstd,2)-1);
dim = dim - lmn;
dim = dim ./ lstd;

sz = size(dim);
shift = floor(( max(sz) - sz) / 2);

%
% 3. pad with zeros
%
pim = zeros(max(sz));
pim(1+shift(1):shift(1)+sz(1),1+shift(2):shift(2)+sz(2)) = dim;

