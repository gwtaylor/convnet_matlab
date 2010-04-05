function data = transpose_dataset(data)
%takes a numpixelsxnumcases dataset in vector form
%where images are assumed square (i.e. sqrt(numpixels)xsqrt(numpixels))
%transposes each image
%and returns in vector form
%not touching the case order
data = reshape(data, ...
  [sqrt(size(data,1)) sqrt(size(data,1)) size(data,2)]);
data = permute(data,[2 1 3]); %transposes each image but leaves case order intact
data = reshape(data, ...
  [size(data,1)*size(data,2) size(data,3)]);
