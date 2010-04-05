%for convolutional net
%Rob's ipp code needs singles
%also we don't want to reshape vectors into images every time we access
%data
%so this reshapes batchdata, testbatchdata
%and makes them singles
%run after smallnorb_makebatches


%pixels come first
batchdata = permute(batchdata,[2 1 3]);
testbatchdata = permute(testbatchdata,[2 1 3]);

[numdims numcases numbatches] = size(batchdata);

batchdata = reshape(batchdata,[sqrt(numdims) sqrt(numdims) numcases ...
                    numbatches]);

[numdims numcases numbatches] = size(testbatchdata);

testbatchdata = reshape(testbatchdata,[sqrt(numdims) sqrt(numdims) numcases ...
                    numbatches]);

