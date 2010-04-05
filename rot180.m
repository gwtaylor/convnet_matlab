function B = rot180(A)

%based on rot90.m
%but generalized for 3-d arrays
[m,n,k]=size(A);

%just uses indexing
B = A(m:-1:1,n:-1:1,:);