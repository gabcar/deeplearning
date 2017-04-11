%exercise1
clc;clear all;close all;
rng(400)
addpath Datasets/cifar-10-batches-mat/;
tic;

%"one" dataset
[trainX,trainY,trainy] = LoadBatch('data_batch_1.mat');
[validX,validY,validy] = LoadBatch('data_batch_2.mat');
[testX,testY,testy] = LoadBatch('test_batch.mat');

%center data
mean_X = mean(trainX, 2);
trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]);
validX = validX - repmat(mean_X, [1, size(trainX, 2)]);
testX = testX - repmat(mean_X, [1, size(trainX, 2)]);



% %all datasets
% -----------------------
% [train1X,train1Y,train1y] = LoadBatch('data_batch_1.mat');
% [train2X,train2Y,train2y] = LoadBatch('data_batch_2.mat');
% [train3X,train3Y,train3y] = LoadBatch('data_batch_3.mat');
% [train4X,train4Y,train4y] = LoadBatch('data_batch_4.mat');
% [train5X,train5Y,train5y] = LoadBatch('data_batch_5.mat');
% [test5X,test5Y,test5y] = LoadBatch('test_batch.mat');
%
% trainX = [train1X,train2X,train3X,train4X,train5X(:,1:9000)];
% trainY = [train1Y,train2Y,train3Y,train4Y,train5Y(:,1:9000)];
% trainy = [train1y;train2y;train3y;train4y;train5y(1:9000)];
%
% validX = train5X(:,9001:end);
% validY = train5Y(:,9001:end);
% validy = train5y(9001:end);
% testX = test5X;
% testY = test5Y;
% testy = test5y;

%------------------------

%disp([size(trainX),size(validX),size(testX)])
N = length(trainy);
d = size(trainX,1);
K = max(trainy);
lambda = 0;

%this section tests functions
[W,b] = initializeParams(trainX,trainY);
P = EvaluateClassifier(trainX,W,b);
acc = ComputeAccuracy(trainX,trainy,W,b);
cost = ComputeCost(trainX,trainY,W,b,lambda);

[grad_W,grad_b] = ComputeGradients(trainX,trainY,W,b,lambda);

toc
%---------%---------%---------%---------%---------

function [X,Y,y] = LoadBatch(filename)
A = load(filename);
X = double(A.data)'/255;
y = double(A.labels+1);
K = max(y);
N = length(y);
Y = zeros(K,N);
for i = 1:N
    Y(y(i),i) = 1;
end
end

function [W,b] = initializeParams(X,Y)
m=50;
d = size(X,1);
K = size(Y,1);
W1 = .001*randn(m,d);
W2 = .001*randn(K,m);
b1 =zeros(m,1);
b2 =zeros(K,1);

b = {b1,b2};
W = {W1,W2};
end

function [P,h,s1,s2] = EvaluateClassifier(X,W,b)
s1 = bsxfun(@plus,W{1}*X,b{1});
h = s1.*(s1>0);
s2 = bsxfun(@plus,W{2}*h,b{2});
P = softmax(s2);
%disp(['size of P: ', num2str(size(P))])
end

function acc = ComputeAccuracy(X,y,W,b)
p = EvaluateClassifier(X,W,b);
[~,kstar] = max(p);
check = kstar'==y;
acc = sum(check)/length(check);
end

%cost function
function J = ComputeCost(X,Y,W,b,lambda)
n = size(X,2);
Wij = sum(sum(W{1}.^2,1),2) + sum(sum(W{2}.^2,1),2);
J = (1/n)*sum(lcross(X,Y,W,b))+lambda*Wij;
end

%function for l cross section
function l = lcross(X,Y,W,b)
l = -log(sum(Y.*EvaluateClassifier(X,W,b)));
end

%gradients
function [grad_W,grad_b]=MyComputeGrads(X,Y,W,b,lambda)
N = size(X,2);
[P,h,s1] = EvaluateClassifier(X,W,b);

dldb1 = zeros(size(b{1}));dldb2 = zeros(size(b{2}));
dldW1 = zeros(size(W{1}));dldW2 = zeros(size(W{2}));
dldb = {dldb1,dldb2};dldW={dldW1,dldW2};

for i=1:N
    g = (-Y(:,i)'/(Y(:,i)'*P(:,i)))*(diag(P(:,i))-(P(:,i)*P(:,i)'));
    dldb{2} = dldb{2}+g';
    dldW{2} = dldW{2}+g'*h(:,i)';
    g = g*dldW{2};
    g = g*diag(s1(:,i).*(s1(:,i)>0));
    dldb{1} = dldb{1}+g';
    dldW{1} = dldW{1}+g'*X(:,i)';
end

grad_W1 = dldW{1}/N + 2*lambda*W{1};
grad_b1 = dldb{1}/N;
grad_W2 = dldW{2}/N + 2*lambda*W{2};
grad_b2 = dldb{2}/N;
grad_W = {grad_W1,grad_W2};
grad_b = {grad_b1,grad_b2};
end

%gradient comparing function
function [agrad_W, agrad_b,diff] = ComputeGradients(X,Y,W,b,lambda)
[agrad_W, agrad_b] = MyComputeGrads(X(:,1), Y(:,1), ...
    W,b, lambda);
[fastngrad_b,fastngrad_W] = ComputeGradsNum(X(:,1),Y(:,1),W,b,lambda,1e-5);
toc
[slowngrad_b,slowngrad_W] = ComputeGradsNumSlow(X(:,1),Y(:,1),W,b,lambda,1e-5);
toc

diffb1_fast = max(max(abs(agrad_b{1}-fastngrad_b{1})))
diffb1_slow = max(max(abs(agrad_b{1}-slowngrad_b{1})))
diffW1_fast = max(max(abs(agrad_W{1}-fastngrad_W{1})))
diffW1_slow = max(max(abs(agrad_W{1}-slowngrad_W{1})))
diffb2_fast = max(max(abs(agrad_b{2}-fastngrad_b{2})))
diffb2_slow = max(max(abs(agrad_b{2}-slowngrad_b{2})))
diffW2_fast = max(max(abs(agrad_W{2}-fastngrad_W{2})))
diffW2_slow = max(max(abs(agrad_W{2}-slowngrad_W{2})))

diff = diffW1_fast;
diff =0;
end


%-----------------------------------------------------------------%
%GRADIENT FUNCTIONS
%SLOW
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = {zeros(size(W{1})),zeros(size(W{2}))};
grad_b = {zeros(size(W{1}, 1), 1),zeros(size(W{2}, 1), 1)};

for k=1:length(b)
    for i=1:length(b{k})
        b_try = b;
        b_try{k}(i) = b_try{k}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        b_try = b;
        b_try{k}(i) = b_try{k}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b{k}(i) = (c2-c1) / (2*h);
    end
end
for k=1:length(W)
    for i=1:numel(W{k})
        W_try = W;
        W_try{k}(i) = W_try{k}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
        
        W_try = W;
        W_try{k}(i) = W_try{k}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
        
        grad_W{k}(i) = (c2-c1) / (2*h);
    end
end
end

%FAST
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = {zeros(size(W{1})),zeros(size(W{2}))};
grad_b = {zeros(size(W{1}, 1), 1),zeros(size(W{2}, 1), 1)};

c = ComputeCost(X, Y, W, b, lambda);

for k=1:length(b)
    for i=1:length(b{k})
        b_try = b;
        b_try{k}(i) = b_try{k}(i) + h;
        
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b{k}(i) = (c2-c) / h;
    end
end

for k=1:length(W)
    for i=1:numel(W{k})
        W_try = W;
        W_try{k}(i) = W_try{k}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
        grad_W{k}(i) = (c2-c) / h;
    end
end
end

