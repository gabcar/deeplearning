%exercise1
clc;clear all;close all;
rng(400)
addpath Datasets/cifar-10-batches-mat/;
tic;

%one dataset
[trainX,trainY,trainy] = LoadBatch('data_batch_1.mat');
[validX,validY,validy] = LoadBatch('data_batch_5.mat');
[testX,testY,testy] = LoadBatch('test_batch.mat');


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

%initialize W and b
W = .01*randn(K,d);
b = .01*randn(K,1);

% %evaluate subsection of data
%P = EvaluateClassifier(trainX, W, b);
% 
%cost = ComputeCost(trainX,trainY,W,b,lambda);
% %5. P argmax
% accuracy = ComputeAccuracy(trainX,trainy,W,b);

%6 control the gradient function
%[ngrad_W, ngrad_b,diff] = ComputeGradients(trainX, trainY, P, W, lambda,b);

%7 minbatch
n_batch = 100;
eta = .01;
n_epochs = 40;
GDparams = [n_batch,eta,n_epochs];
%Mini batch gradient decent
[Wnew, bnew,trainingcost,validatecost] = MiniBatchGD(trainX, trainY, GDparams, ...
    W, b, lambda,validX,validY);

accuracy = ComputeAccuracy(trainX,trainy,Wnew,bnew);
disp(['Training acc: ',num2str(accuracy)])
accuracy = ComputeAccuracy(testX,testy,Wnew,bnew);
disp(['Testing acc: ',num2str(accuracy)])

%plots 
figure()
hold on; grid on;
plot(trainingcost)
plot(validatecost)
xlabel('Epochs');ylabel('cost')
legend('training','validation')
hold off;

disp('done')
for i=1:10
im = reshape(Wnew(i, :), 32, 32, 3);
s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure()
montage(s_im);
load handel
sound(y,Fs)
disp(toc)

%---------------------Functions-------------------------


%function for exercise1
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

%ex1 evaluates classifier
function P = EvaluateClassifier(X,W,b)
s = bsxfun(@plus,W*X,b);
P = softmax(s);
%disp(['size of P: ', num2str(size(P))])
end

%function J that calculates loss function
function J = ComputeCost(X,Y,W,b,lambda)
n = size(X,2);
Wij = sum(sum(W.^2,1),2);
J = (1/n)*sum(lcross(X,Y,W,b))+lambda*Wij;
end

%function for l cross section
function l = lcross(X,Y,W,b)
l = -log(sum(Y.*EvaluateClassifier(X,W,b)));
end

%5. accuracy computation
function acc = ComputeAccuracy(X,y,W,b)
p = EvaluateClassifier(X,W,b);
[~,kstar] = max(p);
check = kstar'==y;
acc = sum(check)/length(check);
end

%gradient comparison function
function [ngrad_W, ngrad_b,diff] = ComputeGradients(X, Y, P, W, lambda,b)
[ngrad_W, ngrad_b] = MyComputeGrads(X(:,1), Y(:,1), ...
    W, b, lambda);
[fastngrad_b,fastngrad_W] = ComputeGradsNumSlow(X(:,1),Y(:,1),W,b,lambda,1e-6);
[slowngrad_b,slowngrad_W] = ComputeGradsNum(X(:,1),Y(:,1),W,b,lambda,1e-6);
diffb1 = max(max(abs(ngrad_b-fastngrad_b)))
diffb2 = max(max(abs(ngrad_b-slowngrad_b)))
diffW1 = max(max(abs(ngrad_W-fastngrad_W)))
diffW2 = max(max(abs(ngrad_W-slowngrad_W)))
diff = diffW1;
diff =0;
end

%my gradient function
function [grad_W,grad_b]=MyComputeGrads(X,Y,W,b,lambda)
N = size(X,2);
P = EvaluateClassifier(X,W,b);
dldb = zeros(size(b));
dldW = zeros(size(W));
for i=1:N
    g = (-Y(:,i)'/(Y(:,i)'*P(:,i)))*(diag(P(:,i))-(P(:,i)*P(:,i)'));
    dldb = dldb + g';
    dldW = dldW + g'*X(:,i)';
end

grad_W = dldW/N + 2*lambda*W;
grad_b = dldb/N;
end

%minibatch
function [Wstar, bstar,traincost,validcost] = MiniBatchGD(Xtrain,...
    Ytrain, GDparams, W, b, lambda,Xvalid,Yvalid)
n_batch = GDparams(1); eta = GDparams(2); n_epochs = GDparams(3);
N = size(Xtrain,2);
traincost = zeros(1,n_epochs);
validcost = traincost;
costtrain = 100000.0;
i = 1; error =1000;
epoch = 1;

for i=1:n_epochs
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        %inds = j_start:j_end;
        Xbatch = Xtrain(:, j_start:j_end);
        Ybatch = Ytrain(:, j_start:j_end);
        [grad_W,grad_b]=MyComputeGrads(Xbatch,Ybatch,W,b,lambda);
        W = W-eta*(grad_W);
        b = b-eta*(grad_b);
    end
    
    %for plotting
    traincost(1,i) = ComputeCost(Xtrain,Ytrain,W,b,lambda);
    validcost(1,i) = ComputeCost(Xvalid,Yvalid,W,b,lambda);
    i = i+1;
    
    prevcost = costtrain;
    costtrain = ComputeCost(Xvalid,Yvalid,W,b,lambda);
    error = (prevcost-costtrain);
    disp(['Loss: ',num2str(costtrain), '   Epoch: ', num2str(epoch)]);
    epoch = epoch+1;
    %eta = eta*0.995;
end
Wstar = W;
bstar = b;
end

%grad quick
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

c = ComputeCost(X, Y, W, b, lambda);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c) / h;
end

for i=1:numel(W)   
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c) / h;
end
end

%grad slow
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end

end
