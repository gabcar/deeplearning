function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = [size(W{1}, 1),size(W{2},1)];
d = size(X, 1);
grad_W = cell(1,length(W));
grad_b = cell(1,length(b));

for k=1:length(b)
    grad_W{k} = zeros(size(W));
    grad_b{k} = zeros(no(k), 1);
    
    for i=1:length(b{k})
        b_try = b{k};
        b_try(i) = b_try(i) - h;
        c1 = ComputeCost(X, Y, W{k}, b_try, lambda);
        b_try = b{k};
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W{k}, b_try, lambda);
        grad_b{k}(i) = (c2-c1) / (2*h);
    end
    
    for i=1:numel(W)
        
        W_try = W{k};
        W_try(i) = W_try(i) - h;
        c1 = ComputeCost(X, Y, W_try, b{k}, lambda);
        
        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b{k}, lambda);
        
        grad_W(i) = (c2-c1) / (2*h);
    end
end

