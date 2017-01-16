function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

regShift = (lambda/(2*m))*sum(theta(2:end).^2);
J  = sum((X*theta - y).^2)/(2*m) + regShift; 

gradSize = size(theta);
grad = zeros(gradSize);

for i=1:gradSize
    for j=1:m 
        jX = X(j,:);
        jY = y(j,:);
        jH = sum(jX.*theta') - jY;
        ijX = X(j,i);
        jC = jH'*ijX; 
        c=sum(jC);
        grad(i) = grad(i) + c; 
    end;
    grad(i) = grad(i)/m + lambda*theta(i)/m;
end;

%vectorized version
%grad = 1 / m * ((X * theta - y)' * X)' + lambda / m * (theta);
grad(1) = grad(1) - lambda*theta(1)/m;

grad = grad(:);

end
