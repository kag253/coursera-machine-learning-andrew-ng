function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% Calculating the the Cost Function with Regularization
h_theta = X * theta;
error = (1 / (2 * m)) * sum((h_theta - y).^2);
regularization = (lambda / (2 * m)) * sum(theta(2:end, :).^2);
J = error + regularization;



% Calculating Gradient for J=0 (index 1 in MATLAB)
% grad(1) = ((1 / m) .* (h_theta - y))' * X(:, 1);
grad(1) = (1 / m) * ((h_theta - y)' * X(:, 1));

% Calculating Gradient for J=1..n (index 2..n+1 in MATLAB)
%part1 = (((1 / m) .* (h_theta - y))' * X(:, 2:end))';
part1 = (1 / m) .* ((h_theta - y)' * X(:, 2:end))';
part2 = (lambda / m) .* theta(2:end);
grad(2:end) =  part1 + part2;







% =========================================================================

grad = grad(:);

end
