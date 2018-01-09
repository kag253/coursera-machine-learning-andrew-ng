function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% Calculating Cost Function J(theta)
z = X * theta; % X is mxn, theta is nx1
h_theta = sigmoid(z); % h_theta is mx1
part1 = -y .* log(h_theta); % y is mx1
part2 = (1 - y) .* log(1 - h_theta);
part3 = (1/m) * sum(part1 - part2);
part4 = (lambda / (2 * m)) * sum(theta(2:end) .* theta(2:end));
J = part3 + part4;

% Calculating Gradient
% n = size(theta, 1);
% for i=1:n
%     grad(i) = (1 / m) * sum((h_theta - y) .* X(:, i)) 
% end
% Calculating Gradient for J=0 (index 1 in MATLAB)
grad(1) = ((1 / m) .* (h_theta - y))' * X(:, 1);

% Calculating Gradient for J=1..n (index 2..n+1 in MATLAB)
part1 = (((1 / m) .* (h_theta - y))' * X(:, 2:end))';
part2 = (lambda / m) .* theta(2:end);
grad(2:end) =  part1 + part2;

% =============================================================

end
