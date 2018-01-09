function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Adding the missing bias units to the training examples
% The bias units are X0, and are equal to 1
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

total_delta2 = zeros(num_labels, hidden_layer_size + 1);
total_delta1 = zeros(hidden_layer_size, input_layer_size + 1);
K = num_labels;
for i=1:m
    
    % Doing Forward propagation to find h_theta
    a1 = X(i, :)'; % a1 is (input_layer_size + 1) x 1     
    z2 = Theta1 * a1; % Theta1 is hidden_layer_size x (input_layer_size + 1), z2 is hidden_layer_size x 1
    a2 = sigmoid(z2); % a2 is hidden_layer_size x 1 
    a2 = [1 ; a2]; % Adding a0(2), a2 is now (hidden_layer_size + 1) x 1 
    z3 = Theta2 * a2; % Theta2 is num_labels x (hidden_layer_size + 1), z3 is num_labels x 1
    h_theta = sigmoid(z3);  % h_theta is num_labels x 1
    
    % Constructing the y vector for the ith training example, it'll be num_labels x 1 
    training_ex_result = y(i);
    y_i_vec = zeros(K, 1);
    y_i_vec(training_ex_result) = 1;
    
    % Calculating Cost Function
    J = J + sum((-y_i_vec .* log(h_theta)) - ((1 - y_i_vec) .* log(1 - h_theta)));
    
    % Doing backpropagation 
    delta3 = h_theta - y_i_vec; % both are num_labels x 1
    delta2 = (Theta2' * delta3); % Theta2' is (hidden_layer_size + 1) x num_labels, delta3 is num_labels x 1
    delta2 = delta2(2:end); % delta2 is hidden_layer_size x 1
    delta2 = delta2 .* sigmoidGradient(z2); % z2 is num_labels x 1
    
    % Accumulating the total deltas
    total_delta2 = total_delta2 + (delta3 * a2'); % delta3 is num_labels x 1, a2' is 1 x (hidden_layer_size + 1)
    total_delta1 = total_delta1 + (delta2 * a1'); % delta2 is hidden_layer_size x 1, a1' is 1 x (input_layer_size + 1)
end

% Finishing last part of Cost Function => dividing my m
J = J * (1 / m);

% Adding Regularization to Cost Function
regularization = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));
J = J + regularization;

% Calculating gradients
grad1_lambda_part = (lambda / m) .* [zeros(hidden_layer_size, 1) Theta1(:, 2:end)];
grad2_lambda_part = (lambda / m) .* [zeros(num_labels, 1) Theta2(:, 2:end)];
Theta1_grad = (total_delta1 ./ m) + grad1_lambda_part;
Theta2_grad = (total_delta2 ./ m) + grad2_lambda_part;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
