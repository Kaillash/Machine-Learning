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

% X = [ones(size(X, 1), 1) X];
% 
% yi = zeros(num_labels, 1);
% for i = 1:m
% 
% 	a1 = X(i,:)';
% 	a2 = sigmoid(Theta1 * a1);
% 
% 	% h(x)
% 	a3 = sigmoid(Theta2 * [1; a2]);
% 	
% 	yi(y(i)) = 1;
% 	J = J + sum(- yi .* log(a3) - (1 - yi) .* log(1 - a3));
% 	yi(y(i)) = 0;
% 
% endfor
% 
% J = (1 / m) * J;

% Input layer
a1 = [ones(size(X, 1), 1) X];
% size(a1) = m x (input_layer_size + 1) = 5000 x 401

% Hidden layer
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
% size(a2) = m x (hidden_layer_size + 1) = 5000 x 26

z3 = a2 * Theta2';
a3 = sigmoid(z3); % h(x)
% size(a3) = m x num_labels = 5000 x 10

yVec = zeros(m, num_labels);
for i = 1:m
	yVec(i,y(i)) = 1;
endfor

J = (1 / m) * sum(sum((- yVec .* log(a3) - (1 - yVec) .* log(1 - a3)), 2));

regularization = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2, 2)) + sum(sum(Theta2(:, 2:end) .^ 2, 2)));

J = J + regularization;

% -------------------------------------------------------------
% The backpropagation algorithm
% -------------------------------------------------------------

%% Part 2 implementation, as per pdf guide
% for t = 1:m
% 
% 	% Input layer
% 	a1 = [1; X(t,:)'];
% 
% 	% Hidden layer
% 	z2 = Theta1 * a1;
% 	a2 = [1; sigmoid(z2)];
% 
% 	z3 = Theta2 * a2;
% 	a3 = sigmoid(z3);
% 
% 	yy = ([1:num_labels]==y(t))';
% 
% 	% Compute deltas
% 	delta3 = a3 - yy;
% 
% 	delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];
% 	delta2 = delta2(2:end); % Taking of the bias row
% 
% 	% No need to calculate delta1, because we do not associate error with the input
% 
% 	% Update big deltas
% 	Theta1_grad = Theta1_grad + delta2 * a1';
% 	Theta2_grad = Theta2_grad + delta3 * a2';
% end
% 
% Theta1_grad = (1/m) * Theta1_grad;
% Theta2_grad = (1/m) * Theta2_grad;

% Compute deltas
delta3 = (a3 - yVec);
% size(delta3) = m x num_labels = 5000 x 10

delta2 = (delta3 * Theta2) .* [ones(m, 1) sigmoidGradient(z2)];
% size(delta2) = m x (hidden_layer_size + 1) = 5000 x 26

% Remove the bias element(s)
delta2 = delta2(:, 2:end);
% size(delta2) = m x hidden_layer_size = 5000 x 25

% No need to calculate delta1, because we do not associate error with the input

% Update big deltas
Theta1_grad = (1 / m) * (Theta1_grad + delta2' * a1);
% size(Theta1_grad) = size(Theta1) = hidden_layer_size, (input_layer_size + 1) = 25 x 401;
% size(delta2) = m x hidden_layer_size = 5000 x 25
% size(a1) = m x (input_layer_size + 1) = 5000 x 401

Theta2_grad = (1 / m) * (Theta2_grad + delta3' * a2);
% size(Theta2_grad) = size(Theta2) = num_labels x (hidden_layer_size + 1) = 10 x 26;
% size(delta3) = m x num_labels = 5000 x 10
% size(a2) = m x hidden_layer_size = 5000 x 26

% Add regularization to the gradient
Theta1_grad = Theta1_grad + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad = Theta2_grad + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
