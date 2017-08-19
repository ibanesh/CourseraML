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

%my code, L is the total number  of layers, could be used for generalizing
%the num of layers
L = 3;
Theta = cell(L-1,1);
Theta{1} = Theta1;
Theta{2} = Theta2;

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

X = [ones(m,1) X];
z2 = Theta1*X'; %25 x 5000
a2 = [ones(1,m); sigmoid(z2)]; %26x5000
z3 = Theta2*a2;
h = sigmoid(z3)';
 
temp = ones(m,num_labels);
for c = 1:num_labels
    temp(:,c) = (y==c);
end
y = temp;

h_flat = h(:);
y_flat = temp(:);
J = (1.0/m)*(-y_flat'*log(h_flat) -((1-y_flat)'*log(1-h_flat)));

%for i = 1:num_labels
%    J = J + (-(log(h(i,:))*(y==i))-(log(1-h(i,:))*(1-(y==i))));
%end
%J = J/m;

%part 3
% if using sumsqr(Theta), it iwll also include the first column in theta
% which coressponds to the bias term
reg = (lambda/(2*m))*(sumsqr(Theta1(:,2:end))+sumsqr(Theta2(:,2:end)));
J = J + reg;


%part 2 - back propagation 

% for r=1:m
%     del3 = h(r,:)' - y(r,:)'; % 10x1
%     del2 = Theta2'*del3.*[1; sigmoidGradient(z2(:,r))]; %26x1
%     del2 = del2(2:end); %25x1
%     Theta2_grad = Theta2_grad + del3*a2(:,r)';
%     Theta1_grad = Theta1_grad + del2*X(r,:);
% end

%vectorized implementation
del3 = h-y ;  %5000x10
del2 = (del3*Theta2).*a2'.*(1-a2'); %5000x26
del2 = del2(:,2:end); %5000x25


% for r = 1:m
%    Theta2_grad = Theta2_grad + del3(r,:)'*a2(:,r)';
%    Theta1_grad = Theta1_grad + del2(r,:)'*X(r,:);
% end
% 
% Theta1_grad = (1/m)*(Theta1_grad+(lambda*[zeros(size(Theta1,1),1) Theta1(:,2:end)]));
% Theta2_grad = (1/m)*(Theta2_grad+(lambda*[zeros(size(Theta2,1),1) Theta2(:,2:end)]));

Theta2_grad = (1/m)*(a2*del3)';
Theta1_grad = (1/m)*(del2'*X);
%adding regularization term
Theta1_grad = Theta1_grad + (1/m)*(lambda*[zeros(size(Theta1,1),1) Theta1(:,2:end)]);
Theta2_grad = Theta2_grad + (1/m)*(lambda*[zeros(size(Theta2,1),1) Theta2(:,2:end)]);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
