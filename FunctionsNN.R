# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p,
                          hidden_p,
                          K,
                          scale = 1e-3,
                          seed = 12345) {
  # [ToDo] Initialize intercepts as zeros
  b1 <- rep(0, hidden_p)
  b2 <- rep(0, K)
  # [ToDo] Initialize weights by drawing them iid from Normal
  # with mean zero and scale as sd
  set.seed(seed)
  W1 <- matrix(rnorm(p * hidden_p, mean = 0, sd = scale),
               nrow = p,
               ncol = hidden_p) 
  W2 <- matrix(rnorm(hidden_p * K, mean = 0, sd = scale),
               nrow = hidden_p,
               ncol = K) 
  
  # Return
  return(list(
    b1 = b1,
    b2 = b2,
    W1 = W1,
    W2 = W2
  ))
}

# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0
#############################################################
# scores - a matrix of size n by K of scores (output layer)
# y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
loss_grad_scores <- function(y, scores, K) {
  n <- length(y) #dimension
  y <- y + 1
  
  # Compute probabilities using softmax
  exp_scores <- exp(scores)
  sum_exp_scores <- rowSums(exp_scores)
  probs <- exp_scores / sum_exp_scores
  
  Ymat <- 0 + outer(y, 1:K, function(a, b)
    a == b)
  
  # [ToDo] Calculate loss when lambda = 0
  loss <- -sum(Ymat * log(probs)) / n
  
  # [ToDo] Calculate misclassification error rate (%)
  # when predicting class labels using scores versus true y
  error <- 100 * (sum(y != max.col(probs, ties.method = "first")) / n)
  
  # [ToDo] Calculate gradient of loss with respect to scores (output)
  # when lambda = 0
  grad <- (-Ymat + probs) / n #Efficient gradient calculation
  
  # Return loss, gradient and misclassification error on training (in %)
  return(list(
    loss = loss,
    grad = grad,
    error = error
  ))
}

# One pass function
################################################
# X - a matrix of size n by p (input)
# y - a vector of size n of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
# lambda - a non-negative scalar, ridge parameter for gradient calculations
one_pass <- function(X, y, K, W1, b1, W2, b2, lambda) {
  n <- length(y)
  h <- length(b1)
  p <- ncol(X)
  # [To Do] Forward pass
  # From input to hidden
  hidden_input <- X %*% W1 + matrix(b1, n, h, byrow = T)
  
  # ReLU
  hidden_output <- pmax(hidden_input, 0)
  
  # From hidden to output scores
  scores <- hidden_output %*% W2 + matrix(b2, n, K, byrow = T)
  
  # [ToDo] Backward pass
  # Get loss, error, gradient at current scores using loss_grad_scores function
  out <- loss_grad_scores(y, scores, K)
  loss <- out$loss
  error <- out$error
  grad <- out$grad
  
  # Get gradient for 2nd layer W2, b2 (use lambda as needed)
  dW2 <- crossprod(hidden_output, grad) + lambda * W2
  db2 <- colSums(grad)
  # Get gradient for hidden, and 1st layer W1, b1 (use lambda as needed)
  dH <- tcrossprod(grad, W2)
  dA1 <- dH * (hidden_input > 0)
  dW1 <- crossprod(X, dA1) + lambda * W1
  db1 <- colSums(dA1)
  
  # Return output (loss and error from forward pass,
  # list of gradients from backward pass)
  return(list(
    loss = out$loss,
    error = out$error,
    grads = list(
      dW1 = dW1,
      db1 = db1,
      dW2 = dW2,
      db2 = db2
    )
  ))
}

# Function to evaluate validation set error
####################################################
# Xval - a matrix of size nval by p (input)
# yval - a vector of size nval of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
evaluate_error <- function(Xval, yval, W1, b1, W2, b2) {
  nval <- length(yval)
  yval <- yval + 1
  
  # [ToDo] Forward pass to get scores on validation data
  hidden_input <- Xval %*% W1 + matrix(b1, nval, length(b1), byrow = T)
  hidden_output <- pmax(hidden_input, 0)
  scores <- hidden_output %*% W2 + matrix(b2, nval, length(b2), byrow = T)
  
  #P_k calculation
  exp_scores <- exp(scores)
  sum_exp_scores <- rowSums(exp_scores)
  probs <- exp_scores / sum_exp_scores
  
  # [ToDo] Evaluate error rate (in %) when
  # comparing scores-based predictions with true yval
  error <- 100 * (sum(yval != max.col(probs, ties.method = "first")) / nval)
  
  return(error)
}


# Full training
################################################
# X - n by p training data
# y - a vector of size n of class labels, from 0 to K-1
# Xval - nval by p validation data
# yval - a vector of size nval of of class labels, from 0 to K-1, for validation data
# lambda - a non-negative scalar corresponding to ridge parameter
# rate - learning rate for gradient descent
# mbatch - size of the batch for SGD
# nEpoch - total number of epochs for training
# hidden_p - size of hidden layer
# scale - a scalar for weights initialization
# seed - for reproducibility of SGD and initialization
NN_train <- function(X,
                     y,
                     Xval,
                     yval,
                     lambda = 0.01,
                     rate = 0.01,
                     mbatch = 20,
                     nEpoch = 100,
                     hidden_p = 20,
                     scale = 1e-3,
                     seed = 12345) {
  # Get sample size and total number of batches
  n = length(y)
  nBatch = floor(n / mbatch)
  K <- max(y) + 1
  p <- ncol(X)
  # [ToDo] Initialize b1, b2, W1, W2 using initialize_bw with seed as seed,
  # and determine any necessary inputs from supplied ones
  initial <- initialize_bw(p, hidden_p, K, scale, seed)
  b1 <- initial$b1
  b2 <- initial$b2
  W1 <- initial$W1
  W2 <- initial$W2
  # Initialize storage for error to monitor convergence
  error = rep(NA, nEpoch)
  error_val = rep(NA, nEpoch)
  
  # Set seed for reproducibility
  set.seed(seed)
  # Start iterations
  for (i in 1:nEpoch) {
    # Allocate bathes
    batchids = sample(rep(1:nBatch, length.out = n), size = n)
    # [ToDo] For each batch
    #  - do one_pass to determine current error and gradients
    #  - perform SGD step to update the weights and intercepts
    error_now <- 0
    for (j in 1:nBatch) {
      # Get error and gradient on the batch
      first_pass <- one_pass(X[batchids == j, ], y[batchids == j], K, W1, b1, W2, b2, lambda)
      
      # Keep track of error
      error_now <- error_now + first_pass$error
      
      # update of W1, b1, W2, b2
      W1 <- W1 - rate * first_pass$grads$dW1
      W2 <- W2 - rate * first_pass$grads$dW2
      b1 <- b1 - rate * first_pass$grads$db1
      b2 <- b2 - rate * first_pass$grads$db2
    }
    # [ToDo] In the end of epoch, evaluate
    # - average training error across batches
    error[i] <- error_now / nBatch
    # - validation error using evaluate_error function
    error_val[i] <- evaluate_error(Xval, yval, W1, b1, W2, b2)
  }
  # Return end result
  return(list(
    error = error,
    error_val = error_val,
    params =  list(
      W1 = W1,
      b1 = b1,
      W2 = W2,
      b2 = b2
    )
  ))
}