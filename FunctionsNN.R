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
  b1 <- rep(0, hidden_p)   # Hidden layer biases
  b2 <- rep(0, K)          # Output layer biases
  
  # [ToDo] Initialize weights by drawing them iid from Normal
  # with mean zero and scale as sd
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
  #Create a matrix where each row corresponds to a sample, and columns represent classes.
  n <- length(y)
  y_one_hot <- matrix(0, nrow = n, ncol = K)
  for (i in 1:n) {
    y_one_hot[i, y[i] + 1] <- 1  # R indices start from 1
  }
  
  scores_exp <- exp(scores - apply(scores, 1, max))
  probs <- scores_exp / rowSums(scores_exp)
  
  # [ToDo] Calculate loss when lambda = 0
  loss <- -(1 / n) * sum(y_one_hot * log(probs + 1e-15))  # Add small epsilon to avoid log(0)
  
  # [ToDo] Calculate misclassification error rate (%)
  # when predicting class labels using scores versus true y
  predicted_classes <- apply(scores, 1, which.max) - 1  # Subtract 1 to match class labels from 0
  error <- 100 * mean(predicted_classes != y)
  
  # [ToDo] Calculate gradient of loss with respect to scores (output)
  # when lambda = 0
  grad <- (probs - y_one_hot) / n
  
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
  # [To Do] Forward pass
  # From input to hidden
  Z1 <- X %*% W1 + matrix(b1,
                          nrow = nrow(X),
                          ncol = length(b1),
                          byrow = TRUE)
  
  # ReLU
  A1 <- pmax(0, Z1)
  
  # From hidden to output scores
  scores <- A1 %*% W2 + matrix(b2,
                               nrow = nrow(A1),
                               ncol = length(b2),
                               byrow = TRUE)
  
  
  # [ToDo] Backward pass
  # Get loss, error, gradient at current scores using loss_grad_scores function
  loss_out <- loss_grad_scores(y, scores, K)
  loss <- loss_out$loss
  error <- loss_out$error
  dscores <- loss_out$grad
  
  # Get gradient for 2nd layer W2, b2 (use lambda as needed)
  dW2 <- t(A1) %*% dscores + lambda * W2
  db2 <- colSums(dscores)
  
  # Get gradient for hidden, and 1st layer W1, b1 (use lambda as needed)
  dA1 <- dscores %*% t(W2) #Gradient w.r.t. Hidden Layer Activations
  dZ1 <- dA1
  dZ1[Z1 <= 0] <- 0 #Apply derivative of ReLU
  dW1 <- t(X) %*% dZ1 + lambda * W1
  db1 <- colSums(dZ1) #Gradient w.r.t. W1 and b1
  
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
  # [ToDo] Forward pass to get scores on validation data
  Z1_val <- Xval %*% W1 + matrix(b1,
                                 nrow = nrow(Xval),
                                 ncol = length(b1),
                                 byrow = TRUE) #Compute Hidden Layer Pre-Activation
  A1_val <- pmax(0, Z1_val) #Apply ReLU Activation to Hidden Layer
  scores_val <- A1_val %*% W2 + matrix(b2,
                                       nrow = nrow(A1_val),
                                       ncol = length(b2),
                                       byrow = TRUE) #Compute Output Layer Scores
  
  # [ToDo] Evaluate error rate (in %) when
  # comparing scores-based predictions with true yval
  predicted_classes_val <- apply(scores_val, 1, which.max) - 1  # Adjust for zero-based labels
  error <- 100 * mean(predicted_classes_val != yval)
  
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
  
  # [ToDo] Initialize b1, b2, W1, W2 using initialize_bw with seed as seed,
  # and determine any necessary inputs from supplied ones
  params <- initialize_bw(
    p = ncol(X),
    hidden_p = hidden_p,
    K = length(unique(y)),
    scale = scale,
    seed = seed
  )
  W1 <- params$W1
  b1 <- params$b1
  W2 <- params$W2
  b2 <- params$b2
  
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
    
    # Initialize variables to accumulate loss and error over batches
    total_loss = 0
    total_error = 0
    
    # Loop over each batch
    for (batch in 1:nBatch) {
      # Extract indices for the current batch
      batch_indices = which(batchids == batch)
      
      # Get the mini-batch data
      X_batch = X[batch_indices, , drop = FALSE]
      y_batch = y[batch_indices]
      
      # Perform one pass to get loss, error, and gradients
      out = one_pass(X_batch, y_batch, length(unique(y)), W1, b1, W2, b2, lambda)
      
      # Update the weights and biases using SGD
      W1 = W1 - rate * out$grads$dW1
      b1 = b1 - rate * out$grads$db1
      W2 = W2 - rate * out$grads$dW2
      b2 = b2 - rate * out$grads$db2
      # Accumulate loss and error over batches
      total_loss = total_loss + out$loss
      total_error = total_error + out$error
    }
    # [ToDo] In the end of epoch, evaluate
    # - average training error across batches
    # - validation error using evaluate_error function
    # Compute average training error over all batches in this epoch
    avg_train_error = total_error / nBatch
    error[i] = avg_train_error
    # Evaluate validation error at the end of the epoch
    error_val[i] = evaluate_error(Xval, yval, W1, b1, W2, b2)
    
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
