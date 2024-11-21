# Source the NN function
source("FunctionsNN.R")

# [ToDo] Source the functions from HW3 (replace FunctionsLR.R with your working code)
source("FunctionsLR.R")

# Set seed for reproducibility
set.seed(42)

# Number of samples per class
n_samples <- 100

# Generate data for Class 0
mean0 <- c(2, 2)
cov0 <- matrix(c(1, 0, 0, 1), nrow = 2)
X0 <- MASS::mvrnorm(n_samples, mu = mean0, Sigma = cov0)
y0 <- rep(0, n_samples)

# Generate data for Class 1
mean1 <- c(-2, -2)
cov1 <- matrix(c(1, 0, 0, 1), nrow = 2)
X1 <- MASS::mvrnorm(n_samples, mu = mean1, Sigma = cov1)
y1 <- rep(1, n_samples)

# Combine the data
X <- rbind(X0, X1)
X <- cbind(1, X)
y <- c(y0, y1)
train_indices <- sample(1:(2 * n_samples), size = n_samples)
test_indices <- setdiff(1:(2 * n_samples), train_indices)

X_train <- X[train_indices, ]
y_train <- y[train_indices]

X_test <- X[test_indices, ]
y_test <- y[test_indices]
result1 <- LRMultiClass(X_train, y_train, X_test, y_test, numIter = 10, eta = 0.1, lambda = 1)
out2 = NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.001,
  rate = 0.1,
  mbatch = 50,
  nEpoch = 150,
  hidden_p = 100,
  scale = 1e-3,
  seed = 12345
)
out2
