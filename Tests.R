# Source the NN function
source("FunctionsNN.R")

# [ToDo] Source the functions from HW3 (replace FunctionsLR.R with your working code)
source("FunctionsLR.R")

# Generate a synthetic multi-class dataset
set.seed(123)

# Number of samples per class
n_samples <- 100

# Generate data for Class 0
mean0 <- c(2, 2)
cov0 <- matrix(c(1, 0.3, 0.3, 1), nrow = 2)
X0 <- MASS::mvrnorm(n_samples, mu = mean0, Sigma = cov0)
y0 <- rep(0, n_samples)

# Generate data for Class 1
mean1 <- c(-2, -2)
cov1 <- matrix(c(1, -0.2, -0.2, 1), nrow = 2)
X1 <- MASS::mvrnorm(n_samples, mu = mean1, Sigma = cov1)
y1 <- rep(1, n_samples)

# Generate data for Class 2
mean2 <- c(0, 5)
cov2 <- matrix(c(1, 0, 0, 1), nrow = 2)
X2 <- MASS::mvrnorm(n_samples, mu = mean2, Sigma = cov2)
y2 <- rep(2, n_samples)

# Combine the data
X <- rbind(X0, X1, X2)
y <- c(y0, y1, y2)

# Shuffle the dataset
shuffle_idx <- sample(1:(3 * n_samples))
X <- X[shuffle_idx, ]
y <- y[shuffle_idx]

# Split into train, validation, and test sets
train_idx <- 1:(0.7 * nrow(X))
val_idx <- (0.7 * nrow(X) + 1):(0.85 * nrow(X))
test_idx <- (0.85 * nrow(X) + 1):nrow(X)

Xtrain <- X[train_idx, ]
Ytrain <- y[train_idx]
Xval <- X[val_idx, ]
Yval <- y[val_idx]
Xtest <- X[test_idx, ]
Ytest <- y[test_idx]
# Train the neural network
out21 = NN_train(
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
plot(1:length(out21$error), out21$error, ylim = c(0, 70))
lines(1:length(out21$error_val), out21$error_val, col = "red")

#########################################
# Set seed for reproducibility
set.seed(42)

# Number of samples per class
n_samples <- 50

# Generate data for Class 0
X0 <- cbind(rnorm(n_samples, mean = 2, sd = 0.5),
            rnorm(n_samples, mean = 2, sd = 0.5))
y0 <- rep(0, n_samples)

# Generate data for Class 1
X1 <- cbind(rnorm(n_samples, mean = -2, sd = 0.5),
            rnorm(n_samples, mean = -2, sd = 0.5))
y1 <- rep(1, n_samples)

# Combine the data
X <- rbind(X0, X1)
y <- c(y0, y1)

# Shuffle the dataset
shuffle_idx <- sample(1:(2 * n_samples))
X <- X[shuffle_idx, ]
y <- y[shuffle_idx]

# Split the data into training, validation, and test sets
set.seed(123)
train_idx <- 1:(0.7 * nrow(X))
val_idx <- (0.7 * nrow(X) + 1):(0.85 * nrow(X))
test_idx <- (0.85 * nrow(X) + 1):nrow(X)

Xtrain <- X[train_idx, ]
Ytrain <- y[train_idx]
Xval <- X[val_idx, ]
Yval <- y[val_idx]
Xtest <- X[test_idx, ]
Ytest <- y[test_idx]

# Train the neural network
out22 <- NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.001,
  rate = 0.1,
  mbatch = 10,
  nEpoch = 50,
  hidden_p = 10,
  scale = 1e-3,
  seed = 12345
)
plot(1:length(out22$error), out22$error, ylim = c(0, 70))
lines(1:length(out22$error_val), out22$error_val, col = "red")
