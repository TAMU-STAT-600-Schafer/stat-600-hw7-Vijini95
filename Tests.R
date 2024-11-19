# Source the NN function
source("FunctionsNN.R")

# [ToDo] Source the functions from HW3 (replace FunctionsLR.R with your working code)
source("FunctionsLR.R")

set.seed(123)
n_samples <- 100 # Number of samples per class
p <- 2 # Number of features (excluding intercept)

# Generate data for class 0
X0 <- matrix(rnorm(n_samples * p, mean = -2, sd = 1), n_samples, p)

# Generate data for class 1
X1 <- matrix(rnorm(n_samples * p, mean = 2, sd = 1), n_samples, p)
X <- rbind(X0, X1) # Combine the data
# Add intercept term
X <- cbind(1, X)  # Now X has p + 1 columns 
y <- c(rep(0, n_samples), rep(1, n_samples)) # Create labels

# Split data into training and testing sets
set.seed(456)  # Different seed for splitting
train_indices <- sample(1:(2 * n_samples), size = n_samples)
test_indices <- setdiff(1:(2 * n_samples), train_indices)

X_train <- X[train_indices, ]
y_train <- y[train_indices]

X_test <- X[test_indices, ]
y_test <- y[test_indices]

result1 <- LRMultiClass(X_train, y_train, X_test, y_test, numIter = 10, eta = 0.1, lambda = 1)
result2 <- LRMultiClass(X_train, y_train, X_test, y_test, numIter = 50, eta = 0.1, lambda = 1)

#objective value = 2.275099
plot(result1$objective, type = 'o')

plot(result1$objective, type = 'o')
plot(result1$error_train, type = 'o') 
plot(result1$error_test, type = 'o') 

result1$error_train #19.77778
result1$error_test  #25.23333
id_val = 100:200
Yval = y[id_val]
Xval = X[id_val, ]
Ytrain = y[-id_val]
Xtrain = X[-id_val, ]

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
plot(1:length(out2$error), out2$error, ylim = c(0, 70))
lines(1:length(out2$error_val), out2$error_val, col = "red")

# Evaluate error on testing data
test_error = evaluate_error(X_test,
                            y_test,
                            out2$params$W1,
                            out2$params$b1,
                            out2$params$W2,
                            out2$params$b2)
test_error
