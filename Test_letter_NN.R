# Load the data

# Training data
letter_train <- read.table("Data/letter-train.txt",
                           header = F,
                           colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])

# Update training to set last part as validation
id_val = 1801:2000
Yval = Y[id_val]
Xval = X[id_val, ]
Ytrain = Y[-id_val]
Xtrain = X[-id_val, ]

# Testing data
letter_test <- read.table("Data/letter-test.txt",
                          header = F,
                          colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])

# Source the NN function
source("FunctionsNN.R")

# [ToDo] Source the functions from HW3 (replace FunctionsLR.R with your working code)
source("FunctionsLR.R")

# Recall the results of linear classifier from HW3
# Add intercept column
Xinter <- cbind(rep(1, nrow(Xtrain)), Xtrain)
Xtinter <- cbind(rep(1, nrow(Xt)), Xt)

#  Apply LR (note that here lambda is not on the same scale as in NN due to scaling by training size)
out <- LRMultiClass(
  Xinter,
  Ytrain,
  Xtinter,
  Yt,
  lambda = 1,
  numIter = 150,
  eta = 0.1
)
plot(out$objective, type = 'o')
plot(out$error_train, type = 'o') # around 19.5 if keep training
plot(out$error_test, type = 'o') # around 25 if keep training

out$error_train #19.77778
out$error_test  #25.23333

# Apply neural network training with default given parameters
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
test_error = evaluate_error(Xt,
                            Yt,
                            out2$params$W1,
                            out2$params$b1,
                            out2$params$W2,
                            out2$params$b2)
test_error # 16.1 #I obtained 15.68333

# Evaluate error on training data
training_error = evaluate_error(X,
                                Y,
                                out2$params$W1,
                                out2$params$b1,
                                out2$params$W2,
                                out2$params$b2)
training_error #I obtained 6.45

library(microbenchmark)
timings <- microbenchmark(out <- LRMultiClass(Xinter, Ytrain, Xtinter, Yt, lambda = 1, numIter = 150, eta = 0.1),
  out2 <- NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001, rate = 0.1, mbatch = 50, nEpoch = 150, hidden_p = 100, scale = 1e-3, seed = 12345), times = 10)
timings

# [ToDo] Try changing the parameters above to obtain a better performance,
# this will likely take several trials
#set lambda = 0.01
out3 = NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.01,
  rate = 0.1,
  mbatch = 50,
  nEpoch = 150,
  hidden_p = 100,
  scale = 1e-3,
  seed = 12345
)
test_error1 = evaluate_error(Xt,
                             Yt,
                             out3$params$W1,
                             out3$params$b1,
                             out3$params$W2,
                             out3$params$b2)
test_error1 #28.45

#set lambda = 0.0001
out4 = NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.0001,
  rate = 0.1,
  mbatch = 50,
  nEpoch = 150,
  hidden_p = 100,
  scale = 1e-3,
  seed = 12345
)
test_error2 = evaluate_error(Xt,
                             Yt,
                             out4$params$W1,
                             out4$params$b1,
                             out4$params$W2,
                             out4$params$b2)
test_error2 #14.56667 (when lambda is small, error is small)

#set rate = 0.01, lambda = 0.0001
out5 = NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.0001,
  rate = 0.01,
  mbatch = 50,
  nEpoch = 150,
  hidden_p = 100,
  scale = 1e-3,
  seed = 12345
)
test_error3 = evaluate_error(Xt,
                             Yt,
                             out5$params$W1,
                             out5$params$b1,
                             out5$params$W2,
                             out5$params$b2)
test_error3 #27.44444

#set rate = 0.5, lambda = 0.001
out6 = NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.001,
  rate = 0.5,
  mbatch = 50,
  nEpoch = 150,
  hidden_p = 100,
  scale = 1e-3,
  seed = 12345
)
test_error4 = evaluate_error(Xt,
                             Yt,
                             out6$params$W1,
                             out6$params$b1,
                             out6$params$W2,
                             out6$params$b2)
test_error4 #96.37778

#set rate = 0.11, lambda = 0.001
out7 = NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.001,
  rate = 0.11,
  mbatch = 50,
  nEpoch = 150,
  hidden_p = 100,
  scale = 1e-3,
  seed = 12345
)
test_error5 = evaluate_error(Xt,
                             Yt,
                             out7$params$W1,
                             out7$params$b1,
                             out7$params$W2,
                             out7$params$b2)
test_error5 #17.11667 (If rate is different from 0.1, then error is getting large)

#set mbatch = 100
out8 = NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.001,
  rate = 0.1,
  mbatch = 100,
  nEpoch = 150,
  hidden_p = 100,
  scale = 1e-3,
  seed = 12345
)
test_error6 = evaluate_error(Xt,
                             Yt,
                             out8$params$W1,
                             out8$params$b1,
                             out8$params$W2,
                             out8$params$b2)
test_error6 #18.70556

#set mbatch = 45
out9 = NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.001,
  rate = 0.1,
  mbatch = 45,
  nEpoch = 150,
  hidden_p = 100,
  scale = 1e-3,
  seed = 12345
)
test_error7 = evaluate_error(Xt,
                             Yt,
                             out9$params$W1,
                             out9$params$b1,
                             out9$params$W2,
                             out9$params$b2)
test_error7 #18.91111 (If mbatch is different from 50, then error is getting large)

#set nEpoch = 300
out10 = NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.001,
  rate = 0.1,
  mbatch = 50,
  nEpoch = 300,
  hidden_p = 100,
  scale = 1e-3,
  seed = 12345
)
test_error8 = evaluate_error(Xt,
                             Yt,
                             out10$params$W1,
                             out10$params$b1,
                             out10$params$W2,
                             out10$params$b2)
test_error8 #29.44444 (If mbatch is different from 50, then error is getting large)

#set nEpoch = 100
out11 = NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.001,
  rate = 0.1,
  mbatch = 50,
  nEpoch = 100,
  hidden_p = 100,
  scale = 1e-3,
  seed = 12345
)
test_error9 = evaluate_error(Xt,
                             Yt,
                             out11$params$W1,
                             out11$params$b1,
                             out11$params$W2,
                             out11$params$b2)
test_error9 #16.85556 (If mbatch is different from 150, then error is getting large)

#set hidden_p = 200
out12 = NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.001,
  rate = 0.1,
  mbatch = 50,
  nEpoch = 150,
  hidden_p = 200,
  scale = 1e-3,
  seed = 12345
)
test_error10 = evaluate_error(Xt,
                              Yt,
                              out12$params$W1,
                              out12$params$b1,
                              out12$params$W2,
                              out12$params$b2)
test_error10 #15.13889 (If mbatch is different from 150, then error is getting large)

#set hidden_p = 1000
out13 = NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.001,
  rate = 0.1,
  mbatch = 50,
  nEpoch = 150,
  hidden_p = 1000,
  scale = 1e-3,
  seed = 12345
)
test_error11 = evaluate_error(Xt,
                              Yt,
                              out13$params$W1,
                              out13$params$b1,
                              out13$params$W2,
                              out13$params$b2)
test_error11 #13.44444 (If hidden_p is increasing, then error is getting small)

#set scale = 1e-2
out14 = NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.001,
  rate = 0.1,
  mbatch = 50,
  nEpoch = 150,
  hidden_p = 150,
  scale = 1e-2,
  seed = 12345
)
test_error12 = evaluate_error(Xt,
                              Yt,
                              out14$params$W1,
                              out14$params$b1,
                              out14$params$W2,
                              out14$params$b2)
test_error12 #15.06111

#set scale = 1e-5
out15 = NN_train(
  Xtrain,
  Ytrain,
  Xval,
  Yval,
  lambda = 0.001,
  rate = 0.1,
  mbatch = 50,
  nEpoch = 150,
  hidden_p = 150,
  scale = 1e-5,
  seed = 12345
)
test_error13 = evaluate_error(Xt,
                              Yt,
                              out15$params$W1,
                              out15$params$b1,
                              out15$params$W2,
                              out15$params$b2)
test_error13 #16.48889 (If scale is increasing, then error is getting small)
