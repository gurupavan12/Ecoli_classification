# Ecoli_classification

# Project 1 Naive Bayes and Logistic Regression

```
In this project you will code up two of the classification algorithms covered in class: Naive Bayes and Logistic
Regression. The framework code for this question can be downloaded from CANVAS.
```
- Programming Language: You must write your code in R.

- SUBMISSION CHECKLIST
    - Submission executes in less than 20 minutes.
    - Submission is smaller than 100K.
    - Submission is a .tar file.
    - Submission returns matrices of the exact dimension specified.
- Data:All questions will use the following datastructures:
    - xTrain∈Rn×fis a matrix of training data, where each row is a training point, and each column
       is a feature.
    - xTest∈Rm×fis amatrix of test data, where each row is a test point, and each column is a
       feature.
    - yTrain∈{ 1 ,...,c}n×^1 is avector of training labels
    - yTest∈{ 1 ,...,c}m×^1 is a (hidden) vector of test labels.

## 1 Logspace Arithmetic [10 pts]

When working with very small and very large numbers (such as probabilities), it is useful to work in logspace
to avoid numerical precision issues. In logspace, we keep track of the logs of numbers, instead of the
numbers themselves. (We generally use natural logs for this). For example, if p(x) and p(y) are proba-
bility values, instead of storing p(x) and p(y) and computing p(x) ∗ p(y), we work in log space by storing
log p(x), log p(y), log[p(x) ∗ p(y)], where log[p(x) ∗ p(y)] is computed as log p(x) + log p(y).
The challenge is to add and multiply these numbers while remaining in logspace, without exponentiating.
Note that if we exponentiate our numbers at any point in the calculation it completely defeats the purpose
of working in log space.

```
1.Logspace Multiplication [5 pts]
Complete logProd=function(x) which takes as input a vector of numbers in logspace (i.e., xi = log pi),
and returns the product of these numbers in logspace – i.e.,logProd(x)= log∏_i pi.

2.Logspace Addition [5 pts]
Complete logSum=function(x) which takes as input a vector of numbers in logspace (i.e., xi = log pi),
and returns the sum of these numbers in logspace – i.e.,logSum(x)= log ∑_i pi.
```


## 2 Gaussian Naive Bayes [25 pts]

You will implement the Gaussian Naive Bayes Classification∏ algorithm. As a reminder, in the Naive Bayes algorithm we calculate p(c|f) ∝ p(f|c)p(c) = p(c) (^) i p(fi|c). In Gaussian Naive Bayes, we learn a one-dimensional Gaussian for each feature in each class, i .e. p(fi|c) = N(fi; μi,c, σ^2 i,c), where μi,c is the mean of feature fi for those instances in class c, and σi,c^2 is the variance of feature fi for instances in class c. You can ( and should) test your implementation locally using the x Train and y Train data provided.
```
1.Training Model - Learning Class Priors [5 pts]
Complete the function prior=function(yTrain). It returns a c × 1 vector p, where pi is the prior
probability of class i.
```
```
2.Training Model - Learning Class-Conditional Feature Probabilities [8 pts]
Complete the function likelihood=function(xTrain, yTrain). It returns two matrices, M and V. M
is an m × c matrix where Mi,j is the conditional mean of feature i given class j. V is an m × c
matrix where Vi,j is the conditional variance of feature i given class j.
```
```
3.Naive B ayes Classifier [ 8 pts]
Complete the function naiveBayesClassify=function(xTest, M, V, p). It returns a vector t, which is a
m × 1 vector of predicted class values, where ti is the predicted class for the ith row of xTest.
```
```
4.Evaluation [4 pts]
Let’s analyze the accuracy of the classifier on the test data. Create a text file evaluation.txt. Each on a
separate line, report the evaluation metric in decimal format, to 3 decimal places.

- Fraction of test samples classified correctly
- Precision for class 1
- Recall for class 1
- Precision for class 5
- Recall for class 5
```
## 3 Logistic Regression [25 pts]


In this question you will implement the Logistic Regression algorithm. You will learn the weights using
Gradient Descent. Once again you can test your implementation locally using the xT rain and yT rain data
provided.

```
1.Sigmoid Probability [ 7 pts]
Complete the function sigmoidProb = function(y, x, w), where y ∈ 0 , 1 i s a single class, x i s a single
training example, w i s a weights vector. The function returns a value p = p(y|x).
```
```
2.Training Logistic Regression [ 7 pts]
Complete the function logisticRegressionWeights=function(xTrain, yTrain, w0, nIter), where w0 is the
initial weight value and nIter is the number of times to pass through the dataset. It outputs a
G ̈  weights vector w. You can use step-size=0.1 in this question.
```
```
3.Logistic Regression Classifier [ 7 pts]
Complete the function logisticRegressionClassify=function(xTest, w), where w is a f × 1 weights
vector. The output should be a single binary value indicating which class you predict.
```
```
4.Evaluation [4 pts]
Evaluate the accuracy of the classifier on the Ecoli dataset as in Question 2. Report your results in the file
evaluation.txt, compare with the results from Question 2, and comment on the comparison.
```



