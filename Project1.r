##################################
###### STAT 557 (Project 1) ######
##################################

rm(list=ls()) ## To clear your environment

## Read the data
xTrain=read.csv("ecoli_xTrain.csv",header=FALSE)
yTrain=read.csv("ecoli_yTrain.csv",header=FALSE)
xTest=read.csv("ecoli_xTest.csv",header=FALSE)
yTest=read.csv("ecoli_yTest.csv",header=FALSE)


#### Part 1 ####
logProd <- function(x){
  return(sum(x))
}

logSum <- function(x){
  a = max(x)
  argument = sum(exp(x-a))
  total_sum = a+log(argument)
  return(total_sum)
}

#### Part 2 ####
prior <- function(yTrain){
  total=nrow(yTrain)
  total_classes = nrow(unique(yTrain))
  res=c()
  #calculating prior for each class
  for (c in 1:total_classes){
    res1=sum(yTrain==c)/total
    res=rbind(res,res1)
  }
  return(res)
}


likelihood <- function(xTrain, yTrain){
  dat = cbind(xTrain, yTrain)
  
  total_classes = nrow(unique(yTrain))
  total_features = ncol(xTrain)
  colnames(dat) = c("X1","X2","X3","X4","X5","Y")
  
  mean=c()
  variances=c()
  
  for (i in 1:total_classes){
    dat1= dat[dat$Y==i,1:total_features] #Obtaining the necessary data
    mean=rbind(mean,colMeans(dat1)) #Calculating mean
    
    #Calculating variance 
    cols=1
    prow_variances=c()
    while (cols<=total_features){
      prow_variances=cbind(prow_variances,(sd(dat1[,cols])^2))
      cols=cols+1
    }
    variances = rbind(variances,prow_variances)
  }
  return(list("M"=mean,"V"=variances))
}

naiveBayesClassify <- function(xTest, M, V, p){
  total_test=nrow(xTest)
  total_classes=nrow(M)
  
  predicted=c()
  for (i in 1:total_test){
    dat1= xTest[i,]
    mat=c()
    for (row in 1:total_classes){
      lhs = -0.5*( log(2*pi*V[row,]) + ((dat1-M[row,])^2)/V[row,])
      mat=rbind(mat,sum(lhs))
    }
    
    argument = log(p)+mat
    predicted = rbind(predicted,which.max(argument))
  }
  return(predicted)
}

evaluation <- function(yPred,yTest)
  {
  yPred=data.frame(yPred)
  y=cbind(yPred,yTest)
  colnames(y)=c('yPred','yTest')
  conf_mat = table(y$yPred,y$yTest)
  
  cat("Confusion matrix is as follows, with the rows as predicted \n",
      "outcomes and the cols as true values \n")
  print(conf_mat)
  
  accuracy = sum(diag(conf_mat))/sum(conf_mat)
  prec_class = diag(conf_mat)/rowSums(conf_mat)
  recall_class = diag(conf_mat)/colSums(conf_mat)
  
  sink("evaluation.txt",append=TRUE)
  
  cat("Naive Bayes Classifier \n")
  cat("Accuracy is = ",round(accuracy*100,digit=3),"%\n")
  cat("Precision for class 1 = ",round(prec_class[1]*100,digit=3),"%\n")
  cat("Precision for class 5 = ",round(prec_class[5]*100,digit=3),"%\n")
  cat("Recall for class 1 = ",round(recall_class[1]*100,digit=3),"%\n")
  cat("Recall for class 5 = ",round(recall_class[5]*100,digit=3),"%\n")
  sink()
}

# Driver to run the part 2. Comment this if passing the file through a script
p = prior(yTrain)
a=likelihood(xTrain,yTrain)
M=a$M
V=a$V
yPred = naiveBayesClassify(xTest,M,V,p)
evaluation(yPred,yTest)


#### Part 3 ####

xTrain1=read.csv("ecoli_new.xTrain.csv",header=FALSE)
yTrain1=read.csv("ecoli_new.yTrain.csv",header=FALSE)
xTest1=read.csv("ecoli_new.xTest.csv",header=FALSE)
yTest1=read.csv("ecoli_new.yTest.csv",header=FALSE)

# Calculating p(y|x) 
sigmoidProb <- function(y,x,w){
  input=sum(x*w)
  p=1/(1+exp(-input))
  return(p)
}


logisticRegressionWeights <- function(xTrain, yTrain, w0, nIter){
  n=nrow(xTrain)
  num_features=ncol(xTrain)
  w=w0;
  
  # Learning the weights using gradient descent
  for (i in 1:nIter)
  {
    z=matrix(0,nrow=num_features,ncol=1)
    for (j in 1:n)
    {
      z=z+((sigmoidProb(yTrain[j,],xTrain[j,],w)-yTrain[j,1])*xTrain[j,])
    }
    w=w-0.1*z
  }
  return (w)
}

logisticRegressionClassify <- function(xTest, w){
  yPred=matrix(0,nrow=109,ncol=1)
  ptest=0
  n1=nrow(xTest)
  
  #Evaluating P(y|x) for test data
  
  for (i in 1:n1){
    ptest[i]=sigmoidProb(1,xTest[i,],w)
    if (ptest[i]>=0.5){
      yPred[i,1]=1
    }
    else{
      yPred[i,1]=0
    }
  }
  return(yPred)
}

evaluationLR <- function(yPred,yTest){
  y=cbind(yPred,yTest)
  conf_mat = table(y[,1],y[,2])
  
  cat("Confusion matrix is as follows, with the rows as predicted \n",
      "outcomes and the cols as true values \n")
  print(conf_mat)
  
  accuracy = sum(diag(conf_mat))/sum(conf_mat)
  prec_class = diag(conf_mat)/rowSums(conf_mat)
  recall_class = diag(conf_mat)/colSums(conf_mat)
  sink("evaluation.txt",append=TRUE)
  
  cat("\nLogistic Regression Classifier \n")
  cat("Accuracy is = ",round(accuracy*100,digit=3),"%\n")
  cat("Precision for class 0 = ",round(prec_class[1]*100,digit=3),"%\n")
  cat("Precision for class 1 = ",round(prec_class[2]*100,digit=3),"%\n")
  cat("Recall for class 0 = ",round(recall_class[1]*100,digit=3),"%\n")
  cat("Recall for class 1 = ",round(recall_class[2]*100,digit=3),"%\n")
  sink()
}

# Driver to run the part 3. Comment this if passing the file through a script
w0=matrix(0,nrow=1,ncol=1)
weights=logisticRegressionWeights(xTrain1,yTrain1,w0,100)
yPred=logisticRegressionClassify(xTest1,weights)
evaluationLR(yPred,yTest1)

# Evaluation.txt for comparing naive bayes with two class
evaluation2ClassNB <- function(yPred,yTest)
{
  yPred=data.frame(yPred)
  y=cbind(yPred,yTest)
  colnames(y)=c('yPred','yTest')
  conf_mat = table(y$yPred,y$yTest)
  
  cat("Confusion matrix is as follows, with the rows as predicted \n",
      "outcomes and the cols as true values \n")
  print(conf_mat)
  
  accuracy = sum(diag(conf_mat))/sum(conf_mat)
  prec_class = diag(conf_mat)/rowSums(conf_mat)
  recall_class = diag(conf_mat)/colSums(conf_mat)
  
  sink("evaluation.txt",append=TRUE)
  
  cat("\nNaive Bayes Classifier for 2 Class Case \n")
  cat("Accuracy is = ",round(accuracy*100,digit=3),"%\n")
  cat("Precision for class 0 = ",round(prec_class[1]*100,digit=3),"%\n")
  cat("Precision for class 1 = ",round(prec_class[2]*100,digit=3),"%\n")
  cat("Recall for class 0 = ",round(recall_class[1]*100,digit=3),"%\n")
  cat("Recall for class 1= ",round(recall_class[2]*100,digit=3),"%\n")
  sink()
}


# Driver to run Naive Bayes Classifier for 2 class case.
# Replacing Class labels to estimate class prior since the function was designed to incorporate class labels
#  1-to-total_class labels. 
yTrain1$V1[yTrain1$V1==1]<-2
yTrain1$V1[yTrain1$V1==0]<-1


p = prior(yTrain1)
a=likelihood(xTrain1[,2:6],yTrain1)
M=a$M
V=a$V
yPred1 = naiveBayesClassify(xTest1[,2:6],M,V,p)
evaluation2ClassNB(yPred1,yTest1)


