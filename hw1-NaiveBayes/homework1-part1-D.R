
#remove variables
rm(list=ls())

wdat<-read.csv('pima-indians-diabetes.data', header=FALSE)

library(klaR)
library(caret)

bigx<-wdat[,-c(9)] #main feature frame
bigy<-as.factor(wdat[,9]) #main label set as factor
tescore<-array(dim=10)
for (wi in 1:10)
{wtd<-createDataPartition(y=bigy, p=.8, list=FALSE) 

#train on training data
 svm<-svmlight(bigx[wtd,], bigy[wtd], pathsvm="/Users/mengyuxie/Dropbox/CS498-aml/hw1-part1")

#predict on test data, gives a posterior probability?
 labels<-predict(svm, bigx[-wtd,])

#determine the predicted class
 foo<-labels$class
 accuracy=sum(foo==bigy[-wtd])/(sum(foo==bigy[-wtd])+sum(!(foo==bigy[-wtd])))
 tescore[wi]<-accuracy
}
mean(tescore)