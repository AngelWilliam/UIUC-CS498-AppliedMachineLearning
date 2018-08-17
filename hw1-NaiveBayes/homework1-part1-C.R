wdat<-read.csv('pima-indians-diabetes.data', header=FALSE)

library(klaR)
library(caret)


bigx<-wdat[,-c(9)]
bigy<-as.factor(wdat[,9])
wtd<-createDataPartition(y=bigy, p=.8, list=FALSE)
trax<-bigx[wtd,] #features on test data
tray<-bigy[wtd] #labels on test data

#model on the traning data
#using grid search
grid <- data.frame(fL=c(0,0.5,1.0), usekernel = TRUE, adjust=c(0,0.5,1.0))

# 10 fold cross-validation with grid search
model<-train(trax, tray, 'nb', tuneGrid=grid,trControl=trainControl(method='cv', number=10))

#test model on 20% test data
teclasses<-predict(model,newdata=bigx[-wtd,])
confusionMatrix(data=teclasses, bigy[-wtd])

