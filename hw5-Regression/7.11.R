wdat1<-read.csv('abalone.data.txt', header=FALSE)
colnames(wdat1) <- c("Sex","Length","Diameter","Height","Whole.weight","Shucked.weight","Viscera.weight","Shell.weight","Rings")
wdat1$Age<-wdat1$Rings+1.5
wdat1$Rings<-NULL
y<-wdat1$Age

#-----------part A---------#
fit<-lm(Age~.,data=wdat1[,-c(1)])
resid<-fit$residuals
jpeg('7.11a.jpg')
plot(fit$fitted.values,resid,xlab = 'fitted values',ylab = 'residual',main = 'residual vs. fitted gender excluded')
dev.off()
fit$coefficients

#-----------Part B---------#
fitB<-lm(Age~.,data=wdat1)
resid<-fitB$residuals
jpeg('7.11b.jpg')
plot(fitB$fitted.values,resid,xlab = 'fitted values',ylab = 'residual',main = 'residual vs. fitted')
dev.off()
fitB$coefficients

#-----------Part C---------#
fitC <- lm(log(Age)~.,data=wdat1[,-c(1)])
resid <- residuals(fitC)
jpeg('7.11c.jpg')
plot(fitC$fitted.values,resid,xlab = 'fitted values',ylab = 'residual',main = 'residual vs. fitted gender excluded in log coordinate')
dev.off()
predicted <- predict(fitC, type="r")    
resnew<- y-exp(predicted)
plot(exp(fitC$fitted.values),resnew,xlab = 'fitted values',ylab = 'residual',main = 'residual vs. fitted in orignal coordinate')
fitC$coefficients
#-----------Part D---------#
fitD <- lm(log(Age)~.,data=wdat1)
resid <- residuals(fitD)
jpeg('7.11d.jpg')
plot(fitD$fitted.values,resid,xlab = 'fitted values',ylab = 'residual',main = 'residual vs. fitted in log coordinate')
dev.off()
predicted <- predict(fitD, type="r")    
resnew<- y-exp(predicted)
plot(exp(fitD$fitted.values),resnew,xlab = 'fitted values',ylab = 'residual',main = 'residual vs. fitted in orignal coordinate')
fitD$coefficients
#-----------Part E---------#
library(GGally)
wdat2<-wdat1
wdat2$Age<-log(wdat2$Age)
#ggpairs(wdat1, axisLabels = 'external')
#ggpairs(wdat2, axisLabels = 'external')
plot(wdat1,main = 'variable plot')
plot(wdat2,main = 'variable plot vs. log(Age)')
#-----------Part F---------#
library(glmnet)

x<-model.matrix(Age~.,data=wdat1[,-c(1)])
cv <- cv.glmnet(x, y, alpha = 0)
jpeg('7.11f-1.jpg')
plot(cv)
dev.off()

x<-model.matrix(Age~.,data = wdat1)
cv <- cv.glmnet(x, y, alpha = 0)
jpeg('7.11f-2.jpg')
plot(cv)
dev.off()


x<-model.matrix(log(Age)~.,data = wdat1[,-c(1)])
cv <- cv.glmnet(x, log(y), alpha = 0)
jpeg('7.11f-3.jpg')
plot(cv)
dev.off()

x<-model.matrix(log(Age)~.,data = wdat1)
cv <- cv.glmnet(x, log(y), alpha = 0)
jpeg('7.11f-4.jpg')
plot(cv)
dev.off()



