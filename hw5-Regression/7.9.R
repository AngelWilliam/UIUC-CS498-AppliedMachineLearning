data<-read.table('brunhild.txt', sep="\t",header=TRUE)
x<-data$Hours
y<-data$Sulfate
fit <- lm(log(data$Sulfate)~log(data$Hours),data=data)
jpeg('7.9a.jpg')
plot(log(y)~log(x),xlab = 'log(Hours)', ylab = 'log(Sulfate)', main = 'linear regression in log-log coordinate')
#summary(fit)
abline(fit,col="red")
dev.off()

jpeg('7.9b.jpg')
#https://stackoverflow.com/questions/46392683/how-to-plot-transformed-regression-back-on-original-scale
plot(y~x,xlab = 'Hours', ylab = 'Sulfate', main = 'regression in orignal coordinate')
#curve(2.766-0.247*log(x),add=TRUE,col=2)
predicted <- predict(fit, type="r")    
lines(x, exp(predicted), col = "blue")
dev.off()
#par(mfrow = c(1,1))  # Split the plotting panel into a 2 x 2 grid
#plot(fit)


#jpeg('7.9c.jpg')
resid <- residuals(fit)
#plot(log(y),resid,xlab = 'fitted values',ylab = 'residual',main = 'residual vs. fitted in log-log coordinate')
jpeg('7.9c1.jpg')
plot(fit$fitted.values,resid,xlab = 'fitted values',ylab = 'residual',main = 'residual vs. fitted in log-log coordinate')
dev.off()

resnew<- y-exp(predicted)
jpeg('7.9c2.jpg')
plot(exp(fit$fitted.values),resnew,xlab = 'fitted values',ylab = 'residual',main = 'residual vs. fitted in orignal coordinate')
#https://onlinecourses.science.psu.edu/stat501/node/279
dev.off()
