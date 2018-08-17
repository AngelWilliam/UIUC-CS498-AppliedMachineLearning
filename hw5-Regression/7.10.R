data1<-read.table('physical.txt', sep="\t",header=TRUE)
y<-data1$Mass
lm.mass <-lm(Mass~.,data=data1)
resid<-lm.mass$residuals
jpeg('7.10a.jpg')
plot(lm.mass$fitted.values,resid,xlab = 'fitted values',ylab = 'residual',main = 'residual vs. fitted')
dev.off()
lm.mass$coefficients


lm.masscube<-lm(Mass^(1/3) ~., data = data1)
jpeg('7.10b1.jpg')
plot(lm.masscube$fitted.values,lm.masscube$residuals,xlab = 'fitted values',ylab = 'residual',main = 'cube residual vs. fitted')
dev.off()
predicted <- predict(lm.masscube, type="r") 

new_cube_res<-y-(predicted)^3
jpeg('7.10b2.jpg')
plot((lm.masscube$fitted.values)^3,new_cube_res,xlab = 'fitted values',ylab = 'residual',main = 'residual vs. fitted in orignal coordinate')
dev.off()
lm.masscube$coefficients

