wdat<-read.csv('pima-indians-diabetes.data', header=FALSE)
library(klaR)
library(caret)
bigx<-wdat[,-c(9)]
bigy<-wdat[,9]

##turn the 0 value to NA
nbx<-bigx
for (i in c(3, 4, 6, 8))
{vw<-bigx[, i]==0
nbx[vw, i]=NA
}


train.score<-array(dim=10)
test.score<-array(dim=10)
for (wi in 1:10)
{wtd<-createDataPartition(y=bigy, p=.8, list=FALSE) #creat test/training partition, a vector of index?
#nbx<-bigx #main feature frame
ntrbx<-nbx[wtd, ] #training feature frame
ntrby<-bigy[wtd] #training label vector
trposflag<-ntrby>0 #traning positive flag

ptregs<-ntrbx[trposflag, ] #positive samples from traning data
ntregs<-ntrbx[!trposflag,] #negative samples from traning data
ntebx<-nbx[-wtd, ] #testing feature frame
nteby<-bigy[-wtd] #testing lable vector

tesposflag<-nteby>0#test positive flag

ptrmean<-sapply(ptregs, mean, na.rm=TRUE) #mean of features from positive samples from traning data
ntrmean<-sapply(ntregs, mean, na.rm=TRUE) #mean of features from negative samples from traning data
ptrsd<-sapply(ptregs, sd, na.rm=TRUE) ##sd of features from positive samples from traning data
ntrsd<-sapply(ntregs, sd, na.rm=TRUE) #sd of features from negative samples from traning data

yp<-matrix(NA,nrow = nrow(ntrbx), ncol = ncol(ntrbx))
yn<-matrix(NA,nrow = nrow(ntrbx), ncol = ncol(ntrbx))

for (r in 1:(nrow(ntrbx))){
  for (c in 1:(ncol(ntrbx))){
    yp[r,c] <- dnorm(ntrbx[r,c], mean=ptrmean[c], sd=ptrsd[c]) 
  }
}
yplog <- rowSums(log(yp),na.rm=TRUE)+(log(nrow(ptregs)/nrow(ntrbx)))

for (r in 1:(nrow(ntrbx))){
  for (c in 1:(ncol(ntrbx))){
    yn[r,c] <- dnorm(ntrbx[r,c], mean=ntrmean[c], sd=ptrsd[c]) 
  }
}
ynlog <- rowSums(log(yn),na.rm=TRUE)+(log(nrow(ntregs)/nrow(ntrbx)))

train.predict.pos<-yplog>ynlog
train.pos.right<-train.predict.pos==trposflag
train.score[wi]<-sum(train.pos.right)/(sum(train.pos.right)+sum(!train.pos.right))
#--------------  


test.yp<-matrix(NA,nrow = nrow(ntebx), ncol = ncol(ntebx))
test.yn<-matrix(NA,nrow = nrow(ntebx), ncol = ncol(ntebx))

for (r in 1:(nrow(ntebx))){
  for (c in 1:(ncol(ntebx))){
    test.yp[r,c] <- dnorm(ntebx[r,c], mean=ptrmean[c], sd=ptrsd[c]) 
  }
}
test.yplog <- rowSums(log(test.yp),na.rm=TRUE)+(log(nrow(ptregs)/nrow(ntrbx)))

for (r in 1:(nrow(ntebx))){
  for (c in 1:(ncol(ntebx))){
    test.yn[r,c] <- dnorm(ntebx[r,c], mean=ntrmean[c], sd=ptrsd[c]) 
  }
}
test.ynlog <- rowSums(log(test.yn),na.rm=TRUE)+(log(nrow(ntregs)/nrow(ntrbx)))

test.predict.pos<-test.yplog>test.ynlog
test.pos.right<-test.predict.pos==tesposflag
test.score[wi]<-sum(test.pos.right)/(sum(test.pos.right)+sum(!test.pos.right))

}  

mean(test.score)
