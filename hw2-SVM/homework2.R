library(readr)
library(plyr)
library(dplyr)
library(caret)
library(ggplot2)
library(reshape)
wdat1<-read.csv('adult.data.txt', header=FALSE)
wdat2<-read.csv('adult.test.txt', header=FALSE)
wdat<-rbind(wdat1,wdat2[-c(1),])
wdat$income <- ifelse(wdat[,15]==" >50K"|wdat[,15]==" >50K.", 1, -1)
as.factor(wdat$income)
#ContinuousFeatureTable<-wdat[,c(1,3,5,11,12,13)]
wdat[,c(1,3,5,11,12,13)]<-lapply(wdat[,c(1,3,5,11,12,13)],as.numeric)


wdat[,c(1,3,5,11,12,13)]<-scale(wdat[,c(1,3,5,11,12,13)])
ScaledFeatures<-wdat[,c(1,3,5,11,12,13)]
ScaledFeatures$Label<-wdat$income

# validation, and test.
fractionTraining   <- 0.80
fractionValidation <- 0.10
fractionTest       <- 0.10

# Compute sample sizes.
sampleSizeTraining   <- floor(fractionTraining   * nrow(ScaledFeatures))
sampleSizeValidation <- floor(fractionValidation * nrow(ScaledFeatures))
sampleSizeTest       <- floor(fractionTest       * nrow(ScaledFeatures))


# Create the randomly-sampled indices for the dataframe. Use setdiff() to
# avoid overlapping subsets of indices.
indicesTraining    <- sort(sample(seq_len(nrow(ScaledFeatures)), size=sampleSizeTraining))
indicesNotTraining <- setdiff(seq_len(nrow(ScaledFeatures)), indicesTraining)
indicesValidation  <- sort(sample(indicesNotTraining, size=sampleSizeValidation))
indicesTest        <- setdiff(indicesNotTraining, indicesValidation)

# Finally, output the three dataframes for training, validation and test.
Training   <- ScaledFeatures[indicesTraining, ]
Validation <- ScaledFeatures[indicesValidation, ]
Test       <- ScaledFeatures[indicesTest, ]

#held out for epoch
# take a random sample of size 50 from a dataset mydata 
# sample without replacement

#HeldOutIndex <- sample(1:nrow(Training), 50, replace = FALSE)
#EpochTraining<-Training[-HeldOutIndex,]
#HeldOut<-Training[HeldOutIndex,]


SVM<-function(lm){
  a<-rep(0,6)
  #b<-rep(0,6)
  b<-0
  TotalMagnitude<-c()
  TotalAccuracy<-c()
  Epoch<-c()
  for(e in 1:100){
    #lm=0.00001
    HeldOutIndex <- sample(1:nrow(Training), 50, replace = FALSE)
    EpochTraining<-Training[-HeldOutIndex,]
    HeldOut<-Training[HeldOutIndex,]

    steplength<-1/(0.01*e+50)
    magnitude<-c()
    accuracy<-c()
    epoch<-c()
    for (wi in 1:10){
      #magnitude<-c()
      #accuracy<-c()
      #epoch<-c()
      samples<-sample(1:nrow(EpochTraining), 30, replace = TRUE)
      for (iter in samples){
        
        y<-EpochTraining[iter,7]
        #print(y)
        x<-EpochTraining[iter,c(1,2,3,4,5,6)]
        #print(x)
        result<-y*(sum(a*x+b))
        #print(result)
        if(result>=1){
          #Grad<-result+lm*(a*a)/2
          a<-a-steplength*lm*a
          b<-b
          #a=a-steplength*(lm*a-y*x)
          #b=b+steplength*y
          
        }
        else{
          #Grad<-lm*(a*a)/2
          a<-a-steplength*(lm*a-y*x)
          b<-b+steplength*y
          #a=a-steplength*lm*a
          #b=b
          
        }
        #print(a)
        #print(b)
      }
      
      
      
      a<-as.numeric(a)
      magnitude[wi]<-sqrt(t(a) %*%a)
      #print(magnitude[wi])
      Sum<-rowSums(t((a*t(HeldOut[,c(1,2,3,4,5,6)])+b)))
      #print(Sum)
      HeldOut$predict <- ifelse(Sum>0, 1, -1)
      #accuracy<-(HeldOut$predict==HeldOut[,c(7)])/nrow(HeldOut)
      right<-HeldOut$predict==HeldOut$Label
      accuracy[wi]<-sum(right)/50
      epoch[wi]<-e
      #print(accuracy[wi])
      #print(count(HeldOut$predict==HeldOut$Label))
      
      #TotalMagnitude<-c(TotalMagnitude,magnitude)
      #TotalAccuracy<-c(TotalAccuracy,accuracy)
      #Epoch<-c(Epoch,e)
      
    }
    TotalMagnitude<-c(TotalMagnitude,magnitude)
    TotalAccuracy<-c(TotalAccuracy,accuracy)
    Epoch<-c(Epoch,epoch)
    #print(magnitude)
    #print(accuracy)
    
  }
  df<-data.frame(TotalMagnitude,TotalAccuracy,Epoch,lm)
  theList<- list("dataframe" = df, "a" = a,"b"=b)
  return(theList)
  
  
}


MyValidation<-function(a,b){
  ValidationSum<-rowSums(t((a*t(Validation[,c(1,2,3,4,5,6)])+b)))
  Validation$predict <- ifelse(ValidationSum>0, 1, -1)
  ValidationRight<-Validation$predict==Validation$Label
  ValidationAccuracy<-sum(ValidationRight)/nrow(Validation)
  return(ValidationAccuracy) 
}




myData<-SVM(1)
myData1<-myData$dataframe
myData1$ID <- seq.int(nrow(myData1))
a<-myData$a
b<-myData$b
accu1<-MyValidation(a,b)
#qplot(x=myData1$ID,y=myData1$TotalMagnitude,xlab ="steps")+ geom_line()
#qplot(x=c(1:500),y=myData1$TotalAccuracy,xlab ="steps")+ geom_line()


myData<-SVM(0.1)
myData2<-myData$dataframe
myData2$ID <- seq.int(nrow(myData2))
a<-myData$a
b<-myData$b
accu2<-MyValidation(a,b)
#qplot(x=c(1:500),y=myData2$TotalMagnitude,xlab ="steps")+ geom_line()
#qplot(x=c(1:500),y=myData2$TotalAccuracy,xlab ="steps")+ geom_line()


myData<-SVM(0.01)
myData3<-myData$dataframe
myData3$ID <- seq.int(nrow(myData3))
a<-myData$a
b<-myData$b
accu3<-MyValidation(a,b)



myData<-SVM(0.001)
myData4<-myData$dataframe
myData4$ID <- seq.int(nrow(myData4))
a<-myData$a
b<-myData$b
accu4<-MyValidation(a,b)


myData<-SVM(0.0001)
myData5<-myData$dataframe
myData5$ID <- seq.int(nrow(myData5))
a<-myData$a
b<-myData$b
accu5<-MyValidation(a,b)


FinalFrame <- melt(list("1" = myData1, "0.1" = myData2, "0.01" = myData3,"0.001" = myData4,"0.0001" = myData5), id.vars = "ID")

colnames(FinalFrame)[colnames(FinalFrame)=="L1"] <- "lambda"

ggplot(FinalFrame[FinalFrame$variable=="TotalMagnitude",], aes(x=FinalFrame[FinalFrame$variable=="TotalMagnitude",]$ID/10, value, colour = lambda)) + 
  geom_line()+labs(x ="Epoch", y = "magnitude of the coefficient vector")
ggsave("Magnitude.png")

ggplot(FinalFrame[FinalFrame$variable=="TotalAccuracy",], aes(x=FinalFrame[FinalFrame$variable=="TotalMagnitude",]$ID/10, value, colour = lambda)) + 
  geom_line()+ylim(0,1)+labs(x ="Epoch", y = "Accuracy")+facet_grid(. ~ lambda)
ggsave("Accuracy.png")

#use lambda=0.001

FinalTest<-SVM(0.001)
a<-FinalTest$a
b<-FinalTest$b

MyTest<-function(a,b){
  TestSum<-rowSums(t((a*t(Test[,c(1,2,3,4,5,6)])+b)))
  Test$predict <- ifelse(TestSum>0, 1, -1)
  TestRight<-Test$predict==Test$Label
  TestAccuracy<-sum(TestRight)/nrow(Test)
  return(TestAccuracy) 
}

accu<-MyTest(a,b)