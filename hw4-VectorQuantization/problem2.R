library(klaR)
library(caret)
library(randomForest)
ActList<-c('Brush_teeth',
           'Climb_stairs',
           'Comb_hair',
           'Descend_stairs',
           'Drink_glass',
           'Eat_meat',
           'Eat_soup',
           'Getup_bed',
           'Liedown_bed',
           'Pour_water',
           'Sitdown_chair',
           'Standup_chair',
           'Use_telephone',
           'Walk')
homework4<-function(chunk,center){
  ###---read from filepath, process segmentation for each activity, 
  ###---build a huge dataframe contains flattened vectors
  ###---and the label of file and activity
  ReadFile<-function(FilePath,FileLists,Act){
    #a list of tables that read from the test files
    #https://piazza.com/class/jchzguhsowz6n9?cid=746
    Activity<-lapply(paste(FilePath,FileLists,sep = '/'), 
                     function(x) read.table(x, header=FALSE))
    tablelist = list()#contains table extracted from this sub-folder
    #  for(t in Activity){
    nlist=length(Activity)
    for(t in 1:nlist){
      #cut the table into chunks
      #https://stackoverflow.com/questions/7060272/split-up-a-dataframe-by-number-of-rows
      ttable<-Activity[[t]]
      chunk <- chunk
      n <- nrow(ttable)
      r  <- rep(1:ceiling(n/chunk),each=chunk)[1:n]
      d <- split(ttable,r)
      #discard the last remainder
      #flatten that table
      #https://stat.ethz.ch/pipermail/r-help/2005-March/068063.html
      #append the vector in a dataframe
      #https://stackoverflow.com/questions/29402528/append-data-frames-together-in-a-for-loop
      d <- d[-length(d)]
      datalist = list()
      for(i in 1:length(d)){
        a<-d[[i]]
        x <- as.vector(t(a))
        #x$i <- act
        datalist[[i]] <- x
      }
      big_data = do.call(rbind, datalist) #row bind 96 length vecoter 
      
      big_data<-as.data.frame(big_data)
      big_data$label<-Act #dataframe table of 1 file in current activity
      big_data$file<-t
      tablelist[[t]] <- big_data
    }
    activity_data<-do.call(rbind, tablelist)
    return(activity_data)
  }
  
  TrainList = list()
  TestList = list()
  for(act in ActList){
    
    filepath<-file.path('HMP_Dataset',act)
    filelist<-list.files(path=filepath)
    #https://stackoverflow.com/questions/17200114/how-to-split-data-into-training-testing-sets-using-sample-function
    smp_size <- floor(0.8 * length(filelist))
    set.seed(1234)
    train_ind <- sample(seq_len(length(filelist)), size = smp_size)
    train <- filelist[train_ind]
    test <- filelist[-train_ind]
    TrainTable<-ReadFile(FilePath=filepath,FileLists=train,Act=act)
    TestTable<-ReadFile(FilePath=filepath,FileLists=test,Act=act)
    TrainList[[act]] <- TrainTable
    TestList[[act]] <- TestTable
  }
  TrainMatrix<-do.call(rbind, TrainList)
  TestMatrix<-do.call(rbind, TestList)
  TrainMatrix$class<-NA
  TestMatrix$class<-NA
  
  chunk<-chunk
  col<-chunk*3
  
  df<-TrainMatrix[,c(1:col)]
  kmean <- kmeans(df, centers = center, iter.max=30,nstart = 25)
  centers<-kmean$centers
  
  euc.dist <- function(x1, x2){sqrt(sum((x1 - x2) ^ 2))}
  
  ToAssign<-TrainMatrix[,c(1:col)]
  
  ClassAssign<-function(x,Centers){
    #q<-array(dim=480)
    q<-apply(Centers,1, euc.dist,x2=x)
    Class<-match(min(q),q)
    return(Class)
  }
  #https://www.rdocumentation.org/packages/base/versions/3.4.3/topics/apply
  s<-apply(ToAssign,1,ClassAssign,Centers=centers)
  TrainMatrix$class<-s
  
  TestAssign<-TestMatrix[,c(1:col)]
  s.test<-apply(TestAssign,1,ClassAssign,Centers=centers)
  TestMatrix$class<-s.test
  
  FileClass<-TrainMatrix[,c('label','file','class')]
  FileClass.test<-TestMatrix[,c('label','file','class')]
  
  
  #function that calculate the hist for feature representation
  FeatureVector<-function(x){
    container<-rep(0, center)
    for(value in 1:center){
      Nrow<-nrow(x[x$class == value,])
      container[value]=Nrow
    }
    container<-array(container/length(container))
    return(container)
  }  
  
  
  BigList<-list()
  TestBigList<-list()
  for(a in ActList){
    
    f<-FileClass[FileClass$label==a,]
    f.test<-FileClass.test[FileClass.test$label==a,]
    #https://stackoverflow.com/questions/9713294/split-data-frame-based-on-levels-of-a-factor-into-new-data-frames
    X <- split(f, factor(f$file))
    X.test<-split(f.test, factor(f.test$file))
    #-------make train feature dataframe-------  
    FeatureList<-list()
    for(n in 1:length(X)){
      x<-X[[n]]
      FeatureList[[n]]<-FeatureVector(x)
    }
    FeatureMatrix<-do.call(rbind, FeatureList)
    FeatureMatrix<-as.data.frame(FeatureMatrix)
    FeatureMatrix$label<-a
    idx<-match(a,ActList)
    BigList[[idx]]<-FeatureMatrix
    
    #-------make train feature dataframe-------
    TestFeatureList<-list()
    for(n in 1:length(X.test)){
      x.test<-X.test[[n]]
      TestFeatureList[[n]]<-FeatureVector(x.test)
    }
    TestFeatureMatrix<-do.call(rbind, TestFeatureList)
    TestFeatureMatrix<-as.data.frame(TestFeatureMatrix)
    TestFeatureMatrix$label<-a
    #idx<-match(a,ActList)
    TestBigList[[idx]]<-TestFeatureMatrix
    
  }
  FeatureMatrix<-do.call(rbind, BigList)
  FeatureMatrix$label = factor(FeatureMatrix$label) 
  TestFeatureMatrix<-do.call(rbind, TestBigList)
  #https://stackoverflow.com/questions/39320408/error-in-y-ymean-non-numeric-argument-to-binary-operator-randomforest-r
  TestFeatureMatrix$label = factor(TestFeatureMatrix$label) 
  
  
  forest <- randomForest(label ~ .,data = FeatureMatrix,ntree=200)
  plot(forest)
  pred<-predict(forest,newdata=TestFeatureMatrix)
  table(pred,TestFeatureMatrix$label)
  accuracy<-pred==TestFeatureMatrix$label
  accuracy<-sum(accuracy)/length(accuracy)
  accuracy
  error<-1-accuracy
  error
  return(error)
  #return(table(pred,TestFeatureMatrix$label))
}

chunk.list<-c(16,32,64)
center.list<-c(20,50,100,200,400,500)
#chunk.list<-c(32)
#center.list<-c(480)
for(ck in chunk.list){
  for(ct in center.list){
    result<-homework4(ck,ct)
    print(c(ck,ct,result))
  }
}

homework4(32,480)
