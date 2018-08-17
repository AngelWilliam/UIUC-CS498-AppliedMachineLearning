library(cluster)
data<-read.table('hw4-data.txt', sep="\t",header=TRUE)
#make a frame with row names for easy label in the graph
frame=data[,2:10]
row.names(frame)<-data[,1]
#-------------single link------------#
agn1 <- agnes(x=frame, diss = FALSE, stand = FALSE, 
             method = "single")
DendAgn1 <-as.dendrogram(agn1)
jpeg('1-single.jpg')
plot(DendAgn1)
dev.off()


#-------------complete link------------#
agn2 <- agnes(x=frame, diss = FALSE, stand = FALSE, 
             method = "complete")
DendAgn2 <-as.dendrogram(agn2)
jpeg('1-complete.jpg')
plot(DendAgn2)
dev.off()

#-------------group average------------#
agn3 <- agnes(x=frame, diss = FALSE, stand = FALSE, 
              method = "average")
DendAgn3 <-as.dendrogram(agn3)
jpeg('1-average.jpg')
plot(DendAgn3)
dev.off()

#https://onlinecourses.science.psu.edu/stat857/node/136

#km <- kmeans(frame, centers = 3, iter.max=30,nstart = 25)

#https://rpubs.com/FelipeRego/K-Means-Clustering
mydata <- frame
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(mydata,
                                     centers=i)$withinss)
jpeg('Elbow.jpg')
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",
     main="Assessing the Optimal Number of Clusters with the Elbow Method",
     pch=20, cex=2)
dev.off()
km1 <- kmeans(frame, centers = 3, iter.max=30,nstart = 100)
km1$cluster[km1$cluster==1]
km2 <- kmeans(frame, centers = 4, iter.max=30,nstart = 100)
km2$cluster
#plot(mydata, col = km1$cluster)
