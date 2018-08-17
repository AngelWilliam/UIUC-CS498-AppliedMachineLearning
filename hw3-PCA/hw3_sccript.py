import cPickle
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from sklearn.decomposition import PCA as sklearnPCA
from scipy.spatial import distance_matrix


def unpickle(file): 
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

#load data 
DataBatch1=unpickle('cifar-10-batches-py/data_batch_1')
DataBatch2=unpickle('cifar-10-batches-py/data_batch_2')
DataBatch3=unpickle('cifar-10-batches-py/data_batch_3')
DataBatch4=unpickle('cifar-10-batches-py/data_batch_4')
DataBatch5=unpickle('cifar-10-batches-py/data_batch_5')
TestBatch=unpickle('cifar-10-batches-py/test_batch')
BatchMeta=unpickle('cifar-10-batches-py/batches.meta')
#DataList=[DataBatch1,DataBatch2,DataBatch3,DataBatch4,DataBatch5,TestBatch]


#concatenate together
FullData=np.concatenate((DataBatch1['data'], DataBatch2['data'],DataBatch3['data'],
                       DataBatch4['data'],DataBatch5['data'],TestBatch['data']), axis=0)
FullLabels=np.concatenate((DataBatch1['labels'], DataBatch2['labels'],DataBatch3['labels'],
                       DataBatch4['labels'],DataBatch5['labels'],TestBatch['labels']), axis=0)

#transform concatenated arrays to pandas dataframe
df = pd.DataFrame(FullData)
#add the label as one colomn
df['label']=FullLabels
##
##now the dataframe has 3072+1 colomn(3072 features and 1 label), 60000 row, each row is a image vector
##

#create a list of 10 dataframes, based on the image class
dflist=[]
for i in range(10):
    dflist.append(df[df['label']==i])
    
#create a dataframe of 3072+1 colomn(3072 features and 1 label), 10 row, each row is a mean image vector of 
#the coresponding class
ClassDf=df.groupby(['label']).mean()

#shift to zero mean
#from sklearn.preprocessing import StandardScaler
#StandardScaler(copy=True, with_mean=True, with_std=True)
scaler = StandardScaler(copy=True, with_mean=True, with_std=False)#do not normalize to unit of 1
mean=[]#list of mean image vector of each class, contains same information as the ClassDf
CenteredDf=[]#list of 10 centered dataframes, based on the image class
             #each has 3072 colomn(3072 features) and 6000 rows
for df in dflist:
    scaler.fit(df.iloc[:,0:3072])
    #scaler.fit_transform(df.iloc[:,0:3072])
    mean.append(scaler.mean_)
    CenteredDf.append(scaler.fit_transform(df.iloc[:,0:3072]))
    
    
#PART 1
#from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=3072)
Error=[]
for df in CenteredDf:##fed the PCA function with centered df of each class
    Y_sklearn = sklearn_pca.fit(df)
    Error.append(Y_sklearn.explained_variance_[20:].sum())
    #explained_variance_ is the eigenvalues sorted by weight
    #take the mean of eigenvalues after from 21-3072, is the totoal error
    #Y_sklearn.explained_variance_ratio_.sum()
#print Error   


labels=BatchMeta['label_names']
y_pos = np.arange(len(labels))
plt.bar(y_pos, Error, align='center', alpha=0.5)
plt.xticks(y_pos, labels)
plt.ylabel('Mean Error')
plt.show()
#plt.savefig('Part1.png')



#PART 2
##get the coordination of mean image
#from sklearn import manifold
mds = manifold.MDS(n_components=2,random_state=0,dissimilarity='euclidean')
coords = mds.fit_transform(ClassDf)

#from sklearn.metrics.pairwise import euclidean_distances
#from sklearn.metrics.pairwise import paired_distances

#from scipy.spatial import distance_matrix
DistanceMatrix=pd.DataFrame(distance_matrix(ClassDf, ClassDf))

#DistanceMatrix
#https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
x1,y1= zip(*coords.tolist())
fig, ax = plt.subplots()
ax.scatter(x1, y1)
for i, txt in enumerate(labels):
    ax.annotate(txt, (x1[i],y1[i]))
plt.show()
#plt.savefig('Part2-2.png')


#PART 3

PCA = sklearnPCA(n_components=20)
PClist=[]
##form a list contains PC for all image class
for df in CenteredDf:
    model = PCA.fit(df)
    PC=model.components_
    PClist.append(PC)
    
#create a list of list by computing E(A|B), each inside list is one image class against other classes
col=[]
for i in range(0,10):
    row=[]
    for j in range(0,10):
        
        r=np.dot(PClist[j],CenteredDf[i].T)
        x=np.dot(r.T,PClist[j])
        x=np.add(x,mean[i])
        simil_error1 = np.sum(np.square(x - dflist[i].iloc[:,0:3072]),axis=1).mean()
        row.append(simil_error1)
    col.append(row) 
    row=[]
        
ErrorMatrix=np.array(col)
#create a list of list by computing distance, each inside list is one image class against other classes
Distcol=[]
for i in range(0,10):
    Distrow=[]
    for j in range(0,10):
        
        simi=(ErrorMatrix[i][j]+ErrorMatrix[j][i])/2.0
        Distrow.append(simi)
    Distcol.append(Distrow) 
    Distrow=[]
DistMatrix=np.array(Distcol)  


A = np.identity(10) - np.reshape(np.ones(100) / 10, (10, 10))

W = -0.5 * np.dot(np.dot(A.T, DistMatrix), A)

# extracts eigenvalues and eigenvector from W
eigen_val, eigen_vec = np.linalg.eig(W)

idx = eigen_val.argsort()[::-1]
eigen_val = eigen_val[idx]
eigen_vec = eigen_vec[:, idx]
lamb_da = np.dot(np.dot(eigen_vec.T, W), eigen_vec)

lamb_da1 = lamb_da[0:2, 0:2]
lamb_da1[0, 0] = np.sqrt(lamb_da1[0, 0])
lamb_da1[1, 1] = np.sqrt(lamb_da1[1, 1])

V = np.dot(lamb_da1, eigen_vec[:, 0:2].T)

x2=V.T[:, 0]
y2=V.T[:, 1]
fig, ax = plt.subplots()
ax.scatter(x2.tolist(), y2.tolist())
for i, txt in enumerate(labels):
    ax.annotate(txt, (x2[i],y2[i]))
plt.show()
#plt.savefig('Part3.png')