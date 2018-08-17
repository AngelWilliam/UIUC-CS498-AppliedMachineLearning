import numpy as np
from PIL import Image
import pandas as pd
from sklearn.cluster import KMeans
import time

def segment_image(picture, segment, seed):
    img = Image.open(picture)
    width = img.size[0]
    height = img.size[1]
    pic = np.array(img.getdata())
    pixel=pd.DataFrame(pic)
    
    pi_j2 = np.zeros(segment)
    kmeans2 = KMeans(n_clusters=segment, random_state=seed).fit(pic) # Kmeans model
    for i in range(segment):
        pi_j2[i] = float((kmeans2.labels_==i).sum()) / len(pic)  # Get initial probabilities
    mu = kmeans2.cluster_centers_ # Get the initial values for mu

    W2_nom = np.zeros((pixel.shape[0], segment))
    ##########################E-step#####################

    t0 = time.time()
    for k in range(0, 20): 
        mu_old=mu
        for i in range(pixel.shape[0]):
            for j in range(segment):
                W2_nom[i, j] = np.exp(-0.5 * (np.dot((pic[i,] - mu[j,]),
                                (pic[i,] - mu[j,])))) * pi_j2[j]

        W2_denom = np.sum(W2_nom, axis=0)
        W2=W2_nom/W2_denom

        mu_denom = np.sum(W2, axis=0)
        mu_nom = np.zeros((segment, 3))
        for m in range(segment):
            for n in range(pixel.shape[0]):
                mu_nom[m,] += pic[n,] * W2[n, m]
        ##########################M-step#####################
        mu = np.zeros((segment, 3)) 
        for m in range(segment): 
            mu[m,]=mu_nom[m,]/mu_denom[m] #update mu

        pi_j2 = np.sum(W2, axis=0) / pixel.shape[0]  # update pi

        
        
        #########################simple-criteria#############
        if abs(mu_old.sum()-mu.sum())<0.7:
            print k
            break

    ### Map each pixel to the cluster mean
    ### with the highest value of the posterior probability for that pixel

    segment_index = []
    for i in range(pixel.shape[0]):  # Get the cluster
        segment_index.append(np.argsort(W2[i,])[segment - 1])

    #add segment index to pixel vector as a index for cluster
    #loop through index and subsetting same indexed rows, take mean, append to cluster_mean_list
    pixel['idx']=segment_index 
    cluster_mean_list=[]
    for i in range(segment):
        cluster=pixel[pixel['idx']==i]
        cluster=np.array(cluster)
        cluster_mean_list.append(cluster.mean(axis=0))

    #make another dataframe to hold to mean pixel value for each class
    df=pd.DataFrame(cluster_mean_list)
    cluster_mean=df.drop(3,axis=1)
    newpic = np.zeros((pixel.shape[0], 3)) #new picture pixel vector
    for n in range(pixel.shape[0]):
        newpic[n,]=cluster_mean.loc[pixel['idx'][n]] #loop through 'pixel', replace the value based on cluster_mean_list
    #newpic

    pixel_back = np.zeros((height, width, 3))
    for k in range(pixel.shape[0]):  # transform the data back to original dimension
        i = k // width
        j = k % width
        pixel_back[i, j,] = newpic[k,]
        
    img = Image.fromarray(pixel_back.astype(np.uint8), "RGB")
    
    t1 = time.time()
    total = t1-t0
    print total
        
    if height == 330:
        img.save("smallsunset" + str(seed) + "_" + str(segment) + "segments.jpg")
    elif height == 399:
        img.save("smallstrelitzia" + "_" + str(segment) + "segments.jpg")
    else:
        img.save("RobertMixed03" + "_" + str(segment) + "segments.jpg")
    return img

#########################question-1#########################
picture_list=['RobertMixed03.jpg','smallstrelitzia.jpg','smallsunset.jpg']
segment_list=[10,20,50]


for p in picture_list:
    for s in segment_list:
        segment_image(p,s,0)
        
        
#################################question-2#########################
for seed in range(1,6):
    segment_image('smallsunset.jpg',20,seed)        