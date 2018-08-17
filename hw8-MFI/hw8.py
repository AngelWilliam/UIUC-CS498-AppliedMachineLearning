import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
from numpy import zeros, float32
from copy import deepcopy
from pylab import imshow, show, cm

#os.chdir("/media/adrian/Files/Documents/MCS-DS/AppliedMachineLearning_CS498/HW8CS498AML")

# https://martin-thoma.com/classify-mnist-with-pybrain/


images = "train-images-idx3-ubyte.gz"

def get_labeled_data(imagefile):
    images = gzip.open(imagefile, 'rb')

    images.read(4)  
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    x = zeros((500, rows, cols), dtype=float32)
    for i in range(500):
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
    return x


dt = get_labeled_data(images) / 255.0 ##changed to float
pixel = deepcopy(dt)

#####binarize#####
for k in range((500)):
    for i in range(pixel.shape[1]):
        for j in range(pixel.shape[2]):
            if pixel[k][i][j] <= 0.5:
                pixel[k][i][j] = -1
            else:
                pixel[k][i][j] = 1

pre_noise_pixel = deepcopy(pixel[0:20]) #first 20 pics

shape = pixel.shape[1]  

#https://stackoverflow.com/questions/24662571/python-import-csv-to-list
#reading coord csv into list of tuples, for each noise and pic, zip tuple value into (col, row) 
import pandas as pd
df = pd.read_csv('SupplementaryAndSampleData/NoiseCoordinates.csv', delimiter=',')
tuples = [tuple(x) for x in df.values]
coord=[]
for i in range(40)[::2]: 
    z=zip(tuples[i],tuples[i+1])
    coord.append(z)

#adding noise by replacing -1 with 1 and vice versa
noise_pixel=deepcopy(pre_noise_pixel)
for i in range(0,20):
    noise_coord=coord[i][1:16]
    
    for noise_point in noise_coord:
        if noise_pixel[i][noise_point[0]][noise_point[1]]==-1:
            noise_pixel[i][noise_point[0]][noise_point[1]]=1
        else:
            noise_pixel[i][noise_point[0]][noise_point[1]]=-1    

#check picture
from PIL import Image
img = Image.fromarray(noise_pixel[2].astype(np.uint8))
#img.show()


##EQ=EQ_logQ-EQ_logP
##calculate in two steps: EQ_logQ and EQ_logP  
def EQ_logQ_calculation(Q,shape,epsilon):
    All_EQ_logQ=[]
    for pi_j in Q:    #Q[H=1]
        pi_j_n=1-pi_j #Q[H=-1]
        Sum_col=np.zeros(shape)
        for row in range(shape):
            q=pi_j[row]
            log_q=np.log(pi_j+epsilon)[row]
            q_n=pi_j_n[row]
            log_q_n=np.log(pi_j_n+epsilon)[row]
            Sum_col[row]=np.dot(q,log_q.T)+np.dot(q_n,log_q_n.T) #entropy
        EQ_logQ=Sum_col.sum()
        All_EQ_logQ.append(EQ_logQ)
    return All_EQ_logQ
    
def EQ_logP_calculation(Q,shape,noise_pixel):
    All_EQ_logP=[]
    for k in range(len(noise_pixel)):
        pi_j=Q[k]
        EQ_logP=np.zeros((shape, shape))
        for i in range(shape):
            for j in range(shape): # 4 neighbors

                if (i in range(1, shape - 1) and j in range(1, shape - 1)):
                        H=0.8*(2 * pi_j[i][j] - 1)*(2*pi_j[i][j + 1] - 1+2 * pi_j[i][j - 1] - 1+\
                                  2 * pi_j[i - 1][j] - 1 + 2 *pi_j[i + 1][j] - 1)
                        X=2 * (2 * pi_j[i][j] - 1)* noise_pixel[k][i][j]

                elif i == 0 and j == 0:  # 2 neighbors
                        H=0.8*(2 * pi_j[i][j] - 1)*(2*pi_j[i][j + 1] - 1+2 * pi_j[i+1][j] - 1)
                        X=2 * (2 * pi_j[i][j] - 1)* noise_pixel[k][i][j]

                elif i == 0 and j == shape - 1:
                    H=0.8*(2 * pi_j[i][j] - 1)*(2*pi_j[i][j - 1] - 1+2 * pi_j[i+1][j] - 1)
                    X=2 * (2 * pi_j[i][j] - 1)* noise_pixel[k][i][j]

                elif i == shape - 1 and j == 0:
                    H=0.8*(2 * pi_j[i][j] - 1)*(2*pi_j[i-1][j] - 1+2 * pi_j[i][j+1] - 1)
                    X=2 * (2 * pi_j[i][j] - 1)* noise_pixel[k][i][j]
                elif i == shape - 1 and j == shape - 1:
                    H=0.8*(2 * pi_j[i][j] - 1)*(2*pi_j[i-1][j] - 1+2 * pi_j[i][j-1] - 1)
                    X=2 * (2 * pi_j[i][j] - 1)* noise_pixel[k][i][j]
                elif j == 0:  # 3 neighbors
                    H=0.8*(2 * pi_j[i][j] - 1)*(2*pi_j[i-1][j] - 1+2 * pi_j[i][j + 1] - 1+\
                                  2 * pi_j[i + 1][j] - 1)
                    X=2 * (2 * pi_j[i][j] - 1)* noise_pixel[k][i][j]
                elif i == 0:
                    H=0.8*(2 * pi_j[i][j] - 1)*(2*pi_j[i+1][j] - 1+2 * pi_j[i][j + 1] - 1+\
                                  2 * pi_j[i][j-1] - 1)
                    X=2 * (2 * pi_j[i][j] - 1)* noise_pixel[k][i][j]
                elif i == shape - 1:
                    H=0.8*(2 * pi_j[i][j] - 1)*(2*pi_j[i-1][j] - 1+2 * pi_j[i][j + 1] - 1+\
                                  2 * pi_j[i][j-1] - 1)
                    X=2 * (2 * pi_j[i][j] - 1)* noise_pixel[k][i][j]
                else:
                    H=0.8*(2 * pi_j[i][j] - 1)*(2*pi_j[i-1][j] - 1+2 * pi_j[i][j - 1] - 1+\
                                  2 * pi_j[i+1][j] - 1)
                    X=2 * (2 * pi_j[i][j] - 1)* noise_pixel[k][i][j]

                EQ_logP[i, j]=H+X
        All_EQ_logP.append(EQ_logP.sum())
    return All_EQ_logP


# initialize matrices
from numpy import genfromtxt
initial_Q = genfromtxt('SupplementaryAndSampleData/InitialParametersModel.csv', delimiter=',')

#load order coords
df = pd.read_csv('SupplementaryAndSampleData/UpdateOrderCoordinates.csv', delimiter=',')
tuples = [tuple(x) for x in df.values]
order=[]
for i in range(40)[::2]: 
    z=zip(tuples[i],tuples[i+1])
    order.append(z)


##### denoising algorithm #####
# the equation in page 263 (march version) was split into two
# as seen on the top portion of the same page: 
# 1) log q_i(1) and 2) log q_i(-1)
# then combine them to calculate pi_j

EQ_to_file=[]    

#calculate initial EQ
Q = [initial_Q for x in range(20)]
EQ_logQ=EQ_logQ_calculation(Q,shape,epsilon=0.0000000001)
EQ_logP=EQ_logP_calculation(Q,shape,noise_pixel)
EQ=np.asarray(EQ_logQ)-np.asarray(EQ_logP)
EQ_to_file.append(EQ)

for iteration in range(10):
    
    for k in range(20):#20 pictures
        
        pi_j = Q[k]
        pi_i=np.zeros((shape, shape))

        for m in range(1,785):#784 pixel
            i=order[k][m][0]
            
            j=order[k][m][1]
            
            #upper_n1 = np.zeros((shape, shape))
            #upper_p1 = np.zeros((shape, shape))
            # 4 neighbors
            if (i in range(1, shape - 1) and j in range(1, shape - 1)):
                                upper_p1 = 0.8 * (2 * pi_j[i][j + 1] - 1 + 2 * pi_j[i][j - 1] - 1 + \
                                                        (2 * pi_j[i - 1][j] - 1) + 2 * pi_j[i + 1][j] - 1) + \
                                                 2 * noise_pixel[k][i][j]
                                upper_n1 = -0.8 * (2 * pi_j[i][j + 1] - 1 + 2 * pi_j[i][j - 1] - 1 + \
                                                         (2 * pi_j[i - 1][j] - 1) + 2 * pi_j[i + 1][j] - 1) - \
                                                 2 * noise_pixel[k][i][j]
            elif i == 0 and j == 0:  # 2 neighbors
                        upper_p1 = 0.8 * ((2 * pi_j[i][j + 1] - 1) + \
                                                2 * pi_j[i + 1][j] - 1) + \
                                         2 * noise_pixel[k][i][j]
                        upper_n1 = -0.8 * ((2 * pi_j[i][j + 1] - 1) + \
                                                 2 * pi_j[i + 1][j] - 1) - \
                                         2 * noise_pixel[k][i][j]

            elif i == 0 and j == shape - 1:
                        upper_p1 = 0.8 * (2 * pi_j[i][j - 1] - 1 + \
                                                2 * pi_j[i + 1][j] - 1) + \
                                         2 * noise_pixel[k][i][j]

                        upper_n1 = -0.8 * (2 * pi_j[i][j - 1] - 1 + \
                                                 2 * pi_j[i + 1][j] - 1) - \
                                         2 * noise_pixel[k][i][j]

            elif i == shape - 1 and j == 0:
                        upper_p1 = 0.8 * (2 * pi_j[i - 1][j] - 1 + \
                                                2 * pi_j[i][j + 1] - 1) + \
                                         2 * noise_pixel[k][i][j] 
                        upper_n1 = -0.8 * (2 * pi_j[i - 1][j] - 1 + \
                                                 2 * pi_j[i][j + 1] - 1) - \
                                         2 * noise_pixel[k][i][j]

            elif i == shape - 1 and j == shape - 1:
                        upper_p1 = 0.8 * (2 * pi_j[i - 1][j] - 1 + \
                                                2 * pi_j[i][j - 1] - 1) + \
                                         2 * noise_pixel[k][i][j]
                        upper_n1 = -0.8 * (2 * pi_j[i - 1][j] - 1 + \
                                                 2 * pi_j[i][j - 1] - 1) - \
                                         2 * noise_pixel[k][i][j]

            elif j == 0:  # 3 neighbors
                        upper_p1 = 0.8 * (2 * pi_j[i - 1][j] - 1 + \
                                                2 * pi_j[i][j + 1] - 1 + 2 * pi_j[i + 1][j] - 1) + \
                                         2 * noise_pixel[k][i][j] 
                        upper_n1 = -0.8 * (2 * pi_j[i - 1][j] - 1 + \
                                                 2 * pi_j[i][j + 1] - 1 + 2 * pi_j[i + 1][j] - 1) - \
                                         2 * noise_pixel[k][i][j]
            elif i == 0:
                        upper_p1 = 0.8 * (2 * pi_j[i + 1][j] - 1 + \
                                                2 * pi_j[i][j + 1] - 1 + 2 * pi_j[i][j - 1] - 1) + \
                                         2 * noise_pixel[k][i][j]
                        upper_n1 = -0.8 * (2 * pi_j[i + 1][j] - 1 + \
                                                 2 * pi_j[i][j + 1] - 1 + 2 * pi_j[i][j - 1] - 1) - \
                                         2 * noise_pixel[k][i][j]
            elif i == shape - 1:
                        upper_p1 = 0.8 * (2 * pi_j[i - 1][j] - 1 + \
                                                2 * pi_j[i][j + 1] - 1 + 2 * pi_j[i][j - 1] - 1) + \
                                         2 * noise_pixel[k][i][j]
                        upper_n1 = -0.8 * (2 * pi_j[i - 1][j] - 1 + \
                                                 2 * pi_j[i][j + 1] - 1 + 2 * pi_j[i][j - 1] - 1) - \
                                         2 * noise_pixel[k][i][j]
            else:
                        upper_p1 = 0.8 * (2 * pi_j[i - 1][j] - 1 + \
                                                2 * pi_j[i][j - 1] - 1 + 2 * pi_j[i + 1][j] - 1) + \
                                         2 * noise_pixel[k][i][j]

                        upper_n1 = -0.8 * (2 * pi_j[i - 1][j] - 1 + \
                                                 2 * pi_j[i][j - 1] - 1 + 2 * pi_j[i + 1][j] - 1) - \
                                         2 * noise_pixel[k][i][j]


            # finally we can put it all together. calculate pi_i 
            # see page 263 (march version) for equation
            
            ###important--update pi_j after each pixel calulation
            pi_j[i][j] = np.exp(upper_p1) / (np.exp(upper_n1) + np.exp(upper_p1))
            
            ###populate pi_i matrix at the mean time
            pi_i[i, j] = np.exp(upper_p1) / (np.exp(upper_n1) + np.exp(upper_p1))
        
        Q[k]= pi_i #update Q matrix by replacing position k with pi_i
                   ###had problem before because I directly used Q[k]= pi_j, 
                   ###for some unknown reason, it causes 20 matrixs in Q are all the same
        
    EQ_logQ=EQ_logQ_calculation(Q,shape,epsilon=0.0000000001)
    EQ_logP=EQ_logP_calculation(Q,shape,noise_pixel)
    EQ=np.asarray(EQ_logQ)-np.asarray(EQ_logP)
    EQ_to_file.append(EQ)
    

##construct de_noise_pixel based on Q value
de_noise_pixel=deepcopy(Q)
for i in range(20):
    for x in range(28):
        for y in range(28):
            if de_noise_pixel[i][x][y]<=0.5:
                de_noise_pixel[i][x][y]=-1
            else:
                de_noise_pixel[i][x][y]=1 
                
de_noise_img = Image.fromarray(de_noise_pixel[3].astype(np.uint8))
EQ_to_file=np.asarray(EQ_to_file)
np.savetxt("energy.csv", EQ_to_file, delimiter=",")

pic_file=[]
for pic in de_noise_pixel[10:20]:
    for row in pic:
        pic_file.append(row)
        
pic_file=np.asarray(pic_file)
np.savetxt("pixel.csv", pic_file.T, delimiter=",")