import numpy as np
from scipy.sparse import bsr_matrix
from scipy.linalg import block_diag
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from scipy.sparse.linalg import qmr
import os
import sys
import scipy.sparse as sps
from builtins import float
import matplotlib.pyplot as plt
import numpy as np
'''
generating the new data from sparse_mat 

and the A is changed here by replacing normal distributin values
'''


def gen_data(sparse_mat, num_samples=100):
    num_rows = sparse_mat.shape[0]
    a=sparse_mat.data
    b=sparse_mat.indices
    c=sparse_mat.indptr
    sparse_mat_temp=bsr_matrix((a,b,c))
    mtx=bsr_matrix((a,b,c))
    for i1 in range(0,len(sparse_mat_temp.data)):
        for i2 in range(0,len(sparse_mat_temp.data[i1])):
            for i3 in range(0,len(sparse_mat_temp.data[i1][i2])):
                mtx.data[i1][i2][i3]=np.random.normal(sparse_mat_temp.data[i1][i2][i3], 0.1, 1)
    sparse_mat_final_1=mtx#sparse_mat_final.tobsr()
    train_in = []
    train_out = []
    for i in range(num_samples):
        fake_x = np.random.uniform(0.0, 1.0, size=(num_rows,))
        b = sparse_mat_final_1.dot(fake_x)
        train_in.append(b)
        train_out.append(fake_x)
    nnz=sparse_mat_final_1.nnz
    sparse_mat_append=sparse_mat_final_1.data.reshape(1, nnz)#,len(batch_in), axis=0)
    return np.array(train_in, dtype=np.float32), np.array(train_out, dtype=np.float32),sparse_mat_append #, sparse_mat_final

def gen_data_test(sparse_mat, num_samples=100):
    num_rows = sparse_mat.shape[0]
    train_in = []
    train_out = []
    for i in range(num_samples):
        fake_x = np.random.uniform(0.0, 1.0, size=(num_rows,))
        b = sparse_mat.dot(fake_x)
        train_in.append(b)
        train_out.append(fake_x)
    nnz=sparse_mat.nnz
    sparse_mat_append=sparse_mat.data.reshape(1, nnz)#,len(batch_in), axis=0)
    return np.array(train_in, dtype=np.float32), np.array(train_out, dtype=np.float32), sparse_mat_append

#def read_mat(base_dir="ML-SOLVER"):
#    with open(os.path.join(base_dir, "a_off.txt"), "r") as f:
#        a_off = np.array([[float(y) for y in x.split(",")] for x in f.readlines()], dtype=np.float32)
#        a_off = a_off.reshape(85768, 5, 5)
#    with open(os.path.join(base_dir, "a_diag.txt"), "r") as f:
#        a_diag = np.array([[float(y) for y in x.split(",")] for x in f.readlines()], dtype=np.float32)
#        a_diag = a_diag.reshape(6309, 5, 5)
#        a_diag = bsr_matrix(block_diag(*a_diag))
#    with open(os.path.join(base_dir, "iam.txt"), "r") as f:
#        iams = np.array([float(x.strip()) for x in f.readlines()], dtype=np.int32)
#    with open(os.path.join(base_dir, "jam.txt"), "r") as f:
#        jams = np.array([float(x.strip()) for x in f.readlines()], dtype=np.int32)
#    sparse_mat = bsr_matrix((a_off, jams - 1, iams - 1)) + a_diag
#    return sparse_mat
    
def read_mat():
    f = open('mat.out','r')
    result = [[0]*4224 for i in range(4224)]
    i=0
    for line in f:
        if i>=2:
            #print(i)
            line = line[:-1]
            lineSplit = line.split(":")
            indiceInfo = lineSplit[1].strip().split(")  (")
    
            for index,ele in enumerate(indiceInfo):
                if index==0:
                    ele = ele[1:]
                if index == len(indiceInfo)-1:
                    ele = ele[:-1]
                someInfo = ele.split(',')
                infoIndex = int(someInfo[0].strip())
                value = float(someInfo[1].strip()) 
                result[i-2][infoIndex] = float(str(value))
        i+=1
        if (4226 == i):
            break
    return bsr_matrix(result)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

	
sparse_mat = read_mat()
nnz = sparse_mat.nnz	
model = Sequential()
model.add(Dense(64,input_shape=(nnz + sparse_mat.shape[0],)))
model.add(Activation('tanh'))
model.add(BatchNormalization())
model.add(Dense(sparse_mat.shape[0]))
model.compile(loss='mse',
              optimizer='rmsprop')

#### MARK: running the below script for n number of samples to get
####       n rmse values
trainSamples = [4, 5, 6, 7, 8, 9]
setsNo = [4, 5, 6, 7, 8, 9]
rmseResults = []

for counter, sampleNo in enumerate(trainSamples):
#    print(counter)
#    print(sampleNo)
#    print(setsNo[counter])
    train_sample=int(0.9*int(sampleNo))  # total number of samples in each set #sys.argv[1]
    print(train_sample)
    iter=int(setsNo[counter])                   # total number of sets #sys.argv[2]
    test_sample=int((train_sample*0.1)*iter)# calculating test set
    print(test_sample)
    
    test_in, test_out,sparse_append = gen_data_test(sparse_mat, num_samples=test_sample)
    test_in = np.hstack((test_in, np.repeat(sparse_append,
                         test_in.shape[0], axis=0)))
    #print(train_in.shape)
    
    #exit(0)
    list_in_arrays=[]
    list_out_arrays=[]
    list_sparse_append=[]
    '''
    generating data 
    
    '''
    
    for iteration in range(0,iter):
        #print(sparse_mat.shape)
    
        train_in, train_out,sparse_append = gen_data(sparse_mat, num_samples=train_sample)
        list_in_arrays.append(train_in)
        list_out_arrays.append(train_out)
        #print(sparse_append.shape)
        #exit(0)
        list_sparse_append.append(sparse_append)
    
    '''
    training the model setwise
    '''
    
    for ind in range(0,iter):
        temp_in=list_in_arrays[ind][:]
        temp_out=list_out_arrays[ind][:]
        sparse_append_to=list_sparse_append[ind][:]
        #exit(0)
        batch_size = 128
        for epoch in range(10):
            batch_losses = []
            for i in range(0, len(temp_in), batch_size):
                batch_in = temp_in[i:i+batch_size]
                batch_out = temp_out[i:i+batch_size]
    #            print(batch_in.shape)
    #            print(batch_out.shape)
                batch_in = np.hstack((batch_in, np.repeat(sparse_append_to,
                             len(batch_in), axis=0)))
                batch_loss = model.train_on_batch(batch_in, batch_out)
                batch_losses.append(batch_loss)
    #            print("\r %s %s" % (i/len(train_in)*100., np.average(batch_losses)), end="")
            #print("")
    #        print("Epoch loss:", np.average(batch_losses))
    #        print("Test loss:", model.evaluate(test_in, test_out))
        print(ind)
    print("TEST iN")
    print(test_in[:1])
    x0 = model.predict(test_in[:1]).flatten()
    rmse_value=rmse(x0,test_out)
    rmseResults.append(rmse_value)
    print(counter, rmse_value)
    #qmrsolve = qmr(sparse_mat, test_in[0][:-nnz], maxiter=10, x0=x0)
    #print(np.average((sparse_mat.dot(qmrsolve[0]) - test_out[0])**2))
    #qmrsolve = qmr(sparse_mat, test_in[0][:-nnz], maxiter=10, x0=np.random.uniform(0.0, 1.0, size=(sparse_mat.shape[0],)))
    #print(np.average((sparse_mat.dot(qmrsolve[0]) - test_out[0])**2))

plt.plot(trainSamples, rmseResults)
plt.xlabel('number of samples')
plt.ylabel('rmse')
title = 'Training set wise'
plt.title(title)
plt.savefig(title+".png")
plt.show()
