import numpy as np
from scipy.sparse import bsr_matrix
from scipy.linalg import block_diag
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from scipy.sparse.linalg import cg
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

def read_b():
    f = open('rhs.out', 'r')
    b = []
    i=0
    for line in f:
        if (4227 == i):
            break
        if i > 2:
            b.append(float(line))
        else:
            print(line)
        i+=1
#    print(np.array(b).shape)
    return np.array(b) 

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def calMeanForSamples(samples, rmseList):
    sum = 0
    for rmse in rmseList:
        sum = sum + rmse
    return sum/samples   

def injectXInSolver(A, b, initialX):
#    print("A shape:", A.shape)
#    print("b shape:", b.shape)
    sol, info = cg(A, b, x0=initialX, tol=1e-07)
    return info

sparse_mat = read_mat()
b_mat = read_b()
print("sparse:",sparse_mat.shape)
nnz = sparse_mat.nnz	
print(nnz)
model = Sequential()
model.add(Dense(64,input_shape=(nnz + sparse_mat.shape[0],)))
model.add(Activation('tanh'))
model.add(BatchNormalization())
model.add(Dense(sparse_mat.shape[0]))
model.compile(loss='mse',
              optimizer='rmsprop')

#### MARK: running the below script for n number of samples to get
####       n rmse values
totalSamples = [10]
setsNo = [4]
rmseResults = []
iterResults = []
rmseForEachSet = []
meanRmseForEachSet = []
iterationForEachSet = []
meanIterForEachSet = []

for counter, sampleNo in enumerate(totalSamples):
    print("COUNTER NO.", counter)
    print("no. of samples: ", sampleNo)
    print("no. of sets in this run: ", setsNo[counter])
    train_sample=int(0.9*int(sampleNo))  # total number of samples in each set #sys.argv[1]
    print("no. of training samples", train_sample)
    iter=int(setsNo[counter])                   # total number of sets #sys.argv[2]
#    test_sample=int((train_sample*0.1)*iter)# calculating test set
#    test_sample=int((sampleNo*0.1)*iter)
    test_sample = int(sampleNo) - train_sample
    print("no. of test samples", test_sample)
#    test_in, test_out,sparse_append = gen_data_test(sparse_mat, num_samples=test_sample)
#    test_in = np.hstack((test_in, np.repeat(sparse_append,
#                         test_in.shape[0], axis=0)))
#    print(test_in.shape)
    list_in_arrays=[]
    list_out_arrays=[]
    list_sparse_append=[]
    list_test_in_arrays=[]
    list_test_out_arrays=[]
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
        
        test_in, test_out,test_sparse_append = gen_data_test(sparse_mat, num_samples=test_sample)
        test_in = np.hstack((test_in, np.repeat(test_sparse_append,test_in.shape[0], axis=0)))
        print("Test_In shape",test_in.shape)
        list_test_in_arrays.append(test_in)
        list_test_out_arrays.append(test_out)
        
    
    '''
    training the model setwise
    '''
    meanRmseForEachSet.clear
    meanIterForEachSet.clear
    for ind in range(0,iter):
        print("set number:", ind)
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
        rmseForEachSet.clear
        iterationForEachSet.clear
        test_temp_in = list_test_in_arrays[ind][:]
        test_temp_out = list_test_out_arrays[ind][:]
        print(test_temp_in.shape)
        for counter in range(test_temp_in.shape[0]):
            initialXForSet = model.predict(test_temp_in[counter:counter+1]).flatten()
            print("initial X for index:", ind, counter,initialXForSet.shape)
            rmseForEachSet.append(rmse(initialXForSet,test_temp_out[counter:counter+1]))
            # inject initial x in solver
            iterationForEachSet.append(injectXInSolver(sparse_mat, b_mat, initialXForSet))
        # TO DO: Construct a plot for rmse and iterations for each set
        meanRmseForEachSet.append(calMeanForSamples(test_temp_in.shape[0], rmseForEachSet))
        meanIterForEachSet.append(calMeanForSamples(test_temp_in.shape[0], iterationForEachSet))

#    x0 = model.predict(test_in[:1]).flatten()
#    rmse_value=rmse(x0,test_out)
#    rmseResults.append(rmse_value)
#    print(counter, rmse_value)
    #qmrsolve = qmr(sparse_mat, test_in[0][:-nnz], maxiter=10, x0=x0)
    #print(np.average((sparse_mat.dot(qmrsolve[0]) - test_out[0])**2))
    #qmrsolve = qmr(sparse_mat, test_in[0][:-nnz], maxiter=10, x0=np.random.uniform(0.0, 1.0, size=(sparse_mat.shape[0],)))
    #print(np.average((sparse_mat.dot(qmrsolve[0]) - test_out[0])**2))
    print("Mean RMSE for all sets:", meanRmseForEachSet)
    rmseResults.append(calMeanForSamples(iter, meanRmseForEachSet))
    print("Mean Iter for all sets:", meanIterForEachSet)
    iterResults.append(calMeanForSamples(iter, meanIterForEachSet))
    
print("RMSE for total sample:", rmseResults)
plt.plot(totalSamples, rmseResults)
plt.xlabel('number of samples')
plt.ylabel('rmse')
title = 'Training set wise'
plt.title(title)
plt.savefig(title+".png")
plt.show()


print("Iter for total sample:", iterResults)
plt.plot(totalSamples, iterResults)
plt.xlabel('number of samples')
plt.ylabel('mean iterations')
title = 'Iterations set wise'
plt.title(title)
plt.savefig(title+".png")
plt.show()