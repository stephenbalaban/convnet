import numpy as np
import gnumpy as gp
import scipy as scp
import scipy.misc
import cPickle

def load_cifar10_alex(backend):
    X=np.zeros((50000,3072))
    T=np.zeros((50000,10))
    T_train_labels=np.zeros(50000)

    X_test=np.zeros((10000,3072))
    T_test=np.zeros((10000,10))
    T_labels=np.zeros(10000)

    fo = open('./Dataset/CIFAR10/alex/data_batch_1', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    X[:10000]=dict['data'].T
    T_train_labels[:10000]= dict['labels']
    for i in range(10000):
        T[i,dict['labels'][i]]= 1
        
    fo = open('./Dataset/CIFAR10/alex/data_batch_2', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    X[10000:20000]=dict['data'].T
    T_train_labels[10000:20000]= dict['labels']
    for i in range(10000):
        T[i+10000,dict['labels'][i]]= 1
        
    fo = open('./Dataset/CIFAR10/alex/data_batch_3', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    X[20000:30000]=dict['data'].T
    T_train_labels[20000:30000]= dict['labels']
    for i in range(10000):
        T[i+20000,dict['labels'][i]]= 1
        
    fo = open('./Dataset/CIFAR10/alex/data_batch_4', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    X[30000:40000]=dict['data'].T
    T_train_labels[30000:40000]= dict['labels']
    for i in range(10000):
        T[i+30000,dict['labels'][i]]= 1
        
    fo = open('./Dataset/CIFAR10/alex/data_batch_5', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    X[40000:50000]=dict['data'].T
    T_train_labels[40000:50000]= dict['labels']
    for i in range(10000):
        T[i+40000,dict['labels'][i]]= 1
        
    fo = open('./Dataset/CIFAR10/alex/data_batch_6', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    X_test[:10000]=dict['data'].T
    T_labels[:10000]= dict['labels']
    for i in range(10000):
        T_test[i,dict['labels'][i]]= 1

    fo = open('./Dataset/CIFAR10/alex/batches.meta', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    X_mean=dict['data_mean']
    X-=X_mean.T
    X_test-=X_mean.T
    
    if backend=="numpy": X=np.array(X);T=np.array(T);X_test=np.array(X_test);T_test=np.array(T_test);T_train_labels=np.array(T_train_labels);T_labels=np.array(T_labels)
    if backend=="gnumpy": X=gp.garray(X);T=gp.garray(T);X_test=gp.garray(X_test);T_test=gp.garray(T_test);T_train_labels=gp.garray(T_train_labels);T_labels=gp.garray(T_labels)
    
    return X,T,X_test,T_test,T_train_labels,T_labels

def load_cifar10_raw(backend):
    X=np.zeros((50000,3072))
    T=np.zeros((50000,10))
    T_train_labels=np.zeros(50000)

    X_test=np.zeros((10000,3072))
    T_test=np.zeros((10000,10))
    T_labels=np.zeros(10000)

    fo = open('./Dataset/CIFAR10/alex/data_batch_1', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    X[:10000]=dict['data'].T
    T_train_labels[:10000]= dict['labels']
    for i in range(10000):
        T[i,dict['labels'][i]]= 1
        
    fo = open('./Dataset/CIFAR10/alex/data_batch_2', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    X[10000:20000]=dict['data'].T
    T_train_labels[10000:20000]= dict['labels']
    for i in range(10000):
        T[i+10000,dict['labels'][i]]= 1
        
    fo = open('./Dataset/CIFAR10/alex/data_batch_3', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    X[20000:30000]=dict['data'].T
    T_train_labels[20000:30000]= dict['labels']
    for i in range(10000):
        T[i+20000,dict['labels'][i]]= 1
        
    fo = open('./Dataset/CIFAR10/alex/data_batch_4', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    X[30000:40000]=dict['data'].T
    T_train_labels[30000:40000]= dict['labels']
    for i in range(10000):
        T[i+30000,dict['labels'][i]]= 1
        
    fo = open('./Dataset/CIFAR10/alex/data_batch_5', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    X[40000:50000]=dict['data'].T
    T_train_labels[40000:50000]= dict['labels']
    for i in range(10000):
        T[i+40000,dict['labels'][i]]= 1
        
    fo = open('./Dataset/CIFAR10/alex/data_batch_6', 'rb')
    dict = cPickle.load(fo)
    fo.close()
    X_test[:10000]=dict['data'].T
    T_labels[:10000]= dict['labels']
    for i in range(10000):
        T_test[i,dict['labels'][i]]= 1

    # fo = open('./Dataset/CIFAR10/alex/batches.meta', 'rb')
    # dict = cPickle.load(fo)
    # fo.close()
    # X_mean=dict['data_mean']
    # X-=X_mean.T
    # X_test-=X_mean.T
    
    if backend=="numpy": X=np.array(X);T=np.array(T);X_test=np.array(X_test);T_test=np.array(T_test);T_train_labels=np.array(T_train_labels);T_labels=np.array(T_labels)
    if backend=="gnumpy": X=gp.garray(X);T=gp.garray(T);X_test=gp.garray(X_test);T_test=gp.garray(T_test);T_train_labels=gp.garray(T_train_labels);T_labels=gp.garray(T_labels)
    
    return X,T,X_test,T_test,T_train_labels,T_labels    

def load_norb(backend,dtype,resize=64,mode="single"):

    rnd_permute = np.arange(24300)
    data = open('./Dataset/NORB/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat','r')
    data = data.read()
    data = np.fromstring(data[20:], dtype='uint32')
    T_train_labels = data[rnd_permute]
    if (mode == "single" or mode == "parallel" or mode == "binocular"): 
        T = np.zeros((24300,5))
        for n in range(24300):
            T[n,T_train_labels[n]] = 1
    elif mode == "serial":
        T_train_labels = np.concatenate((T_train_labels,T_train_labels),axis=1)
        T = np.zeros((48600,5))
        for n in range(48600):
            T[n,T_train_labels[n]]=1     
    

    if mode == "single": X = np.zeros((24300,resize**2))
    elif mode == "parallel": X = np.zeros((24300,2*resize**2))
    elif mode == "serial": X = np.zeros((48600,resize**2))
    elif mode == "binocular": X = np.zeros((24300,2*resize**2))
    
    data_ = open('./Dataset/NORB/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat','r')
    data_ = data_.read() 
    data_ = np.fromstring(data_[24:], dtype='uint8')
    data_ = np.reshape(data_,(24300,2,96,96))
    

    #X = X[rnd_permute,:]
    if mode == "serial":
        data  = data_[:,0,16:80,16:80]
        for n in range(24300):
            X[n] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
        data = data_[:,1,16:80,16:80]
        for n in range(24300,48600):
            X[n] = scipy.misc.imresize(data[n-24300,:,:], (resize,resize) , 'bilinear').flatten()
    elif mode == "parallel":
        data  = data_[:,0,16:80,16:80]
        for n in range(24300):
            X[n][:resize**2] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
        data = data_[:,1,16:80,16:80]
        for n in range(24300):
            X[n][-resize**2:] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
    elif mode == "single":
        data  = data_[:,0,16:80,16:80]
        for n in range(24300):
            X[n] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
    elif mode == "binocular":
        data0 = data_[:,0,16:80,16:80]
        data1 = data_[:,1,16:80,16:80]
        for n in range(24300):
            a = scipy.misc.imresize(data0[n,:,:], (resize,resize) , 'bilinear')
            b = scipy.misc.imresize(data1[n,:,:], (resize,resize) , 'bilinear')
            X[n] = np.concatenate((a,b),axis=1).ravel()
        
    X = X/255.0


    data=open('./Dataset/NORB/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat','r')
    data=data.read() 
    data=np.fromstring(data[20:], dtype='uint32')
    T_labels=data
    if (mode == "single" or mode == "parallel" or mode == "binocular"): 
        T_test = np.zeros((24300,5))
        for n in range(24300):
            T_test[n,T_labels[n]]=1
    elif mode == "serial":
        T_labels = np.concatenate((T_labels,T_labels),axis=1)
        T_test = np.zeros((48600,5))
        for n in range(48600):
            T_test[n,T_labels[n]]=1        
    # print T_test.shape    

    if mode == "single": X_test = np.zeros((24300,resize**2))
    elif mode == "parallel": X_test = np.zeros((24300,2*resize**2))
    elif mode == "serial": X_test = np.zeros((48600,resize**2))
    elif mode == "binocular": X_test = np.zeros((24300,2*resize**2))

    data_=open('./Dataset/NORB/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat','r')
    data_=data_.read() 
    data_=np.fromstring(data_[24:], dtype='uint8')
    data_=np.reshape(data_,(24300,2,96,96))
    data=data_[:,0,16:80,16:80]
    if mode == "serial":
        data  = data_[:,0,16:80,16:80]
        for n in range(24300):
            X_test[n] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
        data = data_[:,1,16:80,16:80]
        for n in range(24300,48600):
            X_test[n] = scipy.misc.imresize(data[n-24300,:,:], (resize,resize) , 'bilinear').flatten()
    elif mode == "parallel":
        data  = data_[:,0,16:80,16:80]
        for n in range(24300):
            X_test[n][:resize**2] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
        data = data_[:,1,16:80,16:80]
        for n in range(24300):
            X_test[n][-resize**2:] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
    elif mode == "single":
        data  = data_[:,0,16:80,16:80]
        for n in range(24300):
            X_test[n] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
    elif mode == "binocular":
        data0 = data_[:,0,16:80,16:80]
        data1 = data_[:,1,16:80,16:80]
        for n in range(24300):
            a = scipy.misc.imresize(data0[n,:,:], (resize,resize) , 'bilinear')
            b = scipy.misc.imresize(data1[n,:,:], (resize,resize) , 'bilinear')
            X_test[n] = np.concatenate((a,b),axis=1).ravel()
    # X = X[rnd_permute,:]
    X_test = X_test/255.0
    # print X.shape # print X_test.shape

    if backend=="numpy": X=np.array(X,dtype);T=np.array(T,dtype);X_test=np.array(X_test,dtype);T_test=np.array(T_test,dtype);T_train_labels=np.array(T_train_labels,dtype);T_labels=np.array(T_labels,dtype)
    # if backend=="gnumpy": X=gp.garray(X);T=gp.garray(T)
    if backend=="gnumpy": X=gp.garray(X);T=gp.garray(T);X_test=gp.garray(X_test);T_test=gp.garray(T_test);T_train_labels=gp.garray(T_train_labels);T_labels=gp.garray(T_labels)
    
    #return X,T
    return X,T,X_test,T_test,T_train_labels,T_labels

def load_mnist(backend,dtype):
    global s
    s=60000

    T=np.zeros((s,10))
    
    data_=open('./Dataset/MNIST/train-images.idx3-ubyte','r')
    data=data_.read() 
    data_=np.fromstring(data[16:], dtype='uint8')
    X=np.reshape(data_,(s,784))/255.0
 
    data_=open('./Dataset/MNIST/train-labels.idx1-ubyte','r')
    data=data_.read()
    T_train_labels = np.fromstring(data[8:], dtype='uint8')

    for n in range(s):
        T[n,T_train_labels[n]]=1
        


    s_test=10000
    X_test=np.zeros((s_test,784))
    T_test=np.zeros((s_test,10))


    data_=open('./Dataset/MNIST/t10k-images.idx3-ubyte','r')
    data=data_.read()
    data_=np.fromstring(data[16:], dtype='uint8')
    X_test=np.reshape(data_,(s_test,784))/255.0


    data_=open('./Dataset/MNIST/t10k-labels.idx1-ubyte','r')
    data=data_.read()
    T_labels = np.fromstring(data[8:], dtype='uint8')

    for n in range(s_test):
        T_test[n,T_labels[n]]=1
    
    if backend=="numpy": X=np.array(X,dtype);T=np.array(T,dtype);X_test=np.array(X_test,dtype);T_test=np.array(T_test,dtype);T_train_labels=np.array(T_train_labels,dtype);T_labels=np.array(T_labels,dtype)
    if backend=="gnumpy": X=gp.garray(X);T=gp.garray(T);X_test=gp.garray(X_test);T_test=gp.garray(T_test);T_train_labels=gp.garray(T_train_labels);T_labels=gp.garray(T_labels)
   
    return X,T,X_test,T_test,T_train_labels,T_labels,s