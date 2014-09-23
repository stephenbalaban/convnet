import numpy as np
import gnumpy as gp
import pylab as plt
import gnumpy.cudamat_gnumpy as ConvNet


class GnumpyBackend(object):

    @staticmethod
    def AvgPoolUndo(images,grad,subsX,startX,strideX):
        return ConvNet.AvgPoolUndo(images,grad,subsX,startX,strideX)

    @staticmethod
    def AvgPool(images,subsX,startX,strideX,outputsX):
        return ConvNet.AvgPool(images,subsX,startX,strideX,outputsX)

    @staticmethod
    def MaxPoolUndo(images,grad,maxes,subsX,startX,strideX):
        return ConvNet.MaxPoolUndo(images,grad,maxes,subsX,startX,strideX)

    @staticmethod
    def MaxPool(images,subsX,startX,strideX,outputsX):
        return ConvNet.MaxPool(images,subsX,startX,strideX,outputsX)

    @staticmethod
    def ConvOut(images, hidActs, moduleStride, paddingStart):
        if hidActs.shape[0]==1:
            hidActs_16 = gp.zeros((16,hidActs.shape[1],hidActs.shape[2],hidActs.shape[3]))
            hidActs_16[:1,:,:,:] = hidActs
            out = ConvNet.convOutp(images, hidActs_16, moduleStride = moduleStride, paddingStart = paddingStart)
            return out[:,:,:,:1]
        elif hidActs.shape[0]==3:
            hidActs_16 = gp.zeros((16,hidActs.shape[1],hidActs.shape[2],hidActs.shape[3]))
            hidActs_16[:3,:,:,:] = hidActs
            out = ConvNet.convOutp(images, hidActs_16, moduleStride = moduleStride, paddingStart = paddingStart)
            return out[:,:,:,:3]            
        elif hidActs.shape[0]%16 == 0:
            return ConvNet.convOutp(images,hidActs, moduleStride, paddingStart)
        else: raise Exception("Hidden Mode 16")

    @staticmethod
    def ConvDown(hidActs, filters, moduleStride, paddingStart):
        if filters.shape[3]==1 and hidActs.shape[0]==1: 
            hidActs_16 = gp.zeros((16,hidActs.shape[1],hidActs.shape[2],hidActs.shape[3]))
            hidActs_16[:1,:,:,:] = hidActs
            filters_16 = gp.zeros((filters.shape[0],filters.shape[1],filters.shape[2],16))
            filters_16[:,:,:,:1] = filters
            return ConvNet.convDown(hidActs_16, filters_16 , moduleStride=moduleStride , paddingStart = paddingStart)
        elif filters.shape[3]==3 and hidActs.shape[0]==3: 
            hidActs_16 = gp.zeros((16,hidActs.shape[1],hidActs.shape[2],hidActs.shape[3]))
            hidActs_16[:3,:,:,:] = hidActs
            filters_16 = gp.zeros((filters.shape[0],filters.shape[1],filters.shape[2],16))
            filters_16[:,:,:,:3] = filters
            return ConvNet.convDown(hidActs_16, filters_16 , moduleStride=moduleStride , paddingStart = paddingStart)            
        elif filters.shape[3]%16==0 and hidActs.shape[0]%16==0:
            return ConvNet.convDown(hidActs, filters, moduleStride, paddingStart)
        else: raise Exception("Hidden or Filters Mode 16")

    @staticmethod
    def ConvUp(images, filters, moduleStride, paddingStart):
        if filters.shape[3]==1: 
                filters_16 = gp.zeros((filters.shape[0],filters.shape[1],filters.shape[2],16))
                filters_16[:,:,:,:1]=filters
                out = ConvNet.convUp(images, filters_16 ,moduleStride = moduleStride, paddingStart = paddingStart)
                return out[:1,:,:,:]
        elif filters.shape[3]==3: 
                filters_16 = gp.zeros((filters.shape[0],filters.shape[1],filters.shape[2],16))
                filters_16[:,:,:,:3]=filters
                out = ConvNet.convUp(images, filters_16 ,moduleStride = moduleStride, paddingStart = paddingStart)
                return out[:3,:,:,:]                
        elif filters.shape[3]%16==0: 
                return ConvNet.convUp(images, filters , moduleStride, paddingStart)
        else: raise Exception("Filters Mode 16")


    @staticmethod
    def ConvUp_single(a_gp, f_gp,moduleStride, paddingStart):
        assert f_gp.shape[3]==1
        f_16 = gp.zeros((f_gp.shape[0],f_gp.shape[1],f_gp.shape[2],16))
        f_16[:,:,:,:1]=f_gp
        q_gp = ConvNet.convUp(a_gp, f_16 ,moduleStride = moduleStride, paddingStart = paddingStart)
        return q_gp[:1,:,:,:]

    @staticmethod
    def argsort(x):
        return gp.garray(np.argsort(x.as_numpy_array()))

    @staticmethod
    def l2_normalize(w):
        l2=gp.sum(w**2,axis=0)**(1./2)
        w[:]=w/l2

    @staticmethod
    def bitwise_or(x,y):
        return x | y
    
    @staticmethod
    def threshold_mask_hard(x,k,mask=None,dropout=None):
        if type(x)==gp.garray: x_ = x.as_numpy_array()
        else: x_ = x
        if dropout!=None:
            dropout_mask =  (np.random.rand(x_.shape[0])>(1-dropout))
        c=np.zeros(x_.shape)
        if k==1: 
            loc=np.arange(x_.shape[0]),x_.argmax(1)
        else: 
            b=np.argsort(x_,kind='quicksort',axis=1)
            loc=np.repeat(np.arange(x_.shape[0]),k),b[:,-k:].flatten()
        c[loc]=1
        if type(x)==gp.garray:
            if dropout!=None: return gp.garray(dropout_mask[:,newaxis]*c)
            else: return gp.garray(c)
        else:
            if dropout!=None: return dropout_mask[:,np.newaxis]*c
            else: return c              

    @staticmethod
    def abs(x): return gp.sign(x)*x
    
    @staticmethod
    def sign(x): return gp.sign(x)
    
    @staticmethod
    def threshold_mask_soft(x,k,dropout=None):
        b=k*gp.std(x,axis=1)[:,gp.newaxis]
        std_matrix=gp.dot(b,gp.ones((1,x.shape[1])))
        if dropout==None: return (x>std_matrix)
        return (x>std_matrix)*(gp.rand(x.shape)>(1-dropout))
    
    @staticmethod
    def mask(x,dropout=1):
        return (gp.rand(x.shape)>(1-dropout))    

    @staticmethod
    def zeros(shape,dtype):
        if type(shape)!=tuple: return gp.zeros(shape)
        return gp.zeros(shape)
    
    @staticmethod
    def ones(shape,dtype):
        if type(shape)!=tuple: return gp.ones(shape)
        return gp.ones(*shape)    
    
    @staticmethod
    def rand(shape,dtype):    return gp.rand(*shape)
    
    @staticmethod
    def rand_binary(shape,dtype):    return gp.rand(*shape)>.5
    
    @staticmethod
    def randn(shape,dtype):    
        if type(shape)!=tuple: return gp.randn(shape)
        return gp.randn(*shape)
    
    @staticmethod
    def array(A,dtype):  return gp.garray(A)
    
    @staticmethod
    def dot(A,B):    return gp.dot(A,B)
    
    @staticmethod
    def exp(A):      return gp.exp(A)
    
    @staticmethod
    def log(A):      return gp.log(A)

    @staticmethod
    def max(A,axis): return gp.max(A,axis=axis)
    
    @staticmethod
    def min(A,axis): return gp.min(A,axis=axis)
    
    @staticmethod
    def sum(A,axis): return gp.sum(A,axis=axis)
    
    @staticmethod
    def mean(A,axis): return gp.mean(A,axis=axis)
    
    @staticmethod
    def sigmoid(x):
        den = 1.0 + gp.exp (-1.0 * x)
        d = 1.0 / den
        return d
    
    @staticmethod
    def sigmoid_prime(x):
        den = 1.0 + gp.exp(-1.0 * x)
        d = (gp.exp(-1.0 * x)) / den**2
        return d
    
    @staticmethod
    def relu(x): return gp.garray(x>0)*x
    
    @staticmethod
    def relu_prime(x): return gp.garray(x>0)
    
    @staticmethod
    def relu_squared(x): return gp.garray(x>0)*(x**2)

    @staticmethod
    def relu_squared_prime(x): return gp.garray(x>0)*(2*x)

    @staticmethod
    def relu_sigma_1(x): 
        b=2*gp.std(x,axis=1)[:,gp.newaxis]
        std_matrix=gp.dot(b,gp.ones((1,x.shape[1])))
        return ((x-std_matrix)>0)*(x-std_matrix)+((x+std_matrix)<0)*(x+std_matrix)
        
    @staticmethod
    def relu_sigma_1_prime(x): 
        b=2*gp.std(x,axis=1)[:,gp.newaxis]
        std_matrix=gp.dot(b,gp.ones((1,x.shape[1])))
        return (x>std_matrix)+(x<-std_matrix)
    
    @staticmethod
    def relu_5(x): return gp.garray(x>.05)*(x-.05)+gp.garray(x<-.05)*(x+.05)
    
    @staticmethod
    def relu_5_prime(x): return gp.garray(x>0.05)+gp.garray(x<-.05)
    
    @staticmethod
    def softmax_old(x):
        y=gp.max(x,axis=1)[:,gp.newaxis]
        logsumexp=y+gp.log(gp.sum((gp.exp(x-y)),axis=1))[:,gp.newaxis]
        return gp.exp(x-logsumexp)
    
    @staticmethod
    def softmax(A):
        A -= gp.max(A,axis=1)[:,gp.newaxis]
        Z  = gp.exp(A)
        return Z / gp.sum(Z,axis=1)[:,gp.newaxis]

    @staticmethod
    def softmax1(A):
        Z  = gp.exp(A)
        return Z / gp.sum(Z,axis=1)[:,gp.newaxis]

    @staticmethod
    def softmax_grounded(b):
        z=gp.zeros((b.shape[0],1))
        b_=gp.concatenate((z,b),axis=1)
        y_=gp.exp(b_)
        return y_ / (y_.sum(1)[:,gp.newaxis])
    
    @staticmethod
    def linear(x): return x
    
    @staticmethod
    def linear_prime(x):
        return gp.ones(x.shape)
    
    @staticmethod
    def relu_truncated(x):
        y=x.copy()
        y[gp.where(x>.9999)]=.9999*gp.ones(x[gp.where(x>.9999)].shape)
        y[gp.where(x<.00001)]=.00001*gp.ones(x[gp.where(x<.00001)].shape)
        return y

    @staticmethod
    def relu1(x): return gp.garray(x>0)*x-gp.garray(x>1)*(x-1)

    @staticmethod
    def relu1_prime(x): return gp.garray(x>0)-gp.garray(x>1)
    
    @staticmethod
    def relu_prime_truncated(x):
        y=gp.ones(x.shape)
        #if y>.9999: print 'salam'
        y[gp.where(x>.9999)]=gp.zeros(x[gp.where(x>.9999)].shape)
        y[gp.where(x<.00001)]=gp.zeros(x[gp.where(x<.00001)].shape)
        return y
    
    @staticmethod
    def KL(rho,rho_target,KL_flat): 
        y=rho.copy()
        if KL_flat: y[gp.where(y<rho_target)]=rho_target*gp.ones(y[gp.where(y<rho_target)].shape)
        return rho_target*gp.log(rho_target/y)+(1-rho_target)*gp.log((1-rho_target)/(1-y))
    
    @staticmethod
    def d_KL(rho,rho_target,KL_flat): 
        y=rho.copy()
        if KL_flat: y[gp.where(y<rho_target)]=rho_target*gp.ones(y[gp.where(y<rho_target)].shape)
        return -rho_target/y+(1-rho_target)/(1-y)
    
    @staticmethod
    def exp_penalty(x,sigma): return x.shape[1]-((gp.exp(-x**2/sigma)).sum())/x.shape[0]
    
    @staticmethod
    def d_exp_penalty(x,sigma): return ((2*(1/sigma)*x*gp.exp(-x**2/sigma)))

    #@staticmethod
    #def newaxis(): return gp.newaxis()



#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/

class NumpyBackend(object):

    @staticmethod
    def AvgPoolUndo(images,grad,subsX,startX,strideX): 
        numChannels, imSizeX_, imSizeX, numImages = images.shape    
        assert imSizeX_ == imSizeX
        numImgColors = numChannels
        numChannels, outputsX_, outputsX, numImages = grad.shape
        assert outputsX_ == outputsX
        targets = np.zeros(images.shape)
        for i in range(numImages):
            for c in range(numChannels):
                o1 = 0
                for s1 in range(startX, imSizeX, strideX):
                    if s1<0:
                        continue
                    o2 = 0
                    for s2 in range(startX, imSizeX, strideX):
                        if s2<0:
                            continue
                        for u1 in range(subsX):
                            for u2 in range(subsX):
                                try:
                                    #if maxes[c,o1,o2,i]==images[c,s1+u1,s2+u2,i]:
                                    targets[c,s1+u1,s2+u2,i]+=grad[c,o1,o2,i]

                                except IndexError:
                                    pass
                        o2 += 1
                    o1 += 1
        return targets/subsX**2


    @staticmethod
    def AvgPool(images,subsX,startX,strideX,outputsX):        
        numChannels, imSizeX, imSizeX, numImages = images.shape    
        numImgColors = numChannels
        targets = np.zeros((numChannels, outputsX, outputsX, numImages))
        for i in range(numImages):
            for c in range(numChannels):
                o1 = 0
                for s1 in range(startX, imSizeX, strideX):
                    o2 = 0
                    for s2 in range(startX, imSizeX, strideX):
                        for u1 in range(subsX):
                            for u2 in range(subsX):
                                try:
                                    targets[c,o1,o2,i] += images[c,s1+u1,s2+u2,i]                           
                                except IndexError:
                                    pass #?
                        o2 += 1
                    o1 += 1
        return 1.0*targets/subsX**2

    @staticmethod
    def MaxPoolUndo(images,grad,maxes,subsX,startX,strideX):
        numChannels, imSizeX_, imSizeX, numImages = images.shape    
        assert imSizeX_ == imSizeX
        numImgColors = numChannels
        numChannels, outputsX_, outputsX, numImages = maxes.shape
        assert outputsX_ == outputsX    
        assert maxes.shape == grad.shape
        targets = np.zeros(images.shape)
        for i in range(numImages):
            for c in range(numChannels):
                o1 = 0
                for s1 in range(startX, imSizeX, strideX): 
                    if s1<0:
                        continue
                    o2 = 0
                    for s2 in range(startX, imSizeX, strideX):
                        if s2<0:
                            continue
                        for u1 in range(subsX):
                            for u2 in range(subsX):
                                try:
                                    if maxes[c,o1,o2,i]==images[c,s1+u1,s2+u2,i]:
                                        targets[c,s1+u1,s2+u2,i]+=grad[c,o1,o2,i]
                                        break
                                except IndexError:
                                    pass                        
                            else: continue
                            break
                        o2 += 1
                    o1 += 1
        return targets

    @staticmethod
    def MaxPool(images,subsX,startX,strideX,outputsX):
        numChannels, imSizeX, imSizeX, numImages = images.shape    
        numImgColors = numChannels
        targets = np.zeros((numChannels, outputsX, outputsX, numImages)) - 1e100
        def max(a,b):
            return a if a>b else b
        for i in range(numImages):
            for c in range(numChannels):
                o1 = 0
                for s1 in range(startX, imSizeX, strideX):
                    if s1<0:
                        continue
                    o2 = 0
                    for s2 in range(startX, imSizeX, strideX):
                        if s2<0:
                            continue
                        for u1 in range(subsX):
                            for u2 in range(subsX):
                                try:
                                    targets[c,o1,o2,i] = max(images[c,s1+u1,s2+u2,i],targets[c,o1,o2,i])
                                except IndexError:
                                    pass #?                           
                        o2 += 1
                    o1 += 1
        return targets

    @staticmethod
    def ConvOut_old(images, hidActs, moduleStride=1,paddingStart = 0):
            numGroups = 1
            assert paddingStart <= 0
            numFilters, numModulesX, numModulesX, numImages = hidActs.shape
            numChannels, imSizeX, imSizeX, numImages = images.shape    
            numFilterChannels = numChannels / numGroups

            filterSizeX = imSizeX - moduleStride*(numModulesX - 1) + 2*abs(paddingStart)
            targets = np.zeros((numFilterChannels, filterSizeX, filterSizeX, numFilters))
            numImgColors = numChannels
            images2 = np.zeros((numChannels,imSizeX+2*abs(paddingStart),imSizeX+2*abs(paddingStart),numImages))
            if paddingStart != 0:
                images2[:, 
                    abs(paddingStart):-abs(paddingStart),
                    abs(paddingStart):-abs(paddingStart),
                    :] = images
            else:
                images2 = images
            for i in range(numImages):
                for f in range(numFilters):
                    for c in range(numChannels):
                        for y1 in range(numModulesX):
                            for y2 in range(numModulesX):
                                for u1 in range(filterSizeX):
                                    for u2 in range(filterSizeX):
                                        x1 = y1*moduleStride + u1 
                                        x2 = y2*moduleStride + u2
                                        targets[c ,u1,u2,f] += hidActs[f, y1, y2, i] * images2[c,x1,x2,i]
            return targets

    @staticmethod
    def ConvOut(images, hidActs, moduleStride=1,paddingStart = 0):
        assert paddingStart <= 0
        numFilters, numModulesX, numModulesX, numImages = hidActs.shape
        numChannels, imSizeX, imSizeX, numImages = images.shape    
        numFilterChannels = numChannels
        filterSizeX = imSizeX - moduleStride*(numModulesX - 1) + 2*abs(paddingStart)
        targets = np.zeros((numFilterChannels, filterSizeX, filterSizeX, numFilters))
        numImgColors = numChannels
        images2 = np.zeros((numChannels,imSizeX+2*abs(paddingStart),imSizeX+2*abs(paddingStart),numImages))
        if paddingStart != 0:
            images2[:, 
                abs(paddingStart):-abs(paddingStart),
                abs(paddingStart):-abs(paddingStart),
                :] = images
        else:
            images2 = images

        numChannels, imSizeX2, imSizeX2, numImages = images2.shape
        filters_2d = targets.reshape(numFilterChannels*filterSizeX*filterSizeX,numFilters)              
            
        for i in range(0,imSizeX2-filterSizeX+1,moduleStride):
            for j in range(0,imSizeX2-filterSizeX+1,moduleStride):
                images_patch = images2[:,i:i+filterSizeX,j:j+filterSizeX,:].reshape(numChannels*filterSizeX**2,-1)
                hidden_patch = hidActs[:,i/moduleStride,j/moduleStride,:].T
                filters_2d[:] += np.dot(images_patch,hidden_patch)

        return targets

    @staticmethod
    def ConvDown_old(hidActs, filters, moduleStride , paddingStart):
        numGroups = 1
        assert paddingStart <= 0
        numFilters, numModulesX, numModulesX, numImages = hidActs.shape
        numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape

        imSizeX = moduleStride*(numModulesX - 1) - 2*abs(paddingStart) + filterSizeX
        imSizeX2 = moduleStride*(numModulesX - 1) + filterSizeX

        numChannels = numFilterChannels * numGroups
        numModules = numModulesX**2 
        
        targets = np.zeros((numChannels, imSizeX, imSizeX, numImages))
        targets2 = np.zeros((numChannels, imSizeX2, imSizeX2, numImages))
        
        for i in range(numImages):
            for f in range(numFilters):
                for c in range(numChannels):
                    for y1 in range(numModulesX):
                        for y2 in range(numModulesX):
                            for u1 in range(filterSizeX):
                                for u2 in range(filterSizeX):
                                    x1 = y1*moduleStride + u1 
                                    x2 = y2*moduleStride + u2
                                    targets2[c,x1,x2,i] += filters[c ,u1,u2,f] * hidActs[f, y1, y2, i]
        if paddingStart != 0:
            targets[:] = targets2[:, 
                abs(paddingStart):-abs(paddingStart),
                abs(paddingStart):-abs(paddingStart),
                :] 
        else:
            targets = targets2
        return targets

    @staticmethod
    def ConvDown(hidActs, filters, moduleStride , paddingStart):
        assert paddingStart <= 0
        numFilters, numModulesX, numModulesX, numImages = hidActs.shape
        numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape
        imSizeX = moduleStride*(numModulesX - 1) + filterSizeX
        targets2 = np.zeros((numFilterChannels, imSizeX, imSizeX, numImages))
        filters_2d = filters.reshape(numFilterChannels*filterSizeX*filterSizeX,numFilters)
        
        for i in range(0,imSizeX-filterSizeX+1,moduleStride):
            for j in range(0,imSizeX-filterSizeX+1,moduleStride):
                targets2_patch = targets2[:,i:i+filterSizeX,j:j+filterSizeX,:]
                hidden_patch = hidActs[:,i/moduleStride,j/moduleStride,:]
                targets2_patch[:] += np.dot(hidden_patch.T,filters_2d.T).T.reshape(numFilterChannels,filterSizeX,filterSizeX,-1)
                
        targets = targets2[:,
                abs(paddingStart):imSizeX-abs(paddingStart),
                abs(paddingStart):imSizeX-abs(paddingStart),
                :] 

        return targets

    @staticmethod
    def ConvUp_old(images, filters, moduleStride, paddingStart):
        global images2
        assert paddingStart <= 0
        numChannels, imSizeX, imSizeX, numImages = images.shape
        numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape
        assert (2*abs(paddingStart) + imSizeX - filterSizeX) % moduleStride == 0
        assert        (2*abs(paddingStart) + imSizeX - filterSizeX) % moduleStride == 0
        numModulesX = (2*abs(paddingStart) + imSizeX - filterSizeX)/moduleStride+1
        numModules = numModulesX**2 
        numGroups = 1
        targets = np.zeros((numFilters, numModulesX, numModulesX, numImages))
        images2 = np.zeros((numChannels, 
                            imSizeX+2*abs(paddingStart), 
                            imSizeX+2*abs(paddingStart), 
                            numImages))  
        if paddingStart != 0:
            images2[:, 
                abs(paddingStart):-abs(paddingStart),
                abs(paddingStart):-abs(paddingStart),
                :] = images
        else:
            images2 = images
        for i in range(numImages):
            for f in range(numFilters):
                for c in range(numChannels):
                    for y1 in range(numModulesX):
                        for y2 in range(numModulesX):
                            for u1 in range(filterSizeX):
                                for u2 in range(filterSizeX):
                                    x1 = y1*moduleStride + u1 
                                    x2 = y2*moduleStride + u2
                                    targets[f, y1, y2, i] += filters[c ,u1,u2,f] * images2[c,x1,x2,i]
        return targets

    @staticmethod
    def ConvUp(images, filters, moduleStride, paddingStart):
        global images2
        assert paddingStart <= 0
        numChannels, imSizeX, imSizeX, numImages = images.shape
        numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape
        assert (2*abs(paddingStart) + imSizeX - filterSizeX) % moduleStride == 0
        assert        (2*abs(paddingStart) + imSizeX - filterSizeX) % moduleStride == 0
        numModulesX = (2*abs(paddingStart) + imSizeX - filterSizeX)/moduleStride+1
        targets = np.zeros((numFilters, numModulesX, numModulesX, numImages))
        images2 = np.zeros((numChannels, 
                            imSizeX+2*abs(paddingStart), 
                            imSizeX+2*abs(paddingStart), 
                            numImages))  
        if paddingStart != 0:
            images2[:, 
                abs(paddingStart):-abs(paddingStart),
                abs(paddingStart):-abs(paddingStart),
                :] = images
        else:
            images2 = images

        numChannels, imSizeX2, imSizeX2, numImages = images2.shape
        filters_2d = filters.reshape(numFilterChannels*filterSizeX*filterSizeX,numFilters)
        
        for i in range(0,imSizeX2-filterSizeX+1,moduleStride):
            for j in range(0,imSizeX2-filterSizeX+1,moduleStride):
                images_patch = images2[:,i:i+filterSizeX,j:j+filterSizeX,:].reshape(numChannels*filterSizeX**2,-1).T
                targets_patch = np.dot(images_patch,filters_2d)
                targets[:,i/moduleStride,j/moduleStride,:]=targets_patch.T
                
        return targets

    @staticmethod
    def argsort(x):
        return np.argsort(x)

    @staticmethod
    def l2_normalize(w):
        l2=np.sum(w**2,axis=0)**(1./2)
        w[:]=w/l2

    @staticmethod
    def bitwise_or(x,y):
        return np.float64(np.int64(x) | np.int64(y))

    @staticmethod
    def abs(x): return np.absolute(x)

    @staticmethod
    def sign(x): return np.sign(x)

    @staticmethod
    def threshold_mask_soft(x,k,mask=None,dropout=None):
        if mask!=None: x *= mask
        b=k*np.std(x,axis=1)[:,np.newaxis]
        std_matrix=np.dot(b,np.ones((1,x.shape[1])))
        if dropout==None: return np.array(x>std_matrix,dtype=x.dtype)
        return np.array(x>std_matrix,dtype=x.dtype)*np.array((np.random.rand(*x.shape)>(1-dropout)),dtype=x.dtype)
    
    @staticmethod
    def mask(x,dropout=1):
        return np.array((np.random.rand(*x.shape)>(1-dropout)),dtype=x.dtype)

    @staticmethod
    def threshold_mask_hard(x,k,mask=None,dropout=None):
        #if dropout!=None: x*=np.array((np.random.rand(*x.shape)>(1-dropout)),dtype=x.dtype)
        if mask!=None: x *= mask
        c=np.zeros(x.shape)
        if dropout==-1: b=np.argsort(np.absolute(x),kind='quicksort',axis=1)
        else:           b=np.argsort(x,kind='quicksort',axis=1)
        loc=np.repeat(np.arange(x.shape[0]),k),b[:,-k:].flatten()
        c[loc]=1
        if (dropout==None or dropout==-1): return c
        if dropout==-2: return 1-c
        return c*np.array((np.random.rand(*x.shape)>(1-dropout)),dtype=x.dtype)

    @staticmethod
    def zeros(shape,dtype):
        if type(shape)!=tuple: return np.array(np.zeros(shape),dtype)
        return np.array(np.zeros(shape),dtype)
    
    @staticmethod
    def ones(shape,dtype):
        if type(shape)!=tuple: return np.array(np.ones(shape),dtype)
        return np.array(np.ones(*shape),dtype)
    
    @staticmethod
    def rand(shape,dtype):    return np.array(np.random.rand(*shape),dtype)
    
    @staticmethod
    def rand_binary(shape,dtype):    return np.array(np.random.rand(*shape)>.5,dtype)

    @staticmethod
    def randn(shape,dtype):    
        if type(shape)!=tuple: return np.array(np.random.randn(shape),dtype)
        return np.array(np.random.randn(*shape),dtype)
    
    @staticmethod
    def array(A,dtype):  return np.array(A,dtype)
    
    @staticmethod
    def dot(A,B):    return np.dot(A,B)
    
    @staticmethod
    def exp(A):      return np.exp(A)
    
    @staticmethod
    def log(A):      return np.log(A)
    
    @staticmethod
    def max(A,axis): return np.max(A,axis=axis)
    
    @staticmethod
    def min(A,axis): return np.min(A,axis=axis)
    
    @staticmethod
    def sum(A,axis): return np.sum(A,axis=axis)
    
    @staticmethod
    def mean(A,axis): return np.mean(A,axis=axis)
    
    @staticmethod
    def sigmoid(x):
        den = 1.0 + np.e ** (-1.0 * x)
        d = 1.0 / den
        return d
    
    @staticmethod
    def sigmoid_prime(x):
        den = 1.0 + np.e ** (-1.0 * x)
        d = (np.e ** (-1.0 * x)) / den**2
        return d
    
    @staticmethod
    def relu(x): return np.array(x>0,dtype=x.dtype)*x
    
    @staticmethod
    def relu_prime(x): return np.array(x>0,dtype=x.dtype)
    
    @staticmethod
    def relu_squared(x): return np.array(x>0,dtype=x.dtype)*(x**2)
    
    @staticmethod
    def relu_squared_prime(x): return np.array(x>0,dtype=x.dtype)*(2*x)

    @staticmethod
    def relu1(x): return np.array(x>0,dtype=x.dtype)*x-np.array(x>1,dtype=x.dtype)*(x-1)

    @staticmethod
    def relu1_prime(x): return np.array(x>0,dtype=x.dtype)-np.array(x>1,dtype=x.dtype)
    
    @staticmethod
    def softmax_old(x):
        y=x.max(1)[:,np.newaxis]
        logsumexp=y+np.log((np.exp(x-y)).sum(1))[:,np.newaxis]
        return np.exp(x-logsumexp)
    
    @staticmethod
    def softmax(A):
        A -= np.max(A,axis=1)[:,np.newaxis]
        # A = A.astype(float)    ##############################################3why
        # print type(A)
        # print A.dtype
        Z  = np.exp(A)
        return Z / np.sum(Z,axis=1)[:,np.newaxis]  
    
    @staticmethod
    def softmax_grounded(b):
        b_=np.insert(b, 0, 0, axis=1)
        y_=np.exp(b_)
        return y_ / (y_.sum(1)[:,np.newaxis])
    
    @staticmethod
    def linear(x): return x
    
    @staticmethod
    def linear_prime(x):
        return np.ones(x.shape)
    
    @staticmethod
    def relu_truncated(x):
        y=x.copy()
        y[x>.9999]=.9999
        y[x<.00001]=.00001
        return y
    
    @staticmethod
    def relu_prime_truncated(x):
        y=np.ones(x.shape)
        #if y>.9999: print 'salam'
        y[x>.9999]=0
        y[x<.00001]=0
        return y
    
    @staticmethod
    def KL(rho,rho_target,KL_flat): 
        y=rho.copy()
        if KL_flat: y[y<rho_target]=rho_target
        return rho_target*np.log(rho_target/y)+(1-rho_target)*np.log((1-rho_target)/(1-y))
    
    @staticmethod
    def d_KL(rho,rho_target,KL_flat): 
        y=rho.copy()
        if KL_flat: y[y<rho_target]=rho_target
        return -rho_target/y+(1-rho_target)/(1-y)
    
    @staticmethod
    def exp_penalty(x,sigma): return x.shape[1]-((np.exp(-x**2/sigma)).sum())/x.shape[0]
    
    @staticmethod
    def d_exp_penalty(x,sigma): return ((2*(1/sigma)*x*np.exp(-x**2/sigma)))

    #@staticmethod
    #def newaxis(): return np.newaxis
    
def percent(a):
    perc=zeros((10,10))
    for h in range(10):
        p_temp = percentile(a,10*(10-h),axis=1)
        for t in range(10):
            perc[h,t]= percentile(p_temp,10*(10-t))
    return perc

def l2_norm(threshold,j):
    l2=np.sum(weights[layers_sum[j]-num_w[j]:layers_sum[j]-layers[j+1]].reshape((layers[j],layers[j+1]))**2,axis=0)**(1./2)
    weights[layers_sum[j]-num_w[j]:layers_sum[j]-layers[j+1]].reshape((layers[j],layers[j+1]))[:,l2>threshold]=weights[layers_sum[j]-num_w[j]:layers_sum[j]-layers[j+1]].reshape((layers[j],layers[j+1]))[:,l2>threshold]/l2[l2>threshold]*threshold

def l1_norm(threshold,j):
    l1=np.sum(np.abs(weights[layers_sum[j]-num_w[j]:layers_sum[j]-layers[j+1]].reshape((layers[j],layers[j+1]))),axis=0)
    weights[layers_sum[j]-num_w[j]:layers_sum[j]-layers[j+1]].reshape((layers[j],layers[j+1]))[:,l1>threshold]=weights[layers_sum[j]-num_w[j]:layers_sum[j]-layers[j+1]].reshape((layers[j],layers[j+1]))[:,l1>threshold]/l1[l1>threshold]*threshold
   

#backend=GnumpyBackend

def zeros(shape,dtype="float64"):return backend.zeros(shape,dtype)
def ones(shape,dtype="float64"):return backend.ones(shape,dtype)
def rand(shape,dtype):return backend.rand(shape,dtype)
def rand_binary(shape,dtype):return backend.rand_binary(shape,dtype)
def randn(shape,dtype="float64"):return backend.randn(shape,dtype)
def array(A,dtype):return backend.array(A,dtype)
def dot(A,B):return backend.dot(A,B)
def exp(A):return backend.exp(A)
def log(A):return backend.log(A)
def max(A,axis):return backend.max(A,axis)
def min(A,axis):return backend.min(A,axis)
def sum(A,axis=None):return backend.sum(A,axis)
def mean(A,axis):return backend.mean(A,axis)
def sigmoid(x):return backend.sigmoid(x)
def sigmoid_prime(x):return backend.sigmoid_prime(x)
def relu(x):return backend.relu(x)
def relu_prime(x):return backend.relu_prime(x)
def softmax(x):return backend.softmax(x)
def softmax_grounded(b):return backend.softmax_grounded(b)
def linear(x):return backend.linear(x)
def linear_prime(x):return backend.linear_prime(x)
#def relu_truncated(x):return backend.relu_truncated(x)
#def relu_prime_truncated(x):return backend.relu_prime_truncated(x)
def KL(rho,rho_target,KL_flat):return backend.KL(rho,rho_target,KL_flat)
def d_KL(rho,rho_target,KL_flat):return backend.d_KL(rho,rho_target,KL_flat)
def exp_penalty(x,sigma):return backend.exp_penalty(x,sigma)
def d_exp_penalty(x,sigma):return backend.d_exp_penalty(x,sigma)
#def newaxis():return backend.newaxis()
def relu1(x):return backend.relu1(x)
def relu1_prime(x):return backend.relu1_prime(x)
def relu_5(x):return backend.relu_5(x)
def relu_5_prime(x):return backend.relu_5_prime(x)
def threshold_mask_soft(x,k,dropout=None):return backend.threshold_mask_soft(x,k,dropout)
def threshold_mask_hard(x,k,mask=None,dropout=None):return backend.threshold_mask_hard(x,k,mask,dropout)
def relu_sigma_1(x):return backend.relu_sigma_1(x)
def relu_sigma_1_prime(x):return backend.relu_sigma_1_prime(x)
def relu_squared(x):return backend.relu_squared(x)
def relu_squared_prime(x):return backend.relu_squared_prime(x)
def mask(x,dropout=1):return backend.mask(x,dropout)
def softmax_prime(x):return None
def softmax_old(x):return backend.softmax_old(x)
def sign(x):return backend.sign(x)
def abs(x):return backend.abs(x)
def bitwise_or(x,y):return backend.bitwise_or(x,y)
def argsort(x):return backend.argsort(x)
def ConvUp(images, filters, moduleStride, paddingStart):return backend.ConvUp(images, filters, moduleStride, paddingStart)
def ConvDown(hidActs, filters, moduleStride, paddingStart): return backend.ConvDown(hidActs, filters, moduleStride,paddingStart)
def ConvOut(images, hidActs, moduleStride, paddingStart): return backend.ConvOut(images, hidActs, moduleStride, paddingStart)
def MaxPool(images,subsX,startX,strideX,outputsX): return backend.MaxPool(images,subsX,startX,strideX,outputsX)
def MaxPoolUndo(images,grad,maxes,subsX,startX,strideX): return backend.MaxPoolUndo(images,grad,maxes,subsX,startX,strideX)
def AvgPool(images,subsX,startX,strideX,outputsX): return backend.AvgPool(images,subsX,startX,strideX,outputsX)
def AvgPoolUndo(images,grad,subsX,startX,strideX): return backend.AvgPoolUndo(images,grad,subsX,startX,strideX)

#def l2_normalize(x):return backend.l2_normalize(x)



def set_backend(name):
    global backend
    if name=="numpy": backend=NumpyBackend
    elif name=="gnumpy": backend=GnumpyBackend
    else: raise Exception("No Valid Backend")
    

def err_plot(err_list,a,b):
    if type(err_list)==gp.garray: err=err.as_numpy_array()
    plt.grid(True)
    plt.plot(np.arange(a,b),err_list[a:b])
#   plt.show()

def plot_filters(x,img_shape,tile_shape):
    if type(x)==gp.garray: print type(x);show_filters(x.as_numpy_array(),img_shape,tile_shape)
    elif type(x)==np.ndarray: print type(x);show_filters(x,img_shape,tile_shape)
    #plt.show()

def imshow(x):
    if type(x)==gp.garray: print type(x);plt.imshow(x.as_numpy_array(), cmap=plt.cm.gray, interpolation='nearest')
    elif type(x)==np.ndarray: print type(x);plt.imshow(x, cmap=plt.cm.gray, interpolation='nearest')
    #plt.show()

#from __main__ import feedforward,batch_size,want_KL,KL_flat,rho_target,want_exp,sigma,H

def show_images(imgs,tile_shape,scale=None,bar=False,unit=True,bg="black"):
    if type(imgs) == gp.garray: imgs=imgs.as_numpy_array()
    if imgs.ndim == 2:
        imgs=imgs.T.reshape(1,imgs.shape[1]**.5,imgs.shape[1]**.5,imgs.shape[0])        
    assert imgs.shape[3] == tile_shape[0]*tile_shape[1]
    img_shape = imgs.shape
    if bg=="white" :out=np.ones(((img_shape[1]+1)*tile_shape[0]+1,(img_shape[2]+1)*tile_shape[1]+1,img_shape[0]))
    if bg=="black" :out=np.zeros(((img_shape[1]+1)*tile_shape[0]+1,(img_shape[2]+1)*tile_shape[1]+1,img_shape[0]))
    for i in range(tile_shape[0]):
        for j in range(tile_shape[1]):
            k = tile_shape[1]*i+j
            if unit: x = scale_to_unit_interval(imgs[:,:,:,k])
            else: x = imgs[:,:,:,k]
            out[(img_shape[1]+1)*i+1:(img_shape[1]+1)*i+1+img_shape[1],(img_shape[2]+1)*j+1:(img_shape[2]+1)*j+1+img_shape[2],:] = np.rollaxis(x,0,3)
    if scale!=None: fig=plt.figure(num=None, figsize=(tile_shape[1]*scale, tile_shape[0]*scale), dpi=80, facecolor='w', edgecolor='k')
    if out.shape[2] == 1: 
        plt.imshow(out.squeeze(),cmap=plt.cm.gray, interpolation='nearest')
        return None
    plt.imshow(out,interpolation='nearest')
    if bar: plt.colorbar()

def scale_to_unit_interval(x):
    x = x.copy()
    x -= x.min()
    x *= 1.0 / (x.max() + 1e-8)
    return x


def make_activation(typename):
    if   typename == linear:   return linear_prime
    elif typename == sigmoid:  return sigmoid_prime
    elif typename == relu:     return relu_prime
    elif typename == softmax:  return softmax_prime
    elif typename == None:     return None

def find_batch_size(x):
    if x.ndim==4: return x.shape[3]
    elif x.ndim==2: return x.shape[0]

def mask_3d(x,k):
    x_ = np.swapaxes(x.as_numpy_array(),3,1).reshape(x.shape[0]*x.shape[3],-1)
    mask_ = threshold_mask_hard(x_,k,dropout = None)
    mask = np.swapaxes(mask_.reshape(x.shape[0],-1,x.shape[2],x.shape[1]),1,3)
    return gp.garray(mask)    