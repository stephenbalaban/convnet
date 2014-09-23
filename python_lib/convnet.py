from IPython.display import clear_output
import numpy as np
import gnumpy as gp
import nn_utils as nn
import pylab as plt
import time
import convnet_utils as cn

class NeuralNet:
    def __init__(self,cfg):
        self.cfg = cfg
        self.size = len(cfg)
        self.weights = cn.WeightSet(self.cfg)
        self.dweights = cn.WeightSet(self.cfg)
        # self.init_weights()
        self.test_mode = False

    def init_weights(self,initial_weights,silent_mode):
        if initial_weights == "layers": 
            for k in range(1,self.size):
                if not(k in self.cfg.index_pooling):
                    layer = self.cfg[k]
                    wk,bk = self.weights[k]
                    assert (self.cfg[k].initW != None and self.cfg[k].initB != None)
                    wk.ravel()[:] = self.cfg[k].initW * nn.randn(layer.num_weights-layer.num_filters)
                    if self.cfg[k].initB == "one":
                        bk[:] = nn.ones(layer.num_filters)
                    else:
                        bk[:] = self.cfg[k].initB * nn.randn(layer.num_filters)             
            if not silent_mode: print "Initialized Using Layers"        
        elif (type(initial_weights) == gp.garray or type(initial_weights) == np.ndarray): 
            self.weights.mem[:] = initial_weights
            if not silent_mode: print "Initialized Using initial_weights"
        elif initial_weights != None: raise Exception("Wrong Initialization!")
        else: print "Continue ..."

        if self.cfg.want_tied: 
            for hidden_pairs in self.cfg.tied_list:  self.weights.make_tied_copy(*hidden_pairs)

       

    def set_cost(self,cost):
        self.compute_cost = cost
        
    def feedforward(self,x):
        self.H = [None] * self.size
        self.H_ = [None] * self.size
        self.dH = [None] * self.size
        self.mask_matrix = [None] * self.size
        batch_size = nn.find_batch_size(x)
        self.H[0]=x
        if 0 in self.cfg.index_convolution: self.H_[0] = self.H[0].reshape(self.cfg[0].shape[0] * self.cfg[0].shape[1] * self.cfg[0].shape[2], batch_size).T
        else: self.H_[0] = self.H[0]
        
        for k in range(1,self.size):
            f=self.cfg[k].activation
            df=self.cfg[k].activation_prime

            if k in self.cfg.index_convolution:
                wk,bk = self.weights[k]
                A = self.cfg[k].applyConvUp(self.H[k-1], wk)              
                # A_ = A.reshape(self.cfg[k].shape[0], self.cfg[k].shape[1] * self.cfg[k].shape[2] * batch_size).T
                A+= bk.reshape(-1,1,1,1)
                # A  = A_.T.reshape(self.cfg[k].shape[0], self.cfg[k].shape[1], self.cfg[k].shape[2], batch_size)
                self.H[k]  = f(A)
                self.dH[k] = df(A)
                if (self.cfg.want_k_sparsity and (not self.test_mode) and self.cfg[k].k_sparsity != None):
                # if (self.cfg.want_k_sparsity and self.cfg[k].k_sparsity != None):
                    self.mask_matrix[k] = nn.mask_3d(self.H[k],self.cfg[k].k_sparsity) 
                    self.H[k] *= self.mask_matrix[k]
                    self.dH[k]*= self.mask_matrix[k]


                self.H_[k] = self.H[k].reshape(self.cfg[k].shape[0] * self.cfg[k].shape[1] * self.cfg[k].shape[2], batch_size).T
            
            elif k in self.cfg.index_pooling:
                self.H[k] = self.cfg[k].applyPooling(self.H[k-1])
                self.H_[k] = self.H[k].reshape(self.cfg[k].shape[0] * self.cfg[k].shape[1] * self.cfg[k].shape[2], batch_size).T
                
            elif k in self.cfg.index_dense:
                wk,bk = self.weights[k]
                A = nn.dot(self.H_[k-1],wk)
                A += bk
                self.H[k]  = f(A)
                self.dH[k] = df(A)
                if (self.cfg.want_k_sparsity and self.cfg[k].k_sparsity != None):
                    self.mask_matrix[k] = nn.threshold_mask_hard(self.H[k],self.cfg[k].k_sparsity) 
                    self.H[k] *= self.mask_matrix[k]
                    self.dH[k]*= self.mask_matrix[k]

                if (self.cfg.want_dropout and self.cfg[k].dropout != None): 
                    if self.test_mode: self.H[k]*=self.cfg[k].dropout
                    else:
                        self.mask_matrix[k] = nn.mask(self.H[k],self.cfg[k].dropout)
                        self.H[k] *= self.mask_matrix[k]
                        self.dH[k]*= self.mask_matrix[k]
                self.H_[k] = self.H[k]

                
    def compute_cost_log(self,x,t):
        batch_size = nn.find_batch_size(x)
        # x = I[:,:,:,batch_size*l:batch_size*(l+1)]
        # O[batch_size*l:batch_size*(l+1)]
        self.feedforward(x)
        out = (1.0/batch_size)*(-t*nn.log(self.H[-1])).sum()
        for k in range(1,len(self.cfg)):
            if self.cfg[k].l2!=None:
                wk,bk = self.weights[k]
                out  += self.cfg[k].l2*.5*((wk**2).sum())   
        return out

    def compute_cost_euclidean(self,x,t):
        batch_size = nn.find_batch_size(x)
        # x = I[:,:,:,batch_size*l:batch_size*(l+1)]
        self.feedforward(x)
        return nn.sum(((1.0/batch_size)*(.5*(t-self.H[-1])**2)),axis=None)

    
    def compute_grad(self,x,t):
        batch_size = nn.find_batch_size(x)
        # x = I[:,:,:,batch_size*l:batch_size*(l+1)]
        self.feedforward(x)

        wk,bk = self.weights[self.size-1]
        dwk,dbk = self.dweights[self.size-1]

        if ((self.size-1) in self.cfg.index_dense): delta_ = self.H[-1]-t
        else:                                       delta = self.H[-1]-t

        for k in range(1,self.size)[::-1]:
                    
            if k in self.cfg.index_dense:
                wk,bk = self.weights[k]
                dwk,dbk = self.dweights[k]
                dwk[:] = 1.0/batch_size*nn.dot(self.H_[k-1].T,delta_)
                dbk[:] = 1.0/batch_size*nn.sum(delta_,axis=0)
                
                delta_ = nn.dot(delta_, wk.T)
                if k-1 in self.cfg.index_pooling: 
                    delta  = delta_.T.reshape(self.cfg[k-1].shape[0], self.cfg[k-1].shape[1], self.cfg[k-1].shape[2], batch_size)
                elif (k-1 in self.cfg.index_convolution and k!=1):
                    delta  = delta_.T.reshape(self.cfg[k-1].shape[0], self.cfg[k-1].shape[1], self.cfg[k-1].shape[2], batch_size)
                    delta *= self.dH[k-1]
                elif (k-1 in self.cfg.index_dense and k!=1): delta_ *= self.dH[k-1]
                               
            elif k in self.cfg.index_pooling:
                delta  = self.cfg[k].applyPoolingUndo(self.H[k-1],delta,self.H[k])
                delta *= self.dH[k-1]
                        
            elif k in self.cfg.index_convolution:
                wk,bk = self.weights[k]
                dwk,dbk = self.dweights[k]    
                delta_ = delta.reshape(self.cfg[k].shape[0], self.cfg[k].shape[1] * self.cfg[k].shape[2] * batch_size).T
                dwk[:] = (1.0/batch_size)*self.cfg[k].applyConvOut(self.H[k-1], delta)
                dbk[:] = (1.0/batch_size)*nn.sum(delta_,axis=0)
                
                if k!=1: delta = self.cfg[k].applyConvDown(delta, wk) #convdown is unnecessary if k==1
                if (k-1 in self.cfg.index_convolution and k!=1): delta *= self.dH[k-1]
        
        #tied weights
        if self.cfg.want_tied:
            for hidden_pairs in self.cfg.tied_list: self.dweights.make_tied(*hidden_pairs)  
                
        for k in range(1,len(self.cfg)):
            if self.cfg[k].l2!=None:
                wk,bk   = self.weights[k]
                dwk,dbk = self.dweights[k]
                dwk    += self.cfg[k].l2*wk                


        
##########################################################################################################

    def train(self,X,T,X_test,T_labels,
              momentum,learning_rate,batch_size,dataset_size,
              initial_weights = None,visual=False,report=True,
              num_epochs=10000,
              hyper=False,u=None,s=None,silent_mode = False):
        
        print "Type of X: ",type(X)
        print "Backend: ",nn.backend
        
        self.init_weights(initial_weights,silent_mode)
   
        num_batch = int(dataset_size/batch_size)
        v = cn.WeightSet(self.cfg)        
        tic = time.time()
        
        self.err_train = np.zeros((num_epochs),'float32')
        self.err_test = np.zeros((num_epochs),'int32')
        
        for epoch in range(1,num_epochs+1):       
            self.epoch = epoch

            x = self.data_provider(X,0,batch_size)
            t = self.data_provider(T,0,batch_size)

            self.err_train[epoch-1] = self.compute_cost(x,t) #dataset size or batch size greater than the actual still works!!!!!!

            if report: self.err_test[epoch-1] = self.test(X_test,T_labels)                        
            if (epoch % 1 == 0 and not silent_mode): print epoch,self.err_train[epoch-1],self.err_test[epoch-1],round(time.time()-tic,2)
            tic = time.time()

            if visual and not silent_mode: self.visualize(X,u,s)

            plt.sys.stdout.flush()            
            for l in range(num_batch):

                x = self.data_provider(X,batch_size*l,batch_size*(l+1))
                t = self.data_provider(T,batch_size*l,batch_size*(l+1))

                self.compute_grad(x,t)
                self.dweights*=-learning_rate
                v*=momentum
                v+=self.dweights #v = momentum*v - learning_rate*self.dweights
                self.weights+=v #v=m*v-l*dw; w=w+v
            if not silent_mode: clear_output()

        if silent_mode: return self.err_test

    def data_provider(self,X,a,b):
        if type(X) == gp.garray: 
            assert nn.backend == nn.GnumpyBackend
            if X.ndim == 4: return X[:,:,:,a:b]
            else : return X[a:b]
        elif type(X) == np.ndarray: 
            if nn.backend == nn.GnumpyBackend: 
                if X.ndim == 4: return gp.garray(X[:,:,:,a:b])
                else : return gp.garray(X[a:b])
            elif nn.backend == nn.NumpyBackend:
                if X.ndim == 4: return X[:,:,:,a:b]
                else : return X[a:b]       

    
    def test(self,X_test,T_labels):
        batch_test_size = 100
        if X_test.ndim ==4: test_size = X_test.shape[3]
        else: test_size = X_test.shape[0]
        assert test_size % batch_test_size == 0
        # assert nn.backend == nn.GnumpyBackend
        num_batch = int(test_size/batch_test_size)
        num_errors = 0
        for l in range(num_batch):
            x = self.data_provider(X_test,batch_test_size*l,batch_test_size*(l+1))
            t = self.data_provider(T_labels,batch_test_size*l,batch_test_size*(l+1))         
            # x = X_test[:,:,:,batch_test_size*l:batch_test_size*(l+1)]
            # t = T_labels[batch_test_size*l:batch_test_size*(l+1)]
            self.test_mode = True
            self.feedforward(x)
            self.test_mode = False
            if nn.backend==nn.GnumpyBackend: num_errors += (np.argmax(self.H[-1].as_numpy_array(),axis=1) != t.as_numpy_array()).sum()
            if nn.backend==nn.NumpyBackend: num_errors += np.array((np.argmax(self.H[-1],axis=1) != t)).sum()
        return int(num_errors)
        

   
    def plot_train(self,a=0,b=None):
        if b==None: b=self.epoch
        plt.grid(True)
        plt.plot(np.arange(a,b),self.err_train[a:b])
        
    def plot_test(self,a=0,b=None):
        if b==None: b=self.epoch
        plt.grid(True)
        plt.plot(np.arange(a,b),self.err_test[a:b])

    
    def visualize(self,X,u=None,s=None):
        if self.H[-1].ndim ==4: 
            self.visualize_conv(X)
            return
        x = self.data_provider(X,0,1)
        w1,b1=self.weights[1]
        plt.figure(num=None, figsize=(15,5), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot(131); self.plot_train(a=2)
        plt.subplot(132); self.plot_test(a=2)
        plt.subplot(133)        
        if 1 in self.cfg.index_dense: #dense 
            if self.cfg[0].shape==[1, 28, 28] or self.cfg[0].shape==784: 
                if w1.shape[1]>25: nn.show_images(w1[:,:25].T,(5,5)) #MNIST dense
                else: nn.show_images(w1[:,:].T,(5,w1.shape[1]/5))  #MNIST softmax          
            elif self.cfg[0].shape==[3, 32, 32]: 
                if w1.shape[1]>25: nn.show_images(w1[:,:25].reshape(3,32,32,25),(5,5)) #CIFAR10 dense
                else: nn.show_images(w1[:,:].reshape(3,32,32,10),(5,2))  #CIFAR10 softmax
            elif self.cfg[0].shape==[3, 8, 8]: #CIFAR10 dense patches
                if u==None: nn.show_images(w1[:,:25].reshape(3,8,8,25),(5,5)) 
                else: nn.show_images(whiten_undo(w1[:,:25].T.as_numpy_array(),u,s).T.reshape(3,8,8,25),(5,5),unit=True)
            elif self.cfg[0].shape==[1, 8, 8]: #MNIST dense patches
                nn.show_images(w1[:,:25].T.as_numpy_array().T.reshape(1,8,8,16),(4,4),unit=True)
        else: 
            if self.cfg[0].shape[0]<4: nn.show_images(w1[:,:,:,:16],(4,4)) #convnet
        plt.show()
        if self.cfg[-1].activation==nn.linear: #autoencoder
            if self.cfg[0].shape==784: #MNIST dense
                plt.subplot(121); nn.show_images(self.H[0][:1,:],(1,1)) 
                plt.subplot(122); nn.show_images(self.H[-1][:1,:],(1,1));        
            if self.cfg[0].shape==[1, 28, 28]: #MNIST dense
                plt.subplot(121); nn.show_images(self.H[0][:,:,:,0].reshape(1, 28, 28,1)) 
                plt.subplot(122); nn.show_images(self.H[-1][0].T.reshape(1,28,28,1),(1,1));
            elif self.cfg[0].shape==[3, 32, 32]: #CIFAR10
                plt.subplot(121); nn.show_images(self.H[0][:,:,:,0].reshape(3, 32, 32,1),(1,1)) 
                plt.subplot(122); nn.show_images(self.H[-1][0].T.reshape(3,32,32,1),(1,1))
            elif self.cfg[0].shape==[1, 8, 8]: #MNIST patches
                    plt.subplot(121); nn.show_images(self.H[0][:,:,:,8:9],(1,1)) 
                    plt.subplot(122); nn.show_images(self.H[-1][8].T.reshape(1,8,8,1),(1,1))
            elif self.cfg[0].shape==[3, 8, 8]: #CIFAR10 patches
                if u==None: 
                    plt.subplot(121); nn.show_images(self.H[0][:,:,:,0].reshape(3, 8, 8,1),(1,1)) 
                    plt.subplot(122); nn.show_images(self.H[-1][0].T.reshape(3,8,8,1),(1,1))
                else:
                    plt.subplot(121); nn.show_images(whiten_undo(X.reshape(192,1000000).T[0].as_numpy_array(),u,s).T.reshape(3,8,8,1),(1,1),unit=True)
                    self.feedforward(x.reshape(3,8,8,1))
                    plt.subplot(122); nn.show_images(whiten_undo(self.H[-1][0].as_numpy_array(),u,s).T.reshape(3,8,8,1),(1,1),unit=True)
        plt.show()
        if not(1 in self.cfg.index_dense):
            self.feedforward(x)        
            plt.figure(num=None, figsize=(15,5), dpi=80, facecolor='w', edgecolor='k')
            plt.subplot(121); nn.show_images(np.swapaxes(self.H[1][:16,:,:,:1].as_numpy_array(),0,3),(4,4),bg="white")
            plt.subplot(122); nn.show_images(np.swapaxes(self.H[2][:16,:,:,:1].as_numpy_array(),0,3),(4,4),bg="white")
            plt.show()

    def visualize_conv(self,X):
        w1,b1 = self.weights[1]
        w2,b2 = self.weights[2]
        x=X[:,:,:,:1]
        self.feedforward(x)
        plt.figure(num=None, figsize=(15,5), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot(131)
        nn.show_images(self.H[2][:,:,:,:1],(1,1))
        plt.subplot(132)
        nn.show_images(w1[:,:,:,:16],(4,4))    
        plt.subplot(133)            
        nn.show_images(np.swapaxes(w2.as_numpy_array(),0,3)[:,:,:,:16],(4,4),unit=True)
        plt.show()
        plt.figure(num=None, figsize=(15,5), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot(131)            
        self.plot_train(a=2)            
        plt.subplot(132)
        nn.show_images(np.swapaxes(self.H[1][:16,:,:,:1].as_numpy_array(),0,3),(4,4),bg="white")
        plt.subplot(133)
        # if self.dataset == "cifar": nn.show_images(self.H1[:,:,:,:1].sum(0).reshape(1,32,32,1),(1,1))
        # elif self.dataset == "mnist": nn.show_images(self.H1[:,:,:,:1].sum(0).reshape(1,28,28,1),(1,1))
        plt.show() 

    def load(self,name):
        if nn.backend==nn.GnumpyBackend: print "gnumpy"; self.weights.mem=gp.garray(np.loadtxt("./out/"+name, delimiter=','))
        else:                            print "numpy" ; self.weights.mem=np.loadtxt("./out/"+name, delimiter=',')

    def save(self,name):       
        if nn.backend==nn.GnumpyBackend: print "gnumpy"; np.savetxt("./out/"+name, self.weights.mem.as_numpy_array(), delimiter=',')
        else:                            print "numpy" ; np.savetxt("./out/"+name, self.weights.mem, delimiter=',')

    def show_filters(self,*size):
        w1,b1=self.weights[1]
        w2,b2=self.weights[2]
        if self.size - 1 in self.cfg.index_convolution:
            # plt.figure(num=None, figsize=(30,90), dpi=80, facecolor='w', edgecolor='k')
            nn.show_images(np.swapaxes(w2.as_numpy_array(),0,3)[:,:,:,:size[0]*size[1]],(size[0],size[1]),unit=True)
            return
        if 1 in self.cfg.index_dense:
            plt.figure(num=None, figsize=(30,90), dpi=80, facecolor='w', edgecolor='k')
            if type(self.cfg[0].shape)==int: nn.show_images(w1[:,:size[0]*size[1]].T,size) #MNIST dense
            elif self.cfg[0].shape[0]==1: nn.show_images(w1[:,:size[0]*size[1]].T,size) #MNIST dense
            elif self.cfg[0].shape[0]==3: nn.show_images(w1[:,:size[0]*size[1]].reshape(3,32,32,size[0]*size[1]),size[::-1]) #CIFAR10 dense
             
            
    def gradient_check(self):

        # assert self.cfg[-1].activation==nn.softmax
        assert nn.backend == nn.NumpyBackend

        if 0 in self.cfg.index_convolution: x=nn.randn((self.cfg[0].shape[0],self.cfg[0].shape[1],self.cfg[0].shape[2],2))
        else: x=nn.randn((2,self.cfg[0].shape))

        if self.size-1 in self.cfg.index_dense: 
            t=nn.randn((2,self.cfg[-1].shape))
            row_sums = t.sum(axis=1);t = t / row_sums[:, np.newaxis] #for softmax gradient checking, rows should sum up to one.
        else: t=nn.randn((self.cfg[-1].shape[0],self.cfg[-1].shape[1],self.cfg[-1].shape[2],2))

        self.weights.mem[:]=.01*nn.randn(self.cfg.num_parameters)
        self.compute_grad(x,t)
        epsilon=.00001
        for k in self.cfg.index_dense:
            if k==0: continue
            wk,bk=self.weights[k]
            dwk,dbk=self.dweights[k]
            f=self.compute_cost(x,t)
            wk[0,0]+=epsilon
            f_new=self.compute_cost(x,t)
            df=(f_new-f)
            print k,df/epsilon/dwk[0,0]
            f=self.compute_cost(x,t)
            bk[0]+=epsilon
            f_new=self.compute_cost(x,t)
            df=(f_new-f)
            print k,df/epsilon/dbk[0]
        for k in self.cfg.index_convolution:
            if k==0: continue
            wk,bk=self.weights[k]
            dwk,dbk=self.dweights[k]
            f=self.compute_cost(x,t)
            wk[0,1,2,0]+=epsilon
            f_new=self.compute_cost(x,t)
            df=(f_new-f)
            print k,df/epsilon/dwk[0,1,2,0]
            f=self.compute_cost(x,t)
            bk[0]+=epsilon
            f_new=self.compute_cost(x,t)
            df=(f_new-f)
            print k,df/epsilon/dbk[0]

    def gradient_check_numpy_ae(self,X,T):
        self.weights.mem[:]=.01*nn.randn(self.cfg.num_parameters)
        self.compute_grad(X,T,0,2)
        epsilon=.00001
        for k in self.cfg.index_dense:
            wk,bk=self.weights[k]
            dwk,dbk=self.dweights[k]
            f=self.compute_cost(x,t)
            wk[0,0]+=epsilon
            f_new=self.compute_cost(x,t)
            df=(f_new-f)
            print k,df/epsilon/dwk[0,0]
            f=self.compute_cost(x,t)
            bk[0]+=epsilon
            f_new=self.compute_cost(x,t)
            df=(f_new-f)
            print k,df/epsilon/dbk[0]
        for k in self.cfg.index_convolution:
            if k==0: continue
            wk,bk=self.weights[k]
            dwk,dbk=self.dweights[k]
            f=self.compute_cost(x,t)
            wk[0,1,2,3]+=epsilon
            f_new=self.compute_cost(x,t)
            df=(f_new-f)
            print k,df/epsilon/dwk[0,1,2,3]
            f=self.compute_cost(x,t)
            bk[0]+=epsilon
            f_new=self.compute_cost(x,t)
            df=(f_new-f)
            print k,df/epsilon/dbk[0]

def whiten_undo(x,u,s):
    return np.dot(np.dot(np.dot(x,u),np.diag(np.sqrt(1.0*s))),u.T)