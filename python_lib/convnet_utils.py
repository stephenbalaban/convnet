import nn_utils as nn

class NeuralNetLayerCfg(object):
    def __init__(self, dropout, k_sparsity, l2):
        self.type = None
        self.activation = None
        self.activation_prime = None
        self.num_filters = None
        self.filter_shape = None
        self.filter_size = None  
        self.num_weights = None
        self.shape = None #to be determined
        self.size = None #to be determined
        self.initW = None
        self.initB = None
        self.padding = None
        self.stride = None        
        self.k_sparsity = k_sparsity
        self.l2 = l2         
        self.dropout = dropout  

class NeuralNetLayerPoolingCfg(NeuralNetLayerCfg):
    def __init__(self, mode, pooling_width, stride, padding, dropout, l2):
        NeuralNetLayerCfg.__init__(self,dropout, None, l2)
        self.type="pooling"
        self.mode = mode
        assert mode == "avg" or mode == "max"
        self.pooling_width = pooling_width
        self.stride = stride
        self.padding = padding
        self.num_weights = 0

    def applyPooling(self, H):
        if self.mode == "max": return nn.MaxPool(H, self.pooling_width, self.padding, self.stride, self.shape[1])
        elif self.mode == "avg": return nn.AvgPool(H, self.pooling_width, self.padding, self.stride, self.shape[1])
    def applyPoolingUndo(self, H, delta, H_next):
        if self.mode == "max": return nn.MaxPoolUndo(H, delta, H_next, self.pooling_width, self.padding, self.stride)
        elif self.mode == "avg": return nn.AvgPoolUndo(H, delta, self.pooling_width, self.padding, self.stride)


class NeuralNetLayerDenseCfg(NeuralNetLayerCfg):
    def __init__(self,num_filters,activation, initW, initB, dropout, k_sparsity, l2):
        NeuralNetLayerCfg.__init__(self,dropout, k_sparsity, l2)
        self.type = "dense"
        self.num_filters = num_filters
        self.activation = activation
        self.activation_prime = nn.make_activation(activation)        
        self.size = num_filters 
        self.shape = num_filters
        self.initW = initW
        self.initB = initB

class NeuralNetLayerConvolutionCfg(NeuralNetLayerCfg):
    def __init__(self, num_filters, activation, filter_width, stride, padding, initW, initB, dropout, l2, k_sparsity):
        NeuralNetLayerCfg.__init__(self,dropout, k_sparsity, l2)
        self.type = "convolution"
        self.num_filters = num_filters
        self.activation = activation
        self.activation_prime = nn.make_activation(activation)        
        self.filter_width = filter_width
        self.stride = stride
        self.padding = padding
        self.initW = initW
        self.initB = initB

    def applyConvUp(self, H, w):
        return nn.ConvUp(H, w, moduleStride = self.stride, paddingStart = -self.padding)
    def applyConvDown(self, delta, w):
        return nn.ConvDown(delta, w, moduleStride = self.stride, paddingStart = -self.padding)
    def applyConvOut(self, H, delta):
        return nn.ConvOut(H, delta, moduleStride = self.stride, paddingStart = -self.padding)     

#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#double check l2_norm

class NeuralNetCfg(object):
    def __init__(self, want_dropout=False,want_k_sparsity = False,
                 want_tied=False, tied_list=[(1,2)]):
        self._layers = [None,None]
        self.index_layer = 0
        self.index_convolution = []
        self.index_pooling = []
        self.index_dense = []
        self.want_dropout = want_dropout
        self.want_tied = want_tied
        self.tied_list = tied_list
        self.want_k_sparsity = want_k_sparsity
            
    def input_conv(self,shape, dropout=None):
        layer = NeuralNetLayerConvolutionCfg(num_filters=None, activation=None, filter_width=None, stride=None, padding=None, initW=None, initB=None, dropout=dropout, l2=None, k_sparsity = None)
        layer.shape = shape
        layer.size = shape[0]*shape[1]*shape[2]
        self._layers[0] = layer
        self.index_convolution.append(self.index_layer); self.index_layer+=1

    def input_dense(self,shape, dropout=None):
        layer = NeuralNetLayerDenseCfg(num_filters=None, activation=None, initW=None, initB=None, dropout=dropout, k_sparsity=None, l2=None)
        layer.shape = shape
        layer.size = shape
        self._layers[0] = layer
        self.index_dense.append(self.index_layer); self.index_layer+=1        
    
    def convolution(self, num_filters, activation, filter_width=None, stride=1, padding=0, initW = None, initB = None, dropout = None, k_sparsity=None, l2=None):
        layer = NeuralNetLayerConvolutionCfg(num_filters=num_filters, activation=activation, filter_width=filter_width, stride=stride, padding=padding, initW=initW, initB=initB, dropout=dropout, k_sparsity=k_sparsity, l2=l2)      
        self._layers.insert(-1,layer)
        self.index_convolution.append(self.index_layer); self.index_layer+=1
   
    def pooling(self, mode, pooling_width, stride, padding = 0, dropout = None, l2=None):
        assert padding == 0 #implement padding for pooling layer
        layer = NeuralNetLayerPoolingCfg(mode=mode, pooling_width=pooling_width, stride=stride, padding=padding, dropout=dropout, l2=l2)    
        self._layers.insert(-1,layer)
        self.index_pooling.append(self.index_layer); self.index_layer+=1
        
    def dense(self, num_filters, activation, initW = None, initB = None, dropout = None, k_sparsity = None, l2 = None):
        layer = NeuralNetLayerDenseCfg(num_filters=num_filters, activation=activation, initW=initW, initB=initB, dropout=dropout, k_sparsity=k_sparsity, l2=l2)
        self._layers.insert(-1,layer)
        self.index_dense.append(self.index_layer); self.index_layer+=1
        
    def output_dense(self, num_filters, activation, initW = None, initB = None, dropout = None, l2 = None):
        layer = NeuralNetLayerDenseCfg(num_filters=num_filters, activation=activation, initW=initW, initB=initB, dropout=dropout, k_sparsity = None, l2=l2)
        self._layers[-1] = layer
        self.index_dense.append(self.index_layer); self.index_layer+=1
        self.finalize()

    def output_conv(self, num_filters, activation, filter_width=None, stride=1, padding=0, initW = None, initB = None, l2=None):
        layer = NeuralNetLayerConvolutionCfg(num_filters=num_filters, activation=activation, filter_width=filter_width, stride=stride, padding=padding, initW=initW, initB=initB, dropout=None, k_sparsity=None, l2=l2)
        self._layers[-1] = layer
        self.index_convolution.append(self.index_layer); self.index_layer+=1
        self.finalize()        

    def finalize(self):
        for k in range(1,len(self)):
            layer_previous = self[k-1]
            layer = self[k]
            
            if k in self.index_convolution:
                try:
                    assert (2*abs(layer.padding) + layer_previous.shape[1] - layer.filter_width) % layer.stride == 0
                    assert (2*abs(layer.padding) + layer_previous.shape[2] - layer.filter_width) % layer.stride == 0
                except AssertionError:
                    print("\x1b[31m\"Error in layer: \"\x1b[0m"),k
                    print [(self[i].shape) for i in range(k)]
                layer.shape = [layer.num_filters,(layer_previous.shape[1]+2*layer.padding-layer.filter_width)/layer.stride+1, (layer_previous.shape[2]+2*layer.padding-layer.filter_width)/layer.stride+1]
                layer.size = layer.shape[1] * layer.shape[2] * layer.num_filters                
                layer.filter_shape = [layer_previous.shape[0],layer.filter_width,layer.filter_width,layer.shape[0]]
                layer.filter_size = layer.filter_width** 2 * layer_previous.shape[0] *layer.shape[0]
                layer.num_weights = layer.filter_size + layer.num_filters
       
            if k in self.index_dense:
                layer.filter_shape = layer_previous.size
                layer.filter_size = layer_previous.size
                layer.num_weights = (layer_previous.size+1)*layer.num_filters

            if k in self.index_pooling:
                try:
                    assert layer_previous.shape[1] % layer.stride == 0
                    assert layer_previous.shape[2] % layer.stride == 0
                except AssertionError:
                    print("\x1b[31m\"Error in layer: \"\x1b[0m"),k
                    print [(self[i].shape) for i in range(k)]
                layer.shape = [layer_previous.num_filters,layer_previous.shape[1]/layer.stride, layer_previous.shape[2]/layer.stride]
                layer.size = layer.shape[1] * layer.shape[2] * layer_previous.num_filters

                
        self.k_sparsity=[(self[k].k_sparsity) for k in range(len(self))]
        self.layer_shape=[(self[k].shape) for k in range(len(self))]
        self.padding=[(self[k].padding) for k in range(len(self))]
        self.stride=[(self[k].stride) for k in range(len(self))]
        self.layer_size= [self[k].size for k in range(len(self))]
        self.activation=[(self[k].activation) for k in range(len(self))]
        self.activation_prime=[(self[k].activation_prime) for k in range(len(self))]
        self.filter_shape=[(self[k].filter_shape) for k in range(1,len(self))]
        self.filter_size=[(self[k].filter_size) for k in range(1,len(self))]
        self.num_weights=[(self[k].num_weights) for k in range(1,len(self))]
        self.num_weights_sum = [sum(self.num_weights[:i+1]) for i in range(len(self)-1)]
        self.num_parameters=sum(self.num_weights)

    def info(self):
        print 'k_sparsity:', self.k_sparsity
        print 'index convolution', self.index_convolution
        print 'index dense', self.index_dense
        print 'index pooling', self.index_pooling
        print 'activations', self.activation
        print 'layer shape: ',self.layer_shape
        print 'layer size: ',self.layer_size
        print 'filter shape: ',self.filter_shape
        print 'filter size: ',self.filter_size
        print 'number of weights: ',self.num_weights
        print 'number of weights sum: ',self.num_weights_sum
        print 'padding: ',self.padding
        print 'stride: ',self.stride

    def __getitem__(self,i): return self._layers[i]
    def __len__(self):       return len(self._layers)
    def __iter__(self):      return self._layers.__iter__()

#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/

class WeightSet:
    def __init__(self,cfg,initial_weights=None):
        self.cfg=cfg 
        self.size=len(cfg.num_weights)           
        if initial_weights == None: self.mem = nn.zeros(self.cfg.num_parameters) #it has to be initialized because of wk,bk = self.weights[k] in the self.init_weights(initial_weights)
        # if initial_weights == None: pass
        else: self.mem = initial_weights
  
    def __iadd__(self, other):
        if isinstance(other, WeightSet):self.mem += other.mem;return self
        else:self.mem += other;return self
        
    def __isub__(self, other):
        if isinstance(other, WeightSet):self.mem -= other.mem;return self
        else:self.mem -= other;return self
    
    def __imul__(self, other):
        if isinstance(other, WeightSet):self.mem *= other.mem;return self
        else:self.mem *= other;return self
    
    def __add__(self, other):
        if isinstance(other, WeightSet):return WeightSet(self.cfg,self.mem + other.mem)
        else:return WeightSet(self.cfg,self.mem + other)
    
    __radd__=__add__
    
    def __sub__(self, other):
        if isinstance(other, WeightSet):return WeightSet(self.cfg,self.mem - other.mem)
        else:return WeightSet(self.cfg,self.mem - other)
    
    __rsub__=__sub__
    
    def __mul__(self, other):
        if isinstance(other, WeightSet):return WeightSet(self.cfg,self.mem * other.mem)
        else: return WeightSet(self.cfg,self.mem * other)
    
    __rmul__=__mul__

    def full(self,i): 
        pass #to be implemented
    
    def make_tied(self,i,j):
        wk0,bk0 = self[i]
        wk1,bk1 = self[j]
        wk0    += wk1.T
        wk1[:]  = wk0.T
        
    def make_tied_copy(self,i,j):
        wk0,bk0 = self[i]
        wk1,bk1 = self[j]
        wk1[:]  = wk0.T 

    def __getitem__(self,i):
        if i in self.cfg.index_pooling: return None, None
        w=self.mem[(self.cfg.num_weights_sum[i-1]-self.cfg.num_weights[i-1]):self.cfg.num_weights_sum[i-1]-self.cfg[i].num_filters]
        b=self.mem[self.cfg.num_weights_sum[i-1]-self.cfg[i].num_filters:self.cfg.num_weights_sum[i-1]]
        if i in self.cfg.index_convolution: return w.reshape(tuple(self.cfg[i].filter_shape)),b
        elif i in self.cfg.index_dense: return w.reshape(self.cfg[i-1].size,self.cfg[i].size),b
        else: raise Exception("Wrong index!")
