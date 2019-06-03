#coding=utf-8
import copy, numpy as np                                       
np.random.seed(0)

# compute sigmoid nonlinearity                         #定义sigmoid函数
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative   #计算sigmoid函数的倒数
def sigmoid_output_to_derivative(output):
    return output*(1-output)

# training dataset generation                            
int2binary = {}                                    #用于将输入的整数转为计算机可运行的二进制数用
binary_dim = 8                                     #定义了二进制数的长度=8

largest_number = pow(2,binary_dim)                         #二进制数最大能取的数就=256喽
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)            
for i in range(largest_number):                                #将二进制数与十进制数做个一一对应关系
    int2binary[i] = binary[i]

# input variables
alpha = 0.1              #反向传播时参数w更新的速度
input_dim = 2            #输入数据的维度，程序是实现两个数相加的
hidden_dim = 16          #隐藏层神经元个数=16
output_dim = 1           #输出结果值是1维的


# initialize neural network weights                          #初始化神经网络的权重参数
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1   #输入至神经元的w0，维度为2X16，取值约束在[-1,1]间
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1  #神经元至输出层的权重w1,维度为16X1，取值约束在[-1,1]间
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1  #神经元前一时刻状态至当前状态权重wh,维度为16X16，取值约束在[-1,1]间                                   

synapse_0_update = np.zeros_like(synapse_0)          #构造与w0相同维度的矩阵，并初始化为全0；
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
for j in range(10000):          #模型迭代次数，可自行更改
    
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version      #约束初始化的输入加数a的值不超过128
    a = int2binary[a_int] # binary encoding                        #将加数a的转为对应二进制数
    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    c_int = a_int + b_int    #真实和
    c = int2binary[c_int]    
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)                                        #用于存储预测的和

    overallError = 0                                           #打印显示误差
    
    layer_2_deltas = list()                                  #反向求导用
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))             #先对隐藏层前一时刻状态初始化为0
    
    # moving along the positions in the binary encoding
    for position in range(binary_dim):                         #前向传播；二进制求和，低位在右，高位在左
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]]) #输入的a与b（二进制形式）
        y = np.array([[c[binary_dim - position - 1]]]).T                            #真实label值

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))  # X*w0+RNN前一时刻状态值*wh

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))          #layer_1*w1

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2                                            #求误差
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))  #代价函数
        #print layer_1.shape,X.shape
        overallError += np.abs(layer_2_error[0])                             #误差，打印显示用
    
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])                    #预测的和
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))                     #深拷贝，将RNN模块状态值存储，用于反向传播
    
    future_layer_1_delta = np.zeros(hidden_dim)
    
    for position in range(binary_dim):   #反向传播，计算从左到右，即二进制高位到低位
        
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) +\
            layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
        #print future_layer_1_delta.shape,layer_2_delta.shape,synapse_1.shape,layer_1.shape
        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)       #对w1进行更新
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)  #对wh进行更新
        synapse_0_update += X.T.dot(layer_1_delta)                            #对w0进行更新
        
        future_layer_1_delta = layer_1_delta
    

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):  #每1000次打印结果
        print ("Error:" + str(overallError))
        print ("Pred:" + str(d))
        print ("True:" + str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print (str(a_int) + " + " + str(b_int) + " = " + str(out))
        print ("------------")
