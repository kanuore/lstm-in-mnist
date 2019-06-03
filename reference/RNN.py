#coding=utf-8
# RNN
import numpy as np
np.random.seed(0)
import copy
def sigmoid(x):
	output = 1/(1+np.exp(-x))
	return output

def sigmoid_output_to_derivative(output):
	return output*(1-output)

#training dataset generation
int2binary_number = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
	int2binary_number[i] = binary[i]


alpha = 0.1
input_dim = 2 
hidden_dim = 16
output_dim = 1

#初始化参数
params_1 = 2*np.random.random((input_dim,hidden_dim)) - 1
params_2 = 2*np.random.random((hidden_dim,output_dim)) - 1
params_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

#updata
params_1_update = np.zeros_like(params_1)
params_2_update = np.zeros_like(params_2)
params_h_update = np.zeros_like(params_h)

for j in range(10000):
	a_int = np.random.randint(largest_number/2)
	a = int2binary_number[a_int]

	b_int = np.random.randint(largest_number/2)
	b = int2binary_number[b_int]

	c_int = a_int + b_int
	c = int2binary_number[c_int]

	pre = np.zeros_like(c)

	overallerror = 0

	#每一层的残差值
	layer_1_values = []
	layer_2_values = []
	loss = []
	#使出开始的循环层有个 -1 状态
	layer_1_values.append(np.zeros(hidden_dim))
	
	# forward
	for position in range(binary_dim):
		X = np.array([[a[binary_dim - position -1],b[binary_dim - position -1]]])
		Y = np.array([[c[binary_dim - position -1]]]).T

		#RNN层
		layer_1 = sigmoid(np.dot(X,params_1) + np.dot(layer_1_values[-1],params_h))

		#输出层
		layer_2 = sigmoid(np.dot(layer_1,params_2))
		#预测值
		pre[binary_dim - position -1] = np.round(layer_2[0][0])
		

		#前向损失计算
		#error = Y - layer_2
		error = layer_2 - Y
		#一个数8位的loss,但是对于RNN 来用就是一位一位的使用
		overallerror += np.abs(error[0])

		#保存每一位的损失，相当于状态保存
		loss.append(error)
		layer_2_values.append(copy.deepcopy(layer_2))
		layer_1_values.append(copy.deepcopy(layer_1))
		#print layer_2_error.shape,layer_2.shape
		#layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))
	
	#保存一下来到循环层的残差
	future_layer_1_delta = np.zeros(hidden_dim)
	#backpoint
	for position in range(binary_dim):
		X = np.array([[a[position],b[position]]])

		layer_1 = layer_1_values[-position-1]
		layer_2 = layer_2_values[-position-1]
		prev_layer_1 = layer_1_values[-position-2]
		#由于经过激活函数，这一步相当于损失通过激活函数，“直面”最后一层
		layer_2_delta = loss[-position-1] * sigmoid_output_to_derivative(layer_2)

		#  params_2是连接从layer_1到layer_2的“路线”上的权重，所以2层回传的损失还要乘以权重回来
		# 由于循环层的输出还是自己层用，所以循环层的“残差”来源还是自己层，只不过是上一个状态,
		# ！！！注意这里的残差使用方式
		layer_1_delta=(future_layer_1_delta.dot(params_h.T)+ \
			layer_2_delta.dot(params_2.T)) * sigmoid_output_to_derivative(layer_1)

		# 根据公式 输入值 * 直面到层的残差 就是损失对w的导数
		# np.atleast_2d转换乘2d矩阵
		params_2_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
		params_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
		params_1_update += X.T.dot(layer_1_delta)

		future_layer_1_delta = layer_1_delta

	# 由于使用的是预测值 - 真值的方式，所以是这样更新参数
	params_1 -= params_1_update * alpha
	params_2 -= params_2_update * alpha
	params_h -= params_h_update * alpha

	params_1_update *= 0
	params_2_update *= 0
	params_h_update *= 0

	# print out progress
	if(j % 1000 == 0):  #每1000次打印结果
		print ("Error:" + str(overallerror))
		print ("Pred:" + str(pre))
		print ("True:" + str(c))
		out = 0
		for index,x in enumerate(reversed(pre)):
			out += x*pow(2,index)
		print (str(a_int) + " + " + str(b_int) + " = " + str(out))
		print ("------------")