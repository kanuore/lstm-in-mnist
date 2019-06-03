#coding=utf-8
# 测试reshape的工作原理，发现他并不能保证数据结构的完整性
import numpy as np

data = np.array(
	[	
		[
			[1,2,3],
			[4,5,6]
		],
		[
			[7,8,9],
			[10,11,12]
		],
		[	
			[13,14,15],
			[16,17,18]
		]
	]
	)

print data.shape

print data.reshape(2,3,3)