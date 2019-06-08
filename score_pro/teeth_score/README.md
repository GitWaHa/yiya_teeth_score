环境: Keras, Tensorflow, Opencv3.4, Python3.5

文件及文件夹说明:

​	U_net: 用U_net网络训练模型

​	application.py ：main函数所在位置
	teeth.py： 用于提取分割牙齿图片
	score.py： 用于对各项指标进行评分
	unet_extract.py： 调用unet模型对单个牙齿分割
	indicators.py： 定义各项指标的具体参数