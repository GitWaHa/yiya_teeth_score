环境: Keras 2.2.4, Tensorflow 1.5.0, Opencv3.1.0.5, Python3.6


文件及文件夹说明:
	BB4_standard_template：存放BB4牙齿标准模版	

	AlexNet：分类拍摄角度（门牙视角与后牙视角）
	ResNet: 预分类牙齿好中坏程度
	lr_mode :线性回归评分模型
​	U_net: 用U_net分割单个牙齿
	Yolo3： 实现自动标注牙齿位置信息

​	application.py ：main函数所在位置
	teeth.py： 用于提取分割牙齿图片
	score.py： 用于对各项指标进行评分
	indicators.py： 定义各项指标的具体参数