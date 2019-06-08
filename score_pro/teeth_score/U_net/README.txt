环境: Keras, Tensorflow, Opencv3.4, Python3.5
说明: 务必使用Python3运行Python脚本,网络的训练必须使用Python3,否则会报错或无法获得正确结果.

UNet-Keras实现参考: 
​	博客: https://blog.csdn.net/u012931582/article/details/70215756
​	Github: https://github.com/zhixuhao/unet

文件及文件夹说明:
	code_python : 存放基于KERAS的Python代码
		trainUnet.py 用于模型训练
		testUnet.py用于模型测试
		model：模型建立
		data: 对数据进行预处理
	data_generator : 用于对ROI区域进行标记，来制造训练所需数据
		create_label_batch.py： 运行create_label_batch.py, 使用矩形框选中目标牙齿区域后,可以对其进行标注, 通过键盘进行图片保存等操作.保存的图片文件包括原始图\灰度图\标注图, 统一保存在labeled_data文件夹下.
	data : 保存训练，测试所需数据集等
		train: 训练模型所用数据集
		test: 模型测试所用原始数据集
		test_out: 模型测试输出
	model_hdf5: 最终模型参数保存位置
​	


