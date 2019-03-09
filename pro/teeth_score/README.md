环境: Keras, Tensorflow, Opencv3.4, Python3.5

说明: 务必使用Python3运行Python脚本,网络的训练必须使用Python3,否则会报错或无法获得正确结果.

UNet-Keras实现参考: 

​	博客: https://blog.csdn.net/u012931582/article/details/70215756

​	Github: https://github.com/zhixuhao/unet



文件及文件夹说明:

​	tooth_segmentation_label_black_reference.py 为最终版本, 可在画出标注框后生成标注文件, 判断有无参照牙齿, 分割出目标牙齿, 并识别龋齿.

​	trainUnet.py 用于模型训练, testUnet.py用于模型测试

​	data_generator 文件夹用于获取训练数据及标注文件, 运行create_label_batch.py, 使用矩形框选中目标牙齿区域后, 可以对其进行标注, 通过键盘进行图片保存等操作.保存的图片文件包括原始图\灰度图\标注图, 统一保存在labeled_data文件夹下.