#!/usr/bin/python3.6
# coding=utf-8

'''
AA1（照片关键要素：10分）：
（1）术前：患牙两边各一颗对照牙齿，如患牙处于最里侧则对照牙不少于一颗
（2）术中：满足术前要素的前提下，患牙清理完毕处理等待充填的状态
（3）术后：满足术前要素的前提下，患牙处于已经治疗完毕的状态

'''
class Indicators_AA1():   # （10分）
    def __init__(self):
        self.AREA_K = 1.5                    # 面积得分*此系数 对包含相邻牙齿得分进行扣分
        self.THR_HEIGHT = 0.25               # 大于 ROI*THR_HEIGHT 判断有相邻牙齿
        self.THR_WIDTH = 0.5                 # 大于 ROI*THR_WIDTH 判断相邻牙齿完整
        self.CONTAINS_NEIGHBOR_SCORE = 7     # 包含相邻牙齿总分数
        self.UNDEFINED_SCORE = 3                   # 未定义三分

        self.neighbor_num = 0
        self.contains_neighbor = 0.0         # 患牙是否含有相邻牙齿得分（7分）
        self.undefined = 0.0                 # 未定义（3分）
        self.grade = 0

    def clear(self):
        self.neighbor_num = 0
        self.contains_neighbor = 0.0
        self.undefined = 0.0
        self.grade = 0

    def sum(self):
        self.grade = round(self.contains_neighbor + self.undefined)

    def print(self):
        print("/**************** AA1 ******************/")
        print("AA1有无相邻牙齿得分：", self.contains_neighbor)
        # print("AA1比例得分：", self.img_ratio)
        # print("AA1分辨率得分：", self.img_resolution)
        print("AA1总分数：", self.grade)


'''
AA2（照片构图合理：10分）：
（1）术前：患牙的中心点与图片X交叉线的中心点，即患牙处于图片中心区域（6分，偏10%~15%扣1.5分，以此类推，扣完为止）；
          患牙及相邻牙占图片整体60%-70%的面积（4分，每高或低7%扣1分，扣完为止）。
（2）术中：患牙的中心点与图片X交叉线的中心点，即患牙处于图片中心区域（4分，偏10%~15%扣1分，以此类推，扣完为止）；
          患牙及相邻牙占图片整体60%-70%的面积（4分，每高或低7%扣1分，扣完为止）；
          术中照片中患牙的拍摄角度跟术前保持一致（2分，每偏差30扣1分，扣完为止）。
（3）术后：同术中
'''
class Indicators_AA2():
    def __init__(self):
        self.CENTER_BIAS_SCORE_FIRST = 6        # 所补牙中心点与图片中心偏差 总分（术前6分，其他4分）
        self.CENTER_BIAS_SCORE_OTHER = 4
        self.CENTER_BIAS_SUBTRACT_FIRST = 1.5   # 所补牙中心点与图片中心偏差 扣分值 （术前1.5分，其他1分）
        self.CENTER_BIAS_SUBTRACT_OTHER = 1
        self.CENTER_BIAS_SUBTRACT_RATIO = 0.05  # 所补牙中心点与图片中心偏差 每此比例扣一次分
        self.CENTER_BIAS_SUBTRACT_START = 0.03  # 所补牙中心点与图片中心偏差 大于此比例开始扣分
        self.CENTER_BIAS_SUBTRACT_START_1 = 0.08  # 所补牙中心点与图片中心偏差 大于此比例开始扣分
        self.CENTER_BIAS_SUBTRACT_START_2 = 0.15  # 所补牙中心点与图片中心偏差 大于此比例开始扣分
        self.CENTER_BIAS_SUBTRACT_START_3 = 0.30  # 所补牙中心点与图片中心偏差 大于此比例开始扣分

        self.AREA_RATIO_SCORE = 4                   # 所补牙与周边牙齿面积所占比例 总分
        self.AREA_RATIO_SUBTRACT_START_MAX = 0.7    # 所补牙与周边牙齿面积所占比例 大于此比例开始扣分
        self.AREA_RATIO_SUBTRACT_START_MIN = 0.4    # 所补牙与周边牙齿面积所占比例 小于此比例开始扣分
        self.AREA_RATIO_SUBTRACT_RATIO = 0.05       # 所补牙与周边牙齿面积所占比例 每此比例扣一次分
        self.AREA_RATIO_SUBTRACT = 1                # 所补牙与周边牙齿面积所占比例 扣分值
        self.AREA_RATIO_MIN = 0.1   

        self.SHOOTING_ANGLE_SCORE_FIRST = 0         # 术后术中拍摄角度与术前的一致性 总分 （术前0分，其他2分）
        self.SHOOTING_ANGLE_SCORE_OTHER = 2
        self.SHOOTING_ANGLE_SUBTRACT_ANGLE = 30     # 术后术中拍摄角度与术前的一致性 每此角度扣一次分
        self.SHOOTING_ANGLE_SUBTRACT = 1            # 术后术中拍摄角度与术前的一致性 扣分值

        self.first_angle = 0
        self.center_bias = 0.0      # 所补牙中心点与图片中心偏差得分
        self.area_ratio = 0.0       # 所补牙与周边牙齿面积所占比例
        self.shooting_angle = 0.0   # 术后术中拍摄角度与术前的一致性
        self.grade = 0

    def clear(self):
        self.center_bias = 0.0       #
        self.area_ratio = 0.0        #
        self.shooting_angle = 0.0    #
        self.grade = 0

    def sum(self):
        self.grade = round(self.center_bias + self.area_ratio + self.shooting_angle)

    def print(self):
        print("/**************** AA2 ******************/")
        print("AA2中心位置偏差得分：", self.center_bias)
        print("AA2面积比例得分：", self.area_ratio)
        print("AA2拍摄角度一致得分：", self.shooting_angle)
        print("AA2总分数：", self.grade)


'''
AA3（照片清晰度：10分）：
（1）术前：JPG文件格式（4分，必要条件，达不到则AA3直接0分）；照片比例16：9（在3%的误差范围内4分，必要条件，达不到则AA3直接0分）；1366*768分辨率的照片（2分，可以高于，但每低100K扣1分，扣完为止）。
（2）术中：同上。
（3）术后：同上。

'''
class Indicators_AA3():   # 指标
    def __init__(self):
        
        self.IMG_TYPE_SCORE = 4

        self.IMG_RATIO_SCORE = 4
        self.IMG_RATIO_STD = 16 / 9 
        self.IMG_RATIO_ERROR = 0.03     # 比例大于此误差零分

        self.IMG_RESOLUTION_SCORE = 4
        self.IMG_RESOLUTION_SUBTRACT_START = 1366*768   # 分辨率 小于此值开始扣分
        self.IMG_RESOLUTION_SUBTRACT = 1                # 分辨率 扣分值
        self.IMG_RESOLUTION_SUBTRACT_SIZE = 100000      # 分辨率 每此值扣一次分

        self.img_type = 0.0         # 图片格式
        self.img_ratio = 0.0        # 图片比例
        self.img_resolution = 0.0   # 分辨率
        self.grade = 0

    def clear(self):
        self.img_type = 0.0         # 图片格式
        self.img_ratio = 0.0        # 图片比例
        self.img_resolution = 0.0   # 分辨率
        self.grade = 0

    def sum(self):
        self.grade = round(self.img_type + self.img_ratio + self.img_resolution)

    def print(self):
        print("/**************** AA3 ******************/")
        print("AA3格式得分：", self.img_type)
        print("AA3比例得分：", self.img_ratio)
        print("AA3分辨率得分：", self.img_resolution)
        print("AA3总分数：", self.grade)


'''
BB1（龋坏清干净否，针对术中20分）：
（1）术前：置0分。
（2）术中：患牙无色素着色，不能有黑色
    （黑色深浅：10分，灰度值低于220时，像素点的最大灰度值每低20扣1分，扣完为止；
      黑色大小：10分，灰度值低于220认定为黑色像素，每30个黑色像素扣1分，扣完为止；）。
（3）术后：直接20分。

'''
class Indicators_BB1():   # （10分）
    def __init__(self):
        self.THR_GRAY = 60

        self.BLACK_DEPTH_SCORE = 10             # 黑色深浅 分数
        self.BLACK_DEPTH_SUBTRACT = 1           # 黑色深浅 扣分值
        self.BLACK_DEPTH_SUBTRACT_VALUE = 10    # 黑色深浅 每此值扣一次分

        self.BLACK_SIZE_SCORE = 10              # 黑色大小 分数
        self.BLACK_SIZE_SUBTRACT = 1            # 黑色大小 扣分值
        self.BLACK_SIZE_SUBTRACT_VALUE = 30     # 黑色大小 每此值扣一次分

        self.black_depth = 0.0                # 黑色深浅
        self.black_size = 0.0                 # 黑色大小
        self.grade = 0

    def clear(self):
        self.black_depth = 0.0
        self.black_size = 0.0
        self.grade = 0

    def sum(self):
        self.grade = round(self.black_depth + self.black_size)

    def print(self):
        print("/**************** BB1 ******************/")
        print("BB1黑色深浅得分：", self.black_depth)
        print("BB1黑色大小得分：", self.black_size)
        print("BB1总分数：", self.grade)


'''
BB2（制备洞型后形态，针对术中20分）：
（1）术前：置0分。
（2）术中：边缘圆润（10分，每1个尖峰扣2分，扣完为止）；
    有无飞边：患牙边缘线是否平滑、无锐利角及边缘（10分，但每低100K扣1分，扣完为止）。
    难点1：由于不同牙齿的制备洞型后形态是不一致的：门牙、磨牙等，在患牙诊断时，会给出是哪一类。
（3）术后：直接20分。
'''
class Indicators_BB2():   # （10分）
    def __init__(self):
        self.oneself_diff = 0.0                # 与自己牙齿色差，即是否平滑过渡(20)
        self.grade = 0

    def clear(self):
        self.oneself_diff = 0.0
        self.grade = 0

    def sum(self):
        self.grade = round(self.oneself_diff)

    def print(self):
        print("/**************** BB2 ******************/")
        print("BB2与自己牙齿色差得分：", self.oneself_diff)
        print("BB2总分数：", self.grade)


'''
BB3（所补牙与本牙颜色是否一致，20分）：
（1）术前：置0分。
（2）术中：置0分
（3）术后：患牙充填材料与患牙周边牙体有无明显颜色差别（20分）。
    其中，患牙整体与两边邻牙整体颜色是否一致（10分），整体分为上中下+左中右9个区域；
    患牙所补区域（结合术中图片获得所补区域）与本牙的其他区域颜色过渡是否平滑（10分）。

'''
class Indicators_BB3():   # 指标
    def __init__(self):
        self.MAX_AVR_DIFF_H = 5
        self.MAX_AVR_DIFF_S = 100

        self.other_diff = 0.0       # 与相邻牙齿色差
        self.oneself_diff = 0.0     # 与自己牙齿色差，即是否平滑过渡
        self.roi_site = 0, 0, 0, 0  # 保存位置信息 

        self.grade = 0

    def clear(self):
        self.other_diff = 0.0
        self.oneself_diff = 0.0

        self.grade = 0

    def sum(self):
        self.grade = round(self.other_diff + self.oneself_diff + 0.5)

    def print(self):
        print("/**************** BB3 ******************/")
        print("BB3与相邻牙齿色差得分：", self.other_diff)
        print("BB3与自己牙齿色差得分：", self.oneself_diff)
        print("BB3总分数：", self.grade)


'''
BB4（有无牙齿尖窝形态，10分，只限于两侧后牙，上下左右各4颗，其他牙给10分）
（1）术前：置0分。
（2）术中：置0分。
（3）术后：有无明显的沟窝纹路（10分，按沟窝纹路匹配度，每10%得1.5分，70%以上得10分）。
    难点2：每侧后牙有8颗门：分8类，右上4类，右下4类。在患牙诊断时，会给出是哪一颗牙，例如右下7。
'''
class Indicators_BB4():
    def __init__(self):
        self.GAP_SCORE = 10
        self.GAP_SUBTRACT = 1.5
        self.GAP_SUBTRACT_START = 0.7
        self.GAP_SUBTRACT_RATIO = 0.1

        self.gap = 0.0    # 沟壑纹路得分
        self.grade = 0

    def clear(self):
        self.gap = 0.0
        self.grade = 0

    def sum(self):
        self.grade = round(self.gap)

    def print(self):
        print("/**************** BB4 ******************/")
        print("BB4沟壑纹路得分：", self.gap)
        print("BB4总分数：", self.grade)
