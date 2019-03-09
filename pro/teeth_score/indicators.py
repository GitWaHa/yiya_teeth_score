'''
AA1（照片关键要素：10分）：
（1）术前：患牙两边各一颗对照牙齿，如患牙处于最里侧则对照牙不少于一颗
（2）术中：满足术前要素的前提下，患牙清理完毕处理等待充填的状态
（3）术后：满足术前要素的前提下，患牙处于已经治疗完毕的状态

'''
class Indicators_AA1():   # （10分）
    def __init__(self):
        self.neighbor_num = 0                # 具有相邻牙齿的数目
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
        self.first_angle = 0
        self.center_bias = 0.0      # 所补牙中心点与图片中心偏差得分（术前6分，其他4分）
        self.area_ratio = 0.0       # 所补牙与周边牙齿面积所占比例（4分）
        self.shooting_angle = 0.0   # 术后术中拍摄角度与术前的一致性（术前0分，其他2分）
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
        self.MIN_RESOLUTION = 1366*768

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
        self.THR_BLACK_DEPTH = 220
        self.THR_BLACK_SIZE = 220
        self.black_depth = 0.0                # 黑色深浅（10分）
        self.black_size = 0.0                 # 黑色大小（10分）
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
        self.edge_shape = 0.0                # 边缘形状是否圆润（10分）
        self.black_size = 0.0                # 黑色大小（10分）
        self.grade = 0

    def clear(self):
        self.edge_shape = 0.0
        self.black_size = 0.0
        self.grade = 0

    def sum(self):
        self.grade = round(self.edge_shape + self.black_size)

    def print(self):
        print("/**************** BB2 ******************/")
        print("BB2边缘形状得分：", self.edge_shape)
        print("BB2有无飞边得分：", self.black_size)
        print("BB2总分数：", self.grade)


'''
BB3（有无明显色差：20分）：
（1）术前：置0分
（2）术中：置0分
（3）术后：患牙充填材料与患牙周边牙体有无明显颜色差别（20分）。

'''
class Indicators_BB3():   # 指标
    def __init__(self):
        self.AVR_K = 0.5
        self.MAX_AVR_DIFF_H = 20
        self.MAX_AVR_DIFF_S = 20
        self.MAX_AVR_DIFF_V = 20

        self.MAX_VAR_DIFF_H = 3000
        self.MAX_VAR_DIFF_S = 3000
        self.MAX_VAR_DIFF_V = 3000
        self.h_avr = 0.0    # 色调均值
        self.s_avr = 0.0
        self.v_avr = 0.0
        self.h_var = 0.0    # 色调方差
        self.s_var = 0.0
        self.v_var = 0.0
        self.grade = 0

    def clear(self):
        self.h_avr = 0.0
        self.s_avr = 0.0
        self.v_avr = 0.0
        self.h_var = 0.0
        self.s_var = 0.0
        self.v_var = 0.0
        self.grade = 0

    def sum(self):
        self.grade = round(self.h_avr + self.s_avr + self.v_avr + self.h_var + self.s_var + self.v_var + 0.5)

    def print(self):
        print("/**************** BB3 ******************/")
        print("BB3均值hsv：", self.h_avr, self.s_avr, self.v_avr)
        print("BB3方差hsv：", self.h_var, self.s_var, self.v_var)
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
        self.THR_GAP_NUM = 350
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
