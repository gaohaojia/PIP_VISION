"""
该文件为主运行文件。
"""
import ctypes
import time
import cv2
import numpy as np
import argparse
import serial
import os
import threading

from cam_conf import init_camera
from check_friends import check_friends
import yolov5TRT

CONF_THRESH = 0.5
"""
置信度:
    识别后的置信度大于该值，识别结果保留。
"""
IOU_THRESHOLD = 0.5
"""
交并比:
    两个识别结果交并比大于该值，识别结果删除一个。
"""
RUN_MODE = 1
"""
运行模式:
    0: release模式，不显示任何调试信息和图像信息，只显示报错信息，节约性能。
    1: debug模式，显示全部的信息和图像，方便调试。
"""
ENGINE_VERSION = 7
"""
运行的模型:
    5: YOLOv5模型。
    7: YOLOv7模型。
"""
FRAME_RAW, FRAME_COL = 1280, 1024
"""
相机大小:
    相机输出的图像大小。
"""
INPUT_RAW, INPUT_COL = 640, 480
"""
检测大小:
    模型检测时的输入图像。
"""
TOLERANT_VALUE = 30
"""
容错值:
    预测坐标的容错范围（像素）。
"""

frame = None # 当前图像
run_path = os.path.split(os.path.realpath(__file__))[0] # 运行目录

# 一些相机参数常量，用于计算显示距离
fx = 1056.4967111
fy = 1056.6221413136
cx = 657.4915775667
cy = 508.2778608
xxx = -0.392652606
cameraMatrix = np.array([[fx, xxx, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)  # 相机内参矩阵
distCoeffs = None
half_Weight = [229 / 4, 152 / 4]
half_Height = [126 / 4, 142 / 4]

# 标签列表
categories5 = ["armor1red", "armor3red", "armor4red", "armor5red",           
              "armor1blue", "armor3blue", "armor4blue",
              "armor5blue", "armor1grey", "armor3grey", "armor4grey", "armor5grey"]

categories7 = ["armor1red", "armor1blue", "armor1grey",
               "armor2red", "armor2blue", "armor2grey",
               "armor3red", "armor3blue", "armor3grey",
               "armor4red", "armor4blue", "armor4grey",
               "armor5red", "armor5blue", "armor5grey",
               "armor7red", "armor7blue", "armor7grey",
               "bluewait", "bluedone", "nonactivate", "reddone", "redwait"]

class boxes():
    """
    description: 用于存储boxes各种信息的类。
    """
    def __init__(self, boxes, scores, classid):
        """
        param:
            boxes:   boxes位置信息。
            scores:  boxes的置信度。
            classid: boxes的id。
        """
        self.boxes = boxes      
        self.scores = scores    
        self.classid = classid  

class data():
    """
    description: 用于存储被检测目标的各种信息。
    """
    def __init__(self, now_x=-1, now_y=-1, last_x=-1, last_y=-1, distance=-1, pre_time=-1):
        """
        param:
            now_x:     当前帧的x坐标。
            now_y:     当前帧的y坐标。
            last_x:    上一帧的x坐标。
            last_y:    上一帧的y坐标。
            distance:  距离。
            pre_time:  每帧所需时间。
        """
        self.now_x = now_x
        self.now_y = now_y
        self.last_x = last_x
        self.last_y = last_y
        self.distance = distance
        self.pre_time = pre_time

def get_ser(port, baudrate, timeout):
    """
    description: Linux系统使用com1口连接串行口。
    param:
        baudrate:  端口。
        timeout:   超时时间。
    return:
        串口信息。
    """
    ser = serial.Serial(port, baudrate, timeout=timeout)
    if RUN_MODE:
        print(f"Serial Bytesize: {ser.bytesize}")
        print(f"Serial Parity:   {ser.parity}")
        print(f"Serial Stopbits: {ser.stopbits}")
    return ser

def get_transdata_from_10b(transdata):
    """
    description: 将10进制信息转化为用于通讯的16进制信息。
    param:
        transdata:  欲通讯的10进制信息。
    return:
        可用于通讯的16进制信息。
    """
    b16s = (4 - len(hex(transdata)[2:])) * '0' + hex(transdata)[2:]
    return [b16s[:2], b16s[2:]]

def calculate_data(result_boxes, detect_data):
    """
    description: 计算测量结果。
    param:
        result_boxes:   boxes类。
        detect_data:    识别的目标信息，data类。
    result:
        输出计算后的detect_data和距离中心最近的box的索引。
    """
    dis_list = [] # 用于保存每一个boxes距离中心的距离

    if result_boxes.boxes:

        # 计算与前一帧的偏移
        delta_x = detect_data.now_x - detect_data.last_x
        delta_y = detect_data.now_y - detect_data.last_y

        # 判断偏移量是否在容忍范围内
        if delta_x ** 2 + delta_y ** 2 <= TOLERANT_VALUE ** 2 and detect_data.last_x != -1 and detect_data.now_x != -1:

            # 计算与预测值之间的标准差
            pre_x = detect_data.now_x + delta_x
            pre_y = detect_data.now_y + delta_y
            dis_list = [float(((result_boxes.boxes[i][0] + result_boxes.boxes[i][2]) / 2 - pre_x) ** 2 + 
                              ((result_boxes.boxes[i][1] + result_boxes.boxes[i][3]) / 2 - pre_y) ** 2) 
                              for i in range(len(result_boxes.boxes))]
        else:

            # 计算与图像中心之间的标准差
            dis_list = [float(((result_boxes.boxes[i][0] + result_boxes.boxes[i][2]) / 2 - (INPUT_RAW / 2)) ** 2 + 
                              ((result_boxes.boxes[i][1] + result_boxes.boxes[i][3]) / 2 - (INPUT_COL / 2)) ** 2) 
                              for i in range(len(result_boxes.boxes))]
    
    minBox_idx = dis_list.index(min(dis_list)) if dis_list else -1 # 获取标准差最小boxes的索引

    detect_data.last_x = detect_data.now_x     # 刷新数据
    detect_data.last_y = detect_data.now_y

    if minBox_idx != -1:
        box = result_boxes.boxes[minBox_idx]                # 获取box

        detect_data.now_x = int((box[0] + box[2]) / 2)      # 计算当前帧x
        detect_data.now_y = int((box[1] + box[3]) / 2)      # 计算当前帧y

        """
        下面进行的操作就是把现实世界3D坐标系的点位，通过相机内参得到2D平面上的点位
        可以看看这篇文章，写的还不错https://www.jianshu.com/p/1bf329da535b
        归根结底还是距离和速度的预测推算
        """

        imgPoints = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
                            dtype=np.float64)
        idn = 0 if minBox_idx % 4 == 0 else 1
        objPoints = np.array([[-half_Weight[idn], -half_Height[idn], 0],
                              [ half_Weight[idn], -half_Height[idn], 0],
                              [ half_Weight[idn],  half_Height[idn], 0],
                              [-half_Weight[idn],  half_Height[idn], 0]], dtype=np.float64)
        retval, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)

        '''
        rvec_matrix = cv2.Rodrigues(rvec)[0]
        proj_matrix = np.hstack((rvec_matrix, rvec))
        eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
        pitch, yaw, roll = str(int(eulerAngles[0])), str(int(eulerAngles[1])), str(int(eulerAngles[2]))
        '''
        detect_data.distance = str(int(np.linalg.norm(tvec) / 10))
    else:
        detect_data.now_x = -1     # 未识别出box，则初始化当前帧x，y
        detect_data.now_y = -1

    return detect_data, minBox_idx

def trans_detect_data(ser, detect_data):
    """
    description: 将detect_data的信息与电控通讯。
    param:
        ser:             串口信息。
        detect_data:     计算后的数据，data类。
    """
    x_1, x_2 = get_transdata_from_10b((detect_data.now_x))
    y_1, y_2 = get_transdata_from_10b((detect_data.now_y))
    speed_1, speed_2 = get_transdata_from_10b(int(0.5 * detect_data.pre_time))
    dis_1, dis_2 = get_transdata_from_10b((int(detect_data.distance)))
    ZERO1, _ = get_transdata_from_10b(0)

    # 串口进行数据传输，传输距离，坐标，预测速度
    ser.write(b'\x45')
    try:
        ser.write(bytes.fromhex(dis_1))  # x-mid
        ser.write(bytes.fromhex(dis_2))
        ser.write(bytes.fromhex(x_1))  # x-mid
        ser.write(bytes.fromhex(x_2))
        ser.write(bytes.fromhex(y_1))  # x-mid
        ser.write(bytes.fromhex(y_2))
        ser.write(bytes.fromhex(speed_1))  # x-mid
        ser.write(bytes.fromhex(speed_2))
    except:
        ser.write(bytes.fromhex(ZERO1))  # x-mid
        ser.write(bytes.fromhex(ZERO1))
        ser.write(bytes.fromhex(ZERO1))  # x-mid
        ser.write(bytes.fromhex(ZERO1))
        ser.write(bytes.fromhex(ZERO1))  # x-mid
        ser.write(bytes.fromhex(ZERO1))
        ser.write(bytes.fromhex(ZERO1))  # x-mid
        ser.write(bytes.fromhex(ZERO1))
    
class listening_ser(threading.Thread):
    """
    description:  监听线程，用于监听电控信号。
    """
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while (1):
            get_ser = None
            try:
                get_ser = ser.read()
                if get_ser != None:
                    print(ser.read())
            except:
                pass

class get_frame(threading.Thread):
    """
    description:   用多线程获取图像，提高速度。
    """
    def __init__(self):
        """
        description:   初始化线程与相机。
        """
        threading.Thread.__init__(self)

    def run(self):
        """
        description:   循环获取图像。
        """
        global frame
        try:
            self.buffer = init_camera.buffer()
        except:
            frame = cv2.resize(cv2.imread("images/000001.jpeg"), (INPUT_RAW, INPUT_COL), interpolation=cv2.INTER_LINEAR)
            return
        while(1):
            raw_frame = self.buffer.get_frame()                                                                # 获取相机图像
            frame = cv2.resize(raw_frame, (INPUT_RAW, INPUT_COL), interpolation=cv2.INTER_LINEAR)              # 裁切图像
            time.sleep(0.003)
            

if __name__ == "__main__":
    """
    description:   运行主函数。
    """
    # 获取调试参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', nargs='?', type=int, default=7, help='The engine version that will be used. Default 5.')
    parser.add_argument('--engine', nargs='?', type=str, 
                        default=run_path+"/YOLOv5withTensorRT/build/best.engine", help='.engine path(s).')
    parser.add_argument('--library', nargs='?', type=str, 
                        default=run_path+"/YOLOv5withTensorRT/build/libmyplugins.so", help='libmyplugins.so path(s).')
    parser.add_argument('--color', nargs='?', type=int, default=1, help='Friend\'s color, 1 is red (default), 2 is blue.')
    parser.add_argument('--mode', nargs='?', type=str, default="debug", help='Running mode. debug (default) or release.')
    parser.add_argument('--image', nargs='?', type=str, default=None, help='Test image path. Default no test image.')
    opt = parser.parse_args()
    RUN_MODE = 1 if opt.mode == "debug" else 0                                  # 保存运行模式
    ctypes.CDLL(opt.library)                                                    # 调用TensorRT所需的library库
    ENGINE_FILE_PATH = opt.engine                                               # 保存模型文件目录
    ENGINE_VERSION = opt.version                                                # 保存模型版本
    categories = categories7 if ENGINE_VERSION == 7 else categories5            # 保存类别版本

    ser = get_ser("/dev/ttyTHS0", 115200, 0.0001)                                             # 获取串口
    yolov5_wrapper = yolov5TRT.YoLov5TRT(ENGINE_FILE_PATH, CONF_THRESH, IOU_THRESHOLD)        # 初始化YOLOv5运行API
    if RUN_MODE:
        print("\n\n\nDebug Mode.")
        print(f"Enginepath: {ENGINE_FILE_PATH}")
    check_friend_wrapper = check_friends(ser, opt.color, RUN_MODE, ENGINE_VERSION)            # 初始化友军检测类

    if opt.image is None:                                                                     # 判断是否为图像测试模式
        get_frame_thread = get_frame()                                                        # 启动获取图像线程
        get_frame_thread.start()
    else:
        raw_frame = cv2.imread("images/" + opt.image)                                         # 读入欲测试的图片
        frame = cv2.resize(raw_frame, (INPUT_RAW, INPUT_COL), interpolation=cv2.INTER_LINEAR)

    ''' 待与电控测试
    listening_thread = listening_ser()  # 运行监听线程
    listening_thread.start()
    '''
    detect_data = data() # 初始化数据信息
    # 循环检测目标与发送信息
    while 1:
        try:
            side1 = time.time() # 计时开始
            
            result = yolov5_wrapper.infer(frame)                                                           # 用YOLOv5检测目标
            result_boxes = boxes(*result)                                                                  # 将结果转化为boxes类

            side2 = time.time()
            result_boxes = check_friend_wrapper.get_enemy_info(result_boxes)                               # 得到敌军的boxes信息
            
            side3 = time.time()
            detect_data, minBox_idx = calculate_data(result_boxes, detect_data)                            # 计算测量结果
            trans_detect_data(ser, detect_data)                                                            # 发送检测结果

            if minBox_idx != -1:                                                                           # 在图片上绘制检测框
                yolov5TRT.plot_one_box(result_boxes.boxes[minBox_idx], frame,
                                       label="{}:{:.2f}".format(categories[int(result_boxes.classid[minBox_idx])], 
                                       result_boxes.scores[minBox_idx]), )

            side4 = time.time()                                     # 结束计时
            detect_data.pre_time = (side4 - side1) * 1000           # 统计用时
 
            cv2.waitKey(1) 
            cv2.imshow("Result", frame)                           # 显示图像输出
            if RUN_MODE: 
                # 输出用时
                print(f"\nTotal Time: {detect_data.pre_time}ms")
                print(f"Detect Time: {(side2 - side1) * 1000}ms")
                print(f"Get Enemy Time: {(side3 - side2) * 1000}ms")
                print(f"Calculate Time: {(side4 - side3) * 1000}ms")
 
        except Exception as e:
            print(e)