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

pre_time = 0.1 # 每帧所需时间
run_path = os.path.split(os.path.realpath(__file__))[0] # 运行目录

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
        self.boxes = boxes       # boxes位置信息
        self.scores = scores     # boxes的置信度
        self.classid = classid   # boxes的id

class data():
    """
    description: 用于存储被检测目标的各种信息。
    """
    def __init__(self, now_x=0, now_y=0, last_x=0, last_y=0, delta_x=0, delta_y=0, distance=0):
        self.now_x = now_x
        self.now_y = now_y
        self.last_x = last_x
        self.last_y = last_y
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.distance = distance

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
    boxes_np = np.array(result_boxes.boxes)
    np_list = np.array([]) # 用于保存每一个boxes距离中心的距离

    # 计算谁离中心近
    for isb in range(boxes_np.shape[0]):
        np_list = np.append(np_list, 
                            float(((boxes_np[isb][0] + boxes_np[isb][2]) / 2 - (INPUT_RAW / 2)) ** 2 + 
                            ((boxes_np[isb][1] + boxes_np[isb][3]) - (INPUT_COL / 2)) ** 2))
    minBox_idx = np.argmin(np_list) if np_list else -1 # 获取距离中心最近的boxes的索引

    half_Weight = [229 / 4, 152 / 4]
    half_Height = [126 / 4, 142 / 4]

    # 下面进行的操作就是把现实世界3D坐标系的点位，通过相机内参得到2D平面上的点位
    # 可以看看这篇文章，写的还不错https://www.jianshu.com/p/1bf329da535b
    # 归根结底还是距离和速度的预测推算

    fx = 1056.4967111
    fy = 1056.6221413136
    cx = 657.4915775667
    cy = 508.2778608
    xxx = -0.392652606
    K = np.array([[fx, xxx, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)  # 相机内参矩阵

    cameraMatrix = K
    distCoeffs = None
    
        # 距离运算
    try:
        if minBox_idx != -1:
            detect_data.now_x = int((boxes_np[minBox_idx][0] + boxes_np[minBox_idx][2]) / 2)
            detect_data.now_y = int((boxes_np[minBox_idx][1] + boxes_np[minBox_idx][3]) / 2)
            detect_data.delta_x = detect_data.now_x - detect_data.last_x
            detect_data.delta_y = detect_data.now_y - detect_data.last_y

            box = result_boxes.boxes[minBox_idx]

            imgPoints = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
                                dtype=np.float64)
            idn = 0 if minBox_idx % 4 == 0 else 1
            objPoints = np.array([[-half_Weight[idn], -half_Height[idn], 0],
                                [half_Weight[idn], -half_Height[idn], 0],
                                [half_Weight[idn], half_Height[idn], 0],
                                [-half_Weight[idn], half_Height[idn], 0]], dtype=np.float64)
            retval, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)

            '''
            rvec_matrix = cv2.Rodrigues(rvec)[0]
            proj_matrix = np.hstack((rvec_matrix, rvec))
            eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
            pitch, yaw, roll = str(int(eulerAngles[0])), str(int(eulerAngles[1])), str(int(eulerAngles[2]))
            '''
            detect_data.distance = str(int(np.linalg.norm(tvec) / 10))
    except:
        print("Wrong Calculate!")
        
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
    xx_1, xx_2 = get_transdata_from_10b((int(detect_data.last_x + detect_data.delta_x / 2)))
    yy_1, yy_2 = get_transdata_from_10b((int(detect_data.last_y + detect_data.delta_y / 2)))
    speed_1, speed_2 = get_transdata_from_10b(int(500 * pre_time))
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

if __name__ == "__main__":
    """
    description:   运行主函数。
    """
    # 获取调试参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', nargs='+', type=int, default=5, help='The engine version that will be used. Default 5.')
    parser.add_argument('--engine', nargs='+', type=str, 
                        default=run_path+"/YOLOv5withTensorRT/build/best.engine", help='.engine path(s).')
    parser.add_argument('--library', nargs='+', type=str, 
                        default=run_path+"/YOLOv5withTensorRT/build/libmyplugins.so", help='libmyplugins.so path(s).')
    parser.add_argument('--color', nargs='+', type=int, default=1, help='Friend\'s color, 1 is red (default), 2 is blue.')
    parser.add_argument('--mode', nargs='+', type=str, default="debug", help='Running mode. debug (default) or release.')
    opt = parser.parse_args()
    RUN_MODE = 1 if opt.mode == "debug" else 0
    ctypes.CDLL(opt.library)
    ENGINE_FILE_PATH = opt.engine
    ENGINE_VERSION = opt.version
    categories = categories7 if ENGINE_VERSION == 7 else categories5

    if RUN_MODE:
        print("Debug Mode.")
        print(f"Enginepath: {ENGINE_FILE_PATH}")

    buffer = init_camera.buffer()
    ser = get_ser("/dev/ttyTHS0", 115200, 0.0001)                                             # 获取串口
    yolov5_wrapper = yolov5TRT.YoLov5TRT(ENGINE_FILE_PATH, CONF_THRESH, IOU_THRESHOLD)        # 初始化YOLOv5运行API
    check_friend_wrapper = check_friends(ser, opt.color, RUN_MODE, ENGINE_VERSION)            # 初始化友军检测类

    ''' 待与电控测试
    listening_thread = listening_ser()  # 运行监听线程
    listening_thread.start()
    '''
    detect_data = data() # 初始化数据信息
    # 循环检测目标与发送信息
    while 1:
        try:
            begin = time.time() # 计时开始

            frame = buffer.get_frame()   # 获取相机图像
            frame = cv2.resize(frame, (INPUT_RAW, INPUT_COL), interpolation=cv2.INTER_LINEAR)              # 裁切图像
            result = yolov5_wrapper.infer(frame)                                                           # 用YOLOv5检测目标
            result_boxes = boxes(*result)                                                                  # 将结果转化为boxes类
            result_boxes = check_friend_wrapper.get_enemy_info(result_boxes)                               # 得到敌军的boxes信息
            
            detect_data, minBox_idx = calculate_data(result_boxes, detect_data)                            # 计算测量结果
            trans_detect_data(ser, detect_data)                                                            # 发送检测结果

            detect_data.last_x = detect_data.now_x                                                         # 刷新数据
            detect_data.last_y = detect_data.now_y

            if minBox_idx != -1:                                                                           # 在图片上绘制检测框
                yolov5TRT.plot_one_box(result_boxes.boxes[minBox_idx], frame,
                                       label="{}:{:.2f}".format(categories[int(result_boxes.classid[minBox_idx])], 
                                       result_boxes.scores[minBox_idx]), )

            end = time.time()          # 结束计时
            pre_time = (end - begin)   # 统计用时

            cv2.waitKey(1)
            cv2.imshow("result", frame)                 # 显示图像输出
            if RUN_MODE:
                print(f"Frame Time: {pre_time * 1000}ms")   # 输出用时

        except Exception as e:
            print(e)