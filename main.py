import ctypes
import time
import cv2
import torch
import numpy as np
import argparse
import serial
import os
import threading

from cam_conf import init_camera
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
FOCUSING_MODEL = False
"""
聚焦模式:
    启用后目标仅检测图像中心附近的目标。
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

def scale_boxes(result_boxes):
    """
    description: 按比例缩放boxes。
    param:
        result_boxes:    欲缩放的boxes。
        raw_rate:        横向缩放比率。
        col_rate:        纵向缩放比率。
    return:
        缩放后的boxes。
    """
    if result_boxes.boxes:
        boxes_np = np.array(result_boxes.boxes)
        for idx in range(boxes_np.shape[0]):
            raw_rate = INPUT_RAW / FRAME_RAW
            col_rate = INPUT_COL / FRAME_COL
            boxes_np[idx][0] = (boxes_np[idx][0] + int((FRAME_RAW - INPUT_RAW) / 2)) * raw_rate
            boxes_np[idx][2] = (boxes_np[idx][2] + int((FRAME_RAW - INPUT_RAW) / 2)) * raw_rate
            boxes_np[idx][1] = (boxes_np[idx][1] + int((FRAME_COL - INPUT_COL) / 2)) * col_rate
            boxes_np[idx][3] = (boxes_np[idx][3] + int((FRAME_COL - INPUT_COL) / 2)) * col_rate
    return result_boxes

def trans_detect_data(ser, result_boxes, image_raw):
    """
    description: 将detect的信息与电控通讯。
    param:
        ser:             串口信息。
        result_boxes:    boxes类。
        image_raw:       标注后的图片。
    """
    # 计算处理时间
    side2 = time.time()
    boxes_np = np.array(result_boxes.boxes)
    inde = boxes_np.shape[0]
    numlist = []

    # 绘制聚焦区域
    if FOCUSING_MODEL:
            start_col = int((FRAME_COL - INPUT_COL) / 2) * (INPUT_COL / FRAME_COL)
            end_col = int((FRAME_COL + INPUT_COL) / 2) * (INPUT_COL / FRAME_COL)
            start_raw = int((FRAME_RAW - INPUT_RAW) / 2) * (INPUT_RAW / FRAME_RAW)
            end_raw = int((FRAME_RAW + INPUT_RAW) / 2) * (INPUT_RAW / FRAME_RAW)
            yolov5TRT.plot_one_box([start_raw, start_col, end_raw, end_col], image_raw, [150, 150, 150], label="", )

    # 计算谁离中心近
    for isb in range(inde):
        numlist.append(
            float(((boxes_np[isb][0] + boxes_np[isb][2]) / 2 - (INPUT_RAW / 2)) ** 2 + ((boxes_np[isb][1] + boxes_np[isb][3]) - (INPUT_COL / 2)) ** 2))
    mindex = -1
    if len(numlist) == 0:
        mindex = -1
    else:
        mindex = np.argmin(numlist)

    # 距离运算
    try:
        global pre_x, pre_y
        x_now = int((boxes_np[mindex][0] + boxes_np[mindex][2]) / 2)
        y_now = int((boxes_np[mindex][1] + boxes_np[mindex][3]) / 2)
        x_1, x_2 = get_transdata_from_10b((x_now))
        y_1, y_2 = get_transdata_from_10b((y_now))
        detax = x_now - pre_x
        detay = y_now - pre_y
        # 计算欧氏距离
        xx_1, xx_2 = get_transdata_from_10b((int(pre_x + detax / 2)))
        yy_1, yy_2 = get_transdata_from_10b((int(pre_y + detay / 2)))
        deta_dis = np.sqrt((detay ** 2 + detax ** 2))
        # 根据距离和时间步长推测速度
        speed_1, speed_2 = get_transdata_from_10b(int(500 * pre_time))
        pre_x = x_now
        pre_y = y_now
        if RUN_MODE:
            print(x_1, x_2, xx_1, xx_2)
    except:
        if RUN_MODE:
            print("Wrong Trans!")

    side3 = time.time()
    if RUN_MODE:
        print(f"Side3 Time: \t{(side3 - side2) * 1000:.3f}")

    tag_size = 0.05
    tag_size_half = 0.02
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

    side4 = time.time()
    if RUN_MODE:
        print(f"Side4 Time: \t{(side4 - side3) * 1000:.3f}")
    
    if mindex != -1:
        box = result_boxes.boxes[mindex]

        imgPoints = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
                             dtype=np.float64)
        if mindex % 4 == 0:
            idn = 0
        else:
            idn = 1
        objPoints = np.array([[-half_Weight[idn], -half_Height[idn], 0],
                              [half_Weight[idn], -half_Height[idn], 0],
                              [half_Weight[idn], half_Height[idn], 0],
                              [-half_Weight[idn], half_Height[idn], 0]], dtype=np.float64)
        retval, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)

        distance = np.linalg.norm(tvec)

        
        if box:
            yolov5TRT.plot_one_box(box, image_raw,
                                   label="{}:{:.2f}".format(categories[int(result_boxes.classid[mindex])], 
                                   result_boxes.scores[mindex]), )

        rvec_matrix = cv2.Rodrigues(rvec)[0]
        proj_matrix = np.hstack((rvec_matrix, rvec))
        eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
        pitch, yaw, roll = str(int(eulerAngles[0])), str(int(eulerAngles[1])), str(int(eulerAngles[2]))
        distance = str(int(distance / 10))
        if RUN_MODE:
            print(f"pryd{pitch}, {yaw}, {roll}, {distance}")

        dis_1, dis_2 = get_transdata_from_10b((int(distance)))
    zero11, zero2 = get_transdata_from_10b(0)

    # 串口进行数据传输,传输距离，坐标，预测速度
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
        ser.write(bytes.fromhex(zero11))  # x-mid
        ser.write(bytes.fromhex(zero11))
        ser.write(bytes.fromhex(zero11))  # x-mid
        ser.write(bytes.fromhex(zero11))
        ser.write(bytes.fromhex(zero11))  # x-mid
        ser.write(bytes.fromhex(zero11))
        ser.write(bytes.fromhex(zero11))  # x-mid
        ser.write(bytes.fromhex(zero11))
    
    side5 = time.time()
    if RUN_MODE:
        print(f"Side5 Time: \t{(side5 - side4) * 1000:.3f}")

class check_friends():
    """
    description: 检测友军
    """
    def __init__(self, ser, color):
        """
        description: 初始化变量以及尝试获取友军信息。
        param:
            ser:    串口信息。
        """
        self.color = color
        self.friends = []
        self.check_fr = 0

        # 尝试10次通讯获取，防止1次获取失败。
        for i in range(10):
            if (self.check_fr):
                break
            self.get_color_and_friends(ser)

    def get_color_and_friends(self, ser):
        """
        description: 与电控通讯，获取友军颜色与友军id。
        param:
            ser:    串口信息。
        """
        try:
            ser.write(b'\x45')
        except:
            print("Wrong Open Serial!")
        
        # TODO:还需要添加风车的编号，已经打过的风车和灰色风车都标记为友军
        # TODO:目前的想法：1.红蓝色已击打看作一个标签进行训练识别   2.红蓝色已击打分为两类训练识别

        if RUN_MODE:
            print(f"Recieve: {ser.read()}")
        # 根据我方红蓝方的设定，进行友军识别
        if ser.read() == b'\xff' or self.color == 1:
            self.color = 1  # red
            self.friends = [0, 3, 6, 9, 12, 15] if ENGINE_VERSION == 7 else [0, 1, 2, 3]
        elif ser.read() == b'\xaa' or self.color == 2:
            self.color = 2  # blue
            self.friends = [1, 4, 7, 10, 13, 16] if ENGINE_VERSION == 7 else [4, 5, 6, 7]
        if RUN_MODE:
            print(f"Friend id: {self.friends}") if self.friends else print("No friend id!")

        # 如果是友军而且友军列表成功添加，那么友军标记边变1，并且友军列表添加死亡的敌人
        fr = []
        if self.check_fr == 0 and self.friends:
            fr = self.friends
            self.check_fr = 1
        self.friends_list = fr + ([2, 5, 8, 11, 14, 17, 19, 20, 21] if ENGINE_VERSION == 7 else [8, 9, 10, 11])

    def get_nonfriend_from_all(self, all, friends):
        """
        description: 获取非友军相关参数。
        param:
            all:    全部参数。
            friend: 友军参数。
        return:
            非友军参数。
        """
        new = []
        for i in all.numpy().tolist():
            if i not in (friends):
                new.append(i)
        return torch.tensor(new)

    def get_enemy_info(self, result_boxes):
        """
        description: 处理识别的box，输出敌军的box信息。
        param:
            result_boxes:    boxes类。
        return:
            只含敌军的boxes类。
        """
        # 分别代表友军的box、box置信度、box的id
        exit_friends_boxes = []
        exit_friends_scores = []
        exit_friends_id = []
        friends_id = []
        for ii in range(len(result_boxes.classid)):
            if int(result_boxes.classid.numpy()[ii]) in self.friends_list:
                friends_id.append(int(result_boxes.classid.numpy()[ii]))
                exit_friends_boxes.append(result_boxes.boxes[ii])
                exit_friends_scores.append(result_boxes.scores[ii])
                exit_friends_id.append(result_boxes.classid[ii])
        if RUN_MODE:
            print(f"Friend Id: {friends_id}") if friends_id else print("No friend id!")
        enemy_list_index = []

        # 获取敌军的列表以及id
        try:
            for i in result_boxes.classid.numpy():
                if int(i) not in friends_id:
                    dex_tem = ((np.where(result_boxes.classid.numpy() == i))[0][0])
                    enemy_list_index.append(dex_tem)
        except:
            "g"

        if RUN_MODE:
            print(f"Enemy Id: {enemy_list_index}") if enemy_list_index else print("No enemy id!")

        ourbox = []
        for dex in enemy_list_index:
            ourbox.append(result_boxes.boxes[dex].numpy())

        result_boxes.boxes = ourbox
        result_boxes.scores = self.get_nonfriend_from_all(result_boxes.scores, exit_friends_scores)  # 置信度处理
        result_boxes.classid = self.get_nonfriend_from_all(result_boxes.classid, exit_friends_id)    # id处理

        if RUN_MODE:
            print(f"Nowboxes: {result_boxes.boxes}")
            print(f"Nowscore: {result_boxes.scores}")
            print(f"Nowid: {result_boxes.classid}")
        return result_boxes
    
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
    parser.add_argument('--focusing', nargs='+', type=bool, default=False, help='Is activate focusing model. Default False.')
    opt = parser.parse_args()
    RUN_MODE = 1 if opt.mode == "debug" else 0
    ctypes.CDLL(opt.library)
    ENGINE_FILE_PATH = opt.engine
    ENGINE_VERSION = opt.version
    FOCUSING_MODEL = opt.focusing
    categories = categories7 if ENGINE_VERSION == 7 else categories5

    if RUN_MODE:
        print("Debug Mode.")
        print(f"Enginepath: {ENGINE_FILE_PATH}")
        if FOCUSING_MODEL:
            print("Focusing model activated.")

    buffer = init_camera.buffer()
    ser = get_ser("/dev/ttyTHS0", 115200, 0.0001)                                             # 获取串口
    yolov5_wrapper = yolov5TRT.YoLov5TRT(ENGINE_FILE_PATH, CONF_THRESH, IOU_THRESHOLD)        # 初始化YOLOv5运行API
    check_friend_wrapper = check_friends(ser, opt.color)                                      # 初始化友军检测类

    ''' 待与电控测试
    listening_thread = listening_ser()  # 运行监听线程
    listening_thread.start()
    '''
    
    # 循环检测目标与发送信息
    while 1:
        try:
            begin = time.time() # 计时开始

            frame = buffer.get_frame()   # 获取相机图像

            input_frame = None
            if FOCUSING_MODEL:
                start_col, end_col = int((frame.shape[0] - INPUT_COL) / 2), int((frame.shape[0] + INPUT_COL) / 2)
                start_raw, end_raw = int((frame.shape[1] - INPUT_RAW) / 2), int((frame.shape[1] + INPUT_RAW) / 2)
                input_frame = frame[start_col:end_col, start_raw:end_raw, :]                               # 裁切图像用于聚焦识别
            else:
                input_frame = cv2.resize(frame, (INPUT_RAW, INPUT_COL), interpolation=cv2.INTER_LINEAR)
                frame = input_frame

            result = yolov5_wrapper.infer(input_frame)                                                     # 用YOLOv5检测目标
            result_boxes = boxes(*result)                                                                  # 将结果转化为boxes类
            result_boxes = check_friend_wrapper.get_enemy_info(result_boxes)                               # 得到敌军的boxes信息
            if FOCUSING_MODEL:
                result_boxes = scale_boxes(result_boxes)                                                   # 将结果还原到原图像
                frame = cv2.resize(frame, (INPUT_RAW, INPUT_COL), interpolation=cv2.INTER_LINEAR)
            
            trans_detect_data(ser, result_boxes, frame)                                                    # 发送检测结果

            end = time.time()          # 结束计时
            pre_time = (end - begin)   # 统计用时

            cv2.waitKey(1)
            cv2.imshow("result", frame)                 # 显示图像输出
            if RUN_MODE:
                print(f"Frame Time: {pre_time * 1000}ms")   # 输出用时

        except Exception as e:
            print(e)