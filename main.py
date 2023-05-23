import ctypes
import time
import cv2
import torch
import numpy as np
import argparse
import serial
import os

from cam_conf import init_camera
from cam_conf import mvsdk
import yolov5TRT

# 两个阈值
# 本阈值是置信度,本代码有算法来对友军和敌军进行识别,并记录识别后的敌友置信度
# 当置信度大于下方阈值,视为敌友识别成功,此时敌友信息才会真正传送给机器人
CONF_THRESH = 0.5
# 本阈值是代码末端nms(非极大抑制)算法所用,IOU可以理解为相邻两个锚框的重叠率
# 重叠率达到这个数值,那么前一个锚框就会被舍去
IOU_THRESHOLD = 0.5

pre_time = 0.1 # 每帧所需时间
run_path = os.path.split(os.path.realpath(__file__))[0] # 运行目录
run_mode = 1 # 运行模式
"""
运行模式:
    0: release模式，不显示任何调试信息和图像信息，只显示报错信息，节约性能。
    1: debug模式，显示全部的信息和图像，方便调试。
"""

model = 7
"""
运行的模型:
    5: YOLOv5模型。
    7: YOLOv7模型。
"""

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
    if run_mode:
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

    # 计算谁离中心近
    for isb in range(inde):
        numlist.append(
            float(((boxes_np[isb][0] + boxes_np[isb][2]) / 2 - 320) ** 2 + ((boxes_np[isb][1] + boxes_np[isb][3]) - 240) ** 2))
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
        if run_mode:
            print(x_1, x_2, xx_1, xx_2)
    except:
        print("Wrong Trans!")

    side3 = time.time()
    if run_mode:
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
    if run_mode:
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

        try:
            yolov5TRT.plot_one_box(box, image_raw,
                         label="{}:{:.2f}".format(categories[int(result_boxes.classid[mindex])], result_boxes.scores[mindex]), )
        except:
            '''g'''

        rvec_matrix = cv2.Rodrigues(rvec)[0]
        proj_matrix = np.hstack((rvec_matrix, rvec))
        eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
        pitch, yaw, roll = str(int(eulerAngles[0])), str(int(eulerAngles[1])), str(int(eulerAngles[2]))
        distance = str(int(distance / 10))
        if run_mode:
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
    if run_mode:
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

        if run_mode:
            print(f"Recieve: {ser.read()}")
        # 根据我方红蓝方的设定，进行友军识别
        if ser.read() == b'\xff' or self.color == 1:
            self.color = 1  # blue
            self.friends = [0, 3, 6, 9, 12, 15] if model == 7 else [0, 1, 2, 3]
        elif ser.read() == b'\xaa' or self.color == 2:
            self.color = 2  # red
            self.friends = [1, 4, 7, 10, 13, 16] if model == 7 else [4, 5, 6, 7]
        if run_mode:
            print(f"Friend id: {self.friends}") if self.friends else print("No friend id!")

        # 如果是友军而且友军列表成功添加，那么友军标记边变1，并且友军列表添加死亡的敌人
        fr = []
        if self.check_fr == 0 and self.friends:
            fr = self.friends
            self.check_fr = 1
        self.friends_list = fr + ([2, 5, 8, 11, 14, 17, 19, 20, 21] if model == 7 else [8, 9, 10, 11])

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
        if run_mode:
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

        if run_mode:
            print(f"Enemy Id: {enemy_list_index}") if enemy_list_index else print("No enemy id!")

        ourbox = []
        for dex in enemy_list_index:
            ourbox.append(result_boxes.boxes[dex].numpy())

        result_boxes.boxes = ourbox
        result_boxes.scores = self.get_nonfriend_from_all(result_boxes.scores, exit_friends_scores)  # 置信度处理
        result_boxes.classid = self.get_nonfriend_from_all(result_boxes.classid, exit_friends_id)    # id处理

        if run_mode:
            print(f"Nowboxes: {result_boxes.boxes}")
            print(f"Nowscore: {result_boxes.scores}")
            print(f"Nowid: {result_boxes.classid}")
        return result_boxes

if __name__ == "__main__":
    """
    description:   运行主函数。
    """
    # 获取调试参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', type=int, default=5, help='The model that will be used. Defualt 5.')
    parser.add_argument('--engine', nargs='+', type=str, 
                        default=run_path+"/YOLOv5withTensorRT/build/best.engine", help='.engine path(s).')
    parser.add_argument('--library', nargs='+', type=str, 
                        default=run_path+"/YOLOv5withTensorRT/build/libmyplugins.so", help='libmyplugins.so path(s).')
    parser.add_argument('--color', nargs='+', type=int, default=1, help='Friend\'s color, 1 is blue, 2 is red.')
    parser.add_argument('--mode', nargs='+', type=str, default="debug", help='Running mode. debug (default) or release.')
    opt = parser.parse_args()
    run_mode = 1 if opt.mode == "debug" else 0
    ctypes.CDLL(opt.library)
    engine_file_path = opt.engine
    model = opt.model
    categories = categories7 if model == 7 else categories5

    if run_mode:
        print("Debug Mode.")
        print(f"Enginepath: {engine_file_path}")

    hCamera, pFrameBuffer = init_camera.get_buffer()              # 获取摄像头
    ser = get_ser("/dev/ttyTHS0", 115200, 0.0001)                 # 获取串口
    yolov5_wrapper = yolov5TRT.YoLov5TRT(engine_file_path, CONF_THRESH, IOU_THRESHOLD)        # 初始化YOLOv5运行API
    check_friend_wrapper = check_friends(ser, opt.color)          # 初始化友军检测类
    
    # 循环检测目标与发送信息
    while 1:
        try:
            begin = time.time() # 每帧总计时开始

            # 获取相机图像
            """
            windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
            linux下直接输出正的，不需要上下翻转

            此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            """
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
                                1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_LINEAR)


            result, image_raw = yolov5_wrapper.infer(frame)                      # 用YOLOv5检测目标
            result_boxes = boxes(*result)                                        # 将结果转化为boxes类
            result_boxes = check_friend_wrapper.get_enemy_info(result_boxes)     # 得到敌军的boxes信息
            trans_detect_data(ser, result_boxes, image_raw)                      # 发送检测结果

            end = time.time()          # 结束计时
            pre_time = (end - begin)   # 统计用时

            cv2.waitKey(1)
            if run_mode:
                cv2.imshow("result", image_raw)             # 显示图像输出
                print(f"Frame Time: {pre_time * 1000}ms")   # 输出用时

        except Exception as e:
            print(e)

