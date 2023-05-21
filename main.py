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

pre_time = 0.1 # 每帧所需时间
run_path = os.path.split(os.path.realpath(__file__))[0] # 运行目录

# 标签列表
categories = ["armor1red", "armor3red", "armor4red", "armor5red",           
              "armor1blue", "armor3blue", "armor4blue",
              "armor5blue", "armor1grey", "armor3grey", "armor4grey", "armor5grey"]

def get_ser(port, baudrate, timeout):
    """
    description: Linux系统使用com1口连接串行口，在代码125行左右用于获得串行口传入tensorRT
    """
    ser = serial.Serial(port, baudrate, timeout=timeout)
    print(ser.bytesize)
    print(ser.parity)
    print(ser.stopbits)
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

def trans_detect_data(ser, result_boxes, result_scores, result_classid, image_raw):
    """
    description: 将detect的信息与电控通讯。
    param:
        ser:             串口信息。
        result_boxes:    识别的boxes。
        result_scores:   boxes的置信度。
        result_classid:  boxes的id。
    """
    # 计算处理时间
    side2 = time.time()
    print("check0")
    boxes = np.array(result_boxes)
    inde = boxes.shape[0]
    numlist = []

    # readata=ser.read
    # 计算谁离中心近
    for isb in range(inde):
        numlist.append(
            float(((boxes[isb][0] + boxes[isb][2]) / 2 - 320) ** 2 + ((boxes[isb][1] + boxes[isb][3]) - 240) ** 2))
    mindex = -1
    print("check2")
    if len(numlist) == 0:
        mindex = -1
    else:
        mindex = np.argmin(numlist)

    try:
        global pre_x, pre_y
        x_now = int((boxes[mindex][0] + boxes[mindex][2]) / 2)
        y_now = int((boxes[mindex][1] + boxes[mindex][3]) / 2)
        x_1, x_2 = get_transdata_from_10b((x_now))
        y_1, y_2 = get_transdata_from_10b((y_now))
        detax = x_now - pre_x
        detay = y_now - pre_y
        xx_1, xx_2 = get_transdata_from_10b((int(pre_x + detax / 2)))
        yy_1, yy_2 = get_transdata_from_10b((int(pre_y + detay / 2)))
        print(x_1, x_2, xx_1, xx_2)
        deta_dis = np.sqrt((detay ** 2 + detax ** 2))
        speed_1, speed_2 = get_transdata_from_10b(int(500 * pre_time))
        pre_x = x_now
        pre_y = y_now
    except:
        print("wrong trans")

    side3 = time.time()
    print(f"side3\t{(side3 - side2) * 1000:.3f}")
    # print(numlist)
    tag_size = 0.05
    tag_size_half = 0.02
    half_Weight = [229 / 4, 152 / 4]
    half_Height = [126 / 4, 142 / 4]

    fx = 1056.4967111
    fy = 1056.6221413136
    cx = 657.4915775667
    cy = 508.2778608
    xxx = -0.392652606
    K = np.array([[fx, xxx, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)  # neican

    '''objPoints = np.array([[0, 0, 0],
                          [0, 52, 0],
                          [218, 0, 0],
                          [218, 52, 0]], dtype=np.float64)  # worldpoint'''
    # imgPoints = np.array([[608, 167], [514, 167], [518, 69], [611,71]], dtype=np.float64)  # camerapoint
    cameraMatrix = K
    distCoeffs = None
    side4 = time.time()
    print(f"side4\t{(side4 - side3) * 1000:.3f}")
    #  print(box)
    if mindex != -1:
        box = result_boxes[mindex]

        imgPoints = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
                             dtype=np.float64)
        # label = "{}:{:.2f}".format(categories[int(result_classid[mindex])], result_scores[mindex])
        if mindex % 4 == 0:
            idn = 0
        else:
            idn = 1
        objPoints = np.array([[-half_Weight[idn], -half_Height[idn], 0],
                              [half_Weight[idn], -half_Height[idn], 0],
                              [half_Weight[idn], half_Height[idn], 0],
                              [-half_Weight[idn], half_Height[idn], 0]], dtype=np.float64)
        retval, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
        # print(f'hahaha{retval}{rvec}{tvec}')
        rotM = cv2.Rodrigues(rvec)[0]
        # position = -np.matrix(rotM).T * np.matrix(tvec)
        distance = np.linalg.norm(tvec)

        try:
            yolov5TRT.plot_one_box(box, image_raw,
                         label="{}:{:.2f}".format(categories[int(result_classid[mindex])], result_scores[mindex]), )
        except:
            '''g'''

        rvec_matrix = cv2.Rodrigues(rvec)[0]
        proj_matrix = np.hstack((rvec_matrix, rvec))
        eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
        pitch, yaw, roll = str(int(eulerAngles[0])), str(int(eulerAngles[1])), str(int(eulerAngles[2]))
        distance = str(int(distance / 10))
        # label=str(label)
        print(f'pryd{pitch},{yaw},{roll},{distance}')

        dis_1, dis_2 = get_transdata_from_10b((int(distance)))
    zero11, zero2 = get_transdata_from_10b(0)
    # e = bin(int(240-boxes[mindex][1]) )

    ser.write(b'\x45')
    try:
        # ser.write(b'\x45')
        ser.write(bytes.fromhex(dis_1))  # x-mid
        ser.write(bytes.fromhex(dis_2))
        ser.write(bytes.fromhex(x_1))  # x-mid
        ser.write(bytes.fromhex(x_2))
        ser.write(bytes.fromhex(y_1))  # x-mid
        ser.write(bytes.fromhex(y_2))
        ser.write(bytes.fromhex(speed_1))  # x-mid
        ser.write(bytes.fromhex(speed_2))
    except:
        # ser.write(b'\x45')
        ser.write(bytes.fromhex(zero11))  # x-mid
        ser.write(bytes.fromhex(zero11))
        ser.write(bytes.fromhex(zero11))  # x-mid
        ser.write(bytes.fromhex(zero11))
        ser.write(bytes.fromhex(zero11))  # x-mid
        ser.write(bytes.fromhex(zero11))
        ser.write(bytes.fromhex(zero11))  # x-mid
        ser.write(bytes.fromhex(zero11))
    side5 = time.time()
    print(f"side5\t{(side5 - side4) * 1000:.3f}")
    end = time.time()

class check_friends():
    """
    description: 检测友军
    """
    def __init__(self, ser):
        """
        description: 初始化变量以及尝试获取友军信息。
        param:
            ser:    串口信息。
        """
        self.color = 0
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
            print("wrong open")

        
        # TODO:还需要添加风车的编号，已经打过的风车和灰色风车都标记为友军
        # TODO:目前的想法：1.红蓝色已击打看作一个标签进行训练识别   2.红蓝色已击打分为两类训练识别
        print(f'my recieve{ser.read()}')
        # 根据我方式红蓝方的设定，进行友军识别
        if ser.read() == b'\xff':
            self.color = 1  # blue
            self.friends = [0, 1, 2, 3]

        elif ser.read() == b'\xaa':
            self.color = 2  # red
            self.friends = [4, 5, 6, 7]
        print(f"fr\t{self.friends}")
        # 函数/类内调用全局变量
        # 如果是友军而且友军列表成功添加，那么友军标记边变1，并且友军列表添加死亡的敌人
        fr = []
        if self.check_fr == 0 and len(self.friends) != 0:
            fr = self.friends
            self.check_fr = 1
        self.friends_list = fr + [8, 9, 10, 11]

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

    def get_enemy_info(self, result_boxes, result_scores, result_classid):
        """
        description: 处理识别的box，输出敌军的box信息。
        param:
            result_boxes:    boxes。
            result_scores:   所有boxes的置信度。
            result_classid:  所有boxes的id信息。
        return:
            敌军的boxes、置信度、id信息。
        """
        # 分别代表友军的box、box置信度、box的id
        exit_friends_boxes = []
        exit_friends_scores = []
        exit_friends_id = []
        friends_id = []
        for ii in range(len(result_classid)):
            if int(result_classid.numpy()[ii]) in self.friends_list:
                friends_id.append(int(result_classid.numpy()[ii]))
                exit_friends_boxes.append(result_boxes[ii])
                exit_friends_scores.append(result_scores[ii])
                exit_friends_id.append(result_classid[ii])
        print(f"id{friends_id}")
        enemy_list_index = []
        print(f"idnumpy{result_classid.numpy()}")
        # 获取敌军的列表以及id
        try:
            for i in result_classid.numpy():
                print(i)
                if int(i) not in friends_id:
                    dex_tem = ((np.where(result_classid.numpy() == i))[0][0])
                    enemy_list_index.append(dex_tem)
        except:
            "g"
        ourbox = []
        # TODO 以下两个变量没有用到
        ourclassid = []
        ourscore = []
        print(f"idene{enemy_list_index}")
        for dex in enemy_list_index:
            ourbox.append(result_boxes[dex].numpy())

        result_boxes = ourbox

        # result_boxes = self.get_nonfriend_from_all(result_boxes, exit_friends_boxes)
        # 置信度处理
        result_scores = self.get_nonfriend_from_all(result_scores, exit_friends_scores)
        # id处理
        result_classid = self.get_nonfriend_from_all(result_classid, exit_friends_id)
        # 从而获取到处理完毕的友方敌方box
        print(f"nowboxes{result_boxes}")
        print(f"nowscore{result_scores}")
        print(f"nowid{result_classid}")
        return result_boxes, result_scores, result_classid

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', nargs='+', type=str, default=run_path+"/YOLOv5withTensorRT/build/best.engine", help='.engine path(s)')
    parser.add_argument('--library', nargs='+', type=str, default=run_path+"/YOLOv5withTensorRT/build/libmyplugins.so", help='libmyplugins.so path(s)')
    parser.add_argument('--save', type=int, default=0, help='save?')
    opt = parser.parse_args()
    PLUGIN_LIBRARY = opt.library
    engine_file_path = opt.engine
    print(f'enginepath:{engine_file_path}')
    ctypes.CDLL(PLUGIN_LIBRARY)

    hCamera, pFrameBuffer = init_camera.get_buffer()
    ser = get_ser("/dev/ttyTHS0", 115200, 0.0001)
    yolov5_wrapper = yolov5TRT.YoLov5TRT(engine_file_path)
    check_friend_wrapper = check_friends(ser)
    
    try:
        while 1:
            begin = time.time()
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
            # linux下直接输出正的，不需要上下翻转

            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
                                   1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

            # cv2.Rodrigues()

            frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_LINEAR)
            #_, frame = frame.read()
            begin2 = time.time()

            result_boxes, result_scores, result_classid, image_raw = yolov5_wrapper.infer(frame)
            result_boxes, result_scores, result_classid = check_friend_wrapper.get_enemy_info(result_boxes, result_scores, result_classid)
            trans_detect_data(ser, result_boxes, result_scores, result_classid, image_raw)

            end2 = time.time()
            end = time.time()
            pre_time = (end - begin)

            cv2.waitKey(1)
            cv2.imshow("result", image_raw)
            print(f"frame time: {(end - begin) * 1000}ms\t\tYOLO time: {(end2 - begin2) * 1000}ms")
    except Exception as e:
        print("ERROR! Run while ERROR!\n" + e)
