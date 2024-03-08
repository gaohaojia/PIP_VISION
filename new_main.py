import yaml
import time
import cv2
import numpy as np
import argparse
import os

import multiprocessing

from cam_conf import camera
import yolov5TRT

# 载入配置
def load_config():

    global config

    RUN_PATH = os.path.split(os.path.realpath(__file__))[0]

    with open("config.yml") as f:
        yml = yaml.load(f)

    parser = argparse.ArgumentParser()
    # parser.add_argument('--yolov', nargs='?', type=int, default=7, help='The engine version that will be used. Default 5.')
    parser.add_argument('--image', nargs='?', type=str, default=yml['image'], help='测试图片路径，默认不进行测试')
    parser.add_argument('--camera', nargs='?', type=str, default=yml['camera'], help='相机型号\nmv:迈德\n其他:调用opencv（默认）')
    parser.add_argument('--multiprocessing', nargs='?', type=bool, default=yml['multiprocessing'], help='是否启用多进程加速，默认启用')
    parser.add_argument('--tensorrt', nargs='?', type=bool, default=yml['tensorrt'], help='是否启用TensorRT加速，默认不启用')
    parser.add_argument('--frameW', nargs='?', type=bool, default=yml['frame_w'], help='图像处理前宽度')
    parser.add_argument('--frameH', nargs='?', type=bool, default=yml['frame_h'], help='图像处理前高度')
    parser.add_argument('--engine', nargs='?', type=str, 
                        default=RUN_PATH+yml['engine'], help='.engine path(s).')
    parser.add_argument('--library', nargs='?', type=str, 
                        default=RUN_PATH+yml['library'], help='libmyplugins.so path(s).')
    config = parser.parse_args()

# 图像获取进程
def get_frame_process(frame_queue: multiprocessing.Queue):
    
    if config.image != 'None':
        # 图片测试模式
        try:
            test_image = cv2.imread(config.image)
        except:
            print(f"[ERROR]没有找到图片‘{config.image}’！")

        while True:
            if frame_queue.full:
                frame_queue.get()
            frame_queue.put(test_image)

    
    elif config.camera == 'mv':
        # 迈德相机模式
        try:
            buffer = camera.buffer()
        except:
            print("[ERROR]未找到迈德相机！")

        while True:
            try:
                frame = buffer.get_frame()
                if frame_queue.full:
                    frame_queue.get()
                frame_queue.put(frame)
            except:
                print("[WARN]未获取到迈德相机图像！")
                time.sleep(0.001)


    else:
        # opencv 相机模式
        try:
            cap = cv2.VideoCapture(config.camera)
        except:
            print(f"[ERROR]没有找到摄像头‘{config.camera}’！")
            return
        
        while True:
            ret, frame = cap.read()
            if ret:
                if frame_queue.full:
                    frame_queue.get()
                frame_queue.put(frame)
            else:
                print("[WARN]未获取到相机图像！")
                time.sleep(0.001)

# 图片处理进程
def frame_processing_process(frame_queue: multiprocessing.Queue, 
                             processed_frame_queue: multiprocessing.Queue):

    while True:
        frame = frame_queue.get()
        frame = cv2.resize(frame, (config.frameW, config.frameH))
        if processed_frame_queue.full:
            processed_frame_queue.get()
        processed_frame_queue.put(frame)
    
# YOLO处理进程
def YOLO_process(processed_frame_queue: multiprocessing.Queue):
    
    if config.tensorrt:

        yolov5_wrapper = yolov5TRT.YoLov5TRT(config.engine, CONF_THRESH, IOU_THRESHOLD)
        while True:


# 主函数
def main():
    load_config()

    frame_queue = multiprocessing.Queue()
    processed_frame_queue = multiprocessing.Queue()

    get_frame_p = multiprocessing.Process(target=get_frame_process, args=(frame_queue, ))
    frame_processing_p = multiprocessing.Process(target=frame_processing_process, args=(frame_queue, processed_frame_queue, ))
    YOLO_p = multiprocessing.Process(target=YOLO_process, args=(processed_frame_queue, ))

if __name__ == "__main__":    
    main()