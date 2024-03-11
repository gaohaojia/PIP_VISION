import ctypes
import yaml
import time
import cv2
import numpy as np
import argparse
import os

from multiprocessing import Queue, Process, set_start_method, Pipe

from camera import controller
import yolov5TRT

# 用于存储boxes各种信息的类。
class Boxes():
    
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

# 输出info信息
def print_info(info: str):
    print(f"[INFO]{info}")

# 输出warn信息
def print_warn(warn: str):
    print(f"\033[33m[WARN]{warn}\033[0m")

# 输出error信息
def print_error(error: str):
    print(f"\033[31m[ERROR]{error}\033[0m")

# 载入配置
def load_config():

    global config, categories

    RUN_PATH = os.path.split(os.path.realpath(__file__))[0]

    try:
        with open("config.yml") as f:
            yml = yaml.full_load(f)
    except:
        print_error("配置文件缺失！")
        exit(0)

    parser = argparse.ArgumentParser()
    # parser.add_argument('--yolov', nargs='?', type=int, default=7, help='The engine version that will be used. Default 5.')
    parser.add_argument('--image', nargs='?', type=str, default=yml['image'], 
                        help='测试图片路径，默认不进行测试')
    parser.add_argument('--camera', nargs='?', type=str, default=yml['camera'], 
                        help='相机型号\nmv:迈德\n其他:调用opencv（默认）')
    # parser.add_argument('--multiprocessing', nargs='?', type=bool, default=yml['multiprocessing'], 
    #                     help='是否启用多进程加速，默认True')
    parser.add_argument('--tensorrt', nargs='?', type=bool, default=yml['tensorrt'], 
                        help='是否启用TensorRT加速，默认False')
    parser.add_argument('--frameW', nargs='?', type=int, default=yml['frame_w'], 
                        help='图像处理前宽度')
    parser.add_argument('--frameH', nargs='?', type=int, default=yml['frame_h'], 
                        help='图像处理前高度')
    parser.add_argument('--engine', nargs='?', type=str, default=RUN_PATH+yml['engine'], 
                        help='.engine path(s).')
    parser.add_argument('--library', nargs='?', type=str, default=RUN_PATH+yml['library'], 
                        help='libmyplugins.so path(s).')
    parser.add_argument('--conf', nargs='?', type=float, default=yml['conf_thresh'], 
                        help='置信度，默认0.5')
    parser.add_argument('--iou', nargs='?', type=float, default=yml['iou_thresh'], 
                        help='交并比，默认0.5')
    parser.add_argument('--result', nargs='?', type=bool, default=yml['show_result'], 
                        help='是否展示结果图片，默认True')
    parser.add_argument('--color', nargs='?', type=str, default=yml['color'], 
                        help='友军颜色\nred:红（默认）\nblue:蓝')
    config = parser.parse_args()

    # 类别
    categories = yml['categories']

    print_info("配置载入完成。")

# 图像处理函数
def frame_processing(config, frame):
    frame = cv2.resize(frame, (config.frameW, config.frameH))
    return frame

# 图像获取进程
def get_frame_process(config, 
                      frame_pipe):
    
    print_info("图像获取进程启动。")

    if config.image != 'None' and not config.image is None:
        # 图片测试模式
        print_info("开启图片测试模式。")

        try:
            test_image = cv2.imread(config.image)
        except:
            print_error(f"没有找到图片‘{config.image}’！")
            exit(0)
        
        test_image = frame_processing(config, test_image)

        while True:
            frame_pipe.send(test_image)

    
    elif config.camera == 'mv':
        # 迈德相机模式
        print_info("开启迈德相机模式。")

        try:
            buffer = controller.buffer()
            buffer.mvsdk_init()
        except Exception as e:
            print_error("未找到迈德相机！")
            exit(0)

        error_cnt = 0 # 错误次数

        while True:
            try:
                frame = buffer.get_frame()
                frame = frame_processing(config, frame)
                frame_pipe.send(frame)
            except:
                error_cnt += 1
                print_warn("[{error_cnt}]未获取到迈德相机图像！")
                if error_cnt >= 10:
                    print_error("未获取到迈德相机图像！")
                    exit(0)
                time.sleep(0.1)


    else:
        # opencv 相机模式
        print_info("开启opencv相机模式。")

        try:
            cap = cv2.VideoCapture(config.camera)
        except:
            print_error(f"没有找到摄像头‘{config.camera}’！")
            exit(0)
        
        if cap.isOpened:
            print_info(f"获取到相机{config.camera}")
        else:
            print_error(f"没有找到摄像头‘{config.camera}’！")
            exit(0)

        error_cnt = 0 # 错误次数
        
        while True:
            ret, frame = cap.read()
            if ret:
                frame = frame_processing(config, frame)
                frame_pipe.send(frame)
            else:
                error_cnt += 1
                print_warn(f"[{error_cnt}]未获取到相机图像！")
                if error_cnt >= 10:
                    print_error("未获取到相机图像！")
                    exit(0)
                time.sleep(0.1)
    
# YOLO处理进程
def yolo_process(config,
                 frame_pipe, 
                 boxes_pipe,
                 processed_pipe,
                 show_pipe):
    
    print_info("启动YOLO处理进程。")

    if config.tensorrt:
        # TensorRT 加速模式
        print_info("启动TensorRT加速模式。")

        try:
            ctypes.CDLL(config.library)   
            yolo_wrapper = yolov5TRT.YoLov5TRT(config.engine, config.conf, config.iou)
        except Exception as e:
            print_error("TensorRT启动失败。")
            exit(0)

        while True:
            frame = frame_pipe.recv()
            result_boxes = Boxes(*yolo_wrapper.infer(frame))
            boxes_pipe.send(result_boxes)
            if config.result:
                processed_pipe.send(frame)
    

    else:
        # 直出模式
        print_info("启动直出模式。")

        while True:
            frame = frame_pipe.recv()
            show_pipe.send(frame)


# 计算绘制进程
def calculate_process(config,
                      categories,
                      boxes_pipe,
                      processed_pipe,
                      result_pipe):
    
    print_info("启动计算绘制进程。")
    while True:
        start_time = time.time()
        result_boxes: Boxes = boxes_pipe.recv()
        end_time = time.time()
        if config.result:
            frame = processed_pipe.recv()
            for idx in range(len(result_boxes.boxes)):
                yolov5TRT.plot_one_box(result_boxes.boxes[idx], 
                                       frame, 
                                       [192,192,192],
                                       label=f"{categories[int(result_boxes.classid[idx])]}:{result_boxes.scores[idx]:.2f}")
            result_pipe.send(frame)
        
        else:
            print(f"\r[INFO]FPS: {1 / (end_time - start_time):.2f}, 类别: {[categories[int(classid)] for classid in result_boxes.classid]}"+' '*10, end="")


# 结果展示进程
def result_process(config,
                   result_pipe):

    print_info("启动图像展示进程。")
    
    error_cnt = 0 # 错误次数
    
    while True:
        try:
            start_time = time.time()
            frame = result_pipe.recv()
            end_time = time.time()
            cv2.putText(frame, f"FPS: {1 / (end_time - start_time):.2f}", 
                        (5,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
            cv2.imshow("Result Window", frame)
            cv2.waitKey(1)
        except:
            error_cnt += 1
            print_warn(f"[{error_cnt}]无法输出结果图像！")
            if error_cnt >= 10:
                print_error("无法输出结果图像！")
                exit(0)
            time.sleep(0.1)


# 主函数
def main():
    load_config()

    set_start_method('spawn')

    frame_pipe = Pipe(duplex=False)
    boxes_pipe = Pipe(duplex=False)
    processed_pipe = Pipe(duplex=False)
    result_pipe = Pipe(duplex=False)

    process = [Process(target=get_frame_process, args=(config, frame_pipe[1], )),
               Process(target=yolo_process, args=(config, frame_pipe[0], boxes_pipe[1], processed_pipe[1], result_pipe[1], )),
               Process(target=calculate_process, args=(config, categories, boxes_pipe[0], processed_pipe[0], result_pipe[1], ))]

    if config.result:
        process.append(Process(target=result_process, args=(config, result_pipe[0], )))

    [p.start() for p in process]
    [p.join() for p in process]

if __name__ == "__main__":    
    main()