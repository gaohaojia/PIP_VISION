import yaml
import time
import cv2
import numpy as np
import argparse
import os

from multiprocessing import Queue, Process

from cam_conf import camera
import yolov5TRT

# 用于存储boxes各种信息的类。
class BoxesWithFrame():
    
    def __init__(self, boxes, scores, classid, frame):
        """
        param:
            boxes:   boxes位置信息。
            scores:  boxes的置信度。
            classid: boxes的id。
        """
        self.boxes = boxes      
        self.scores = scores    
        self.classid = classid  
        self.frame = frame

# 载入配置
def load_config():

    global config, categories

    RUN_PATH = os.path.split(os.path.realpath(__file__))[0]

    try:
        with open("config.yml") as f:
            yml = yaml.full_load(f)
    except:
        print("[ERROR]配置文件缺失！")
        exit(0)

    parser = argparse.ArgumentParser()
    # parser.add_argument('--yolov', nargs='?', type=int, default=7, help='The engine version that will be used. Default 5.')
    parser.add_argument('--image', nargs='?', type=str, default=yml['image'], 
                        help='测试图片路径，默认不进行测试')
    parser.add_argument('--camera', nargs='?', type=str, default=yml['camera'], 
                        help='相机型号\nmv:迈德\n其他:调用opencv（默认）')
    parser.add_argument('--multiprocessing', nargs='?', type=bool, default=yml['multiprocessing'], 
                        help='是否启用多进程加速，默认True')
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
    config = parser.parse_args()

    # 类别
    categories = []

    print("[INFO]配置载入完成。")

# 图像获取进程
def get_frame_process(frame_queue: Queue):
    
    print("[INFO]图像获取进程启动。")

    if config.image != 'None' and not config.image is None:
        # 图片测试模式
        print("[INFO]开启图片测试模式。")

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
        print("[INFO]开启迈德相机模式。")

        try:
            buffer = camera.buffer()
            buffer.mvsdk_init()
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
                time.sleep(0.01)


    else:
        # opencv 相机模式
        print("[INFO]开启opencv相机模式。")

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
                time.sleep(0.01)

# 图像处理进程
def frame_processing_process(frame_queue: Queue, 
                             processed_frame_queue: Queue):
    
    print("[INFO]启动图像处理进程。")
    
    while True:
        frame = frame_queue.get()
        frame = cv2.resize(frame, (config.frameW, config.frameH))
        if processed_frame_queue.full:
            processed_frame_queue.get()
        processed_frame_queue.put(frame)
    
# YOLO处理进程
def yolo_process(processed_frame_queue: Queue, 
                 boxes_queue: Queue):
    
    print("[INFO]启动YOLO处理进程。")

    if config.tensorrt:
        print("[INFO]启动TensorRT加速。")
        try:
            yolo_wrapper = yolov5TRT.YoLov5TRT(config.engine, config.conf, config.iou)
        except Exception as e:
            print(f"[ERROR]TensorRT启动失败。\n{e}")
            exit(0)

        while True:

            frame = processed_frame_queue.get()
            result_boxes = BoxesWithFrame(*yolo_wrapper.infer(frame), frame)
            if boxes_queue.full:
                boxes_queue.get()
            boxes_queue.put(result_boxes)

# 计算绘制进程
def calculate_process(boxes_queue: Queue,
                      show_queue: Queue):
    
    print("[INFO]启动计算绘制进程。")
    while True:
        result_boxes: BoxesWithFrame = boxes_queue.get()
        for idx in range(len(result_boxes.boxes)):
            yolov5TRT.plot_one_box(result_boxes.boxes[idx], 
                                   result_boxes.frame, 
                                   [192,192,192],
                                   label=f"{categories[int(result_boxes.classid[idx])]}:{result_boxes.scores[idx]:.2f}")
            if show_queue.full:
                show_queue.get()
            show_queue.put(result_boxes.frame)


# 图像展示进程
def show_process(show_queue: Queue):

    if config.result:
        print("[INFO]启动图像展示进程。")

        while True:
            try:
                frame = show_queue.get()
                cv2.imshow("Result Window", frame)
                cv2.waitKey(1)
            except:
                print("[WARN]无法输出结果图像！")
                time.sleep(0.01)


# 主函数
def main():
    load_config()

    frame_queue = Queue(1)
    processed_frame_queue = Queue(1)
    boxes_queue = Queue(1)
    show_queue = Queue(1)

    process = [Process(target=get_frame_process, args=(frame_queue, )),
               Process(target=frame_processing_process, args=(frame_queue, processed_frame_queue, )),
               Process(target=yolo_process, args=(processed_frame_queue, boxes_queue, )),
               Process(target=calculate_process, args=(boxes_queue, show_queue, )),
               Process(target=show_process, args=(show_queue, ))]

    [p.start() for p in process]
    [p.join() for p in process]

if __name__ == "__main__":    
    main()