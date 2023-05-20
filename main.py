import ctypes
import time
import cv2
import numpy as np
import argparse
import serial

from cam_conf import init_camera
from cam_conf import mvsdk
from yolov5TRT import YoLov5TRT

pre_time = 0.1

def get_ser(port, baudrate, timeout):
    """
    description: Linux系统使用com1口连接串行口。
    """
    ser = serial.Serial(port, baudrate, timeout=timeout)
    print(ser.bytesize)
    print(ser.parity)
    print(ser.stopbits)
    return ser

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', nargs='+', type=str, default="/home/pip/Desktop/yolov5+tensorrt/yolov5/buildc/best.engine", help='.engine path(s)')
    parser.add_argument('--save', type=int, default=0, help='save?')
    opt = parser.parse_args()
    PLUGIN_LIBRARY = "/home/pip/Desktop/yolov5+tensorrt/yolov5/builds/libmyplugins.so"
    engine_file_path = opt.engine
    print(f'enginepath:{engine_file_path}')
    ctypes.CDLL(PLUGIN_LIBRARY)

    categories = ["armor1red", "armor3red", "armor4red", "armor5red", "armor1blue", "armor3blue", "armor4blue",
                  "armor5blue", "armor1grey", "armor3grey", "armor4grey", "armor5grey"]

    hCamera, pFrameBuffer = init_camera.get_buffer()
    ser = get_ser("/dev/ttyTHS0", 115200, 0.0001)
    yolov5_wrapper = YoLov5TRT(engine_file_path, ser, categories)
    datavount=0
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

            img, t = yolov5_wrapper.infer(frame, pre_time)
            end2 = time.time()
            end = time.time()
            pre_time = (end - begin)

            cv2.waitKey(1)
            cv2.imshow("result", img)
            print(f"frame time: {(end - begin) * 1000}ms\t\tYOLO time: {(end2 - begin2) * 1000}ms")
    except:
        print("ERROR!")
