import cv2
import numpy as np
import sys
import multiprocessing
import time

sys.path.insert(0, sys.path[0]+"/../")
from camera import controller

def camera_process(frame_queue: multiprocessing.Queue):
    try:
        buffer = controller.buffer()
        buffer.mvsdk_init()
    except Exception as e:
        print(e)
        exit(0)

    while True:
        frame = buffer.get_frame()
        cv2.imshow("Result", frame)
        if frame_queue.empty():
            frame_queue.put(frame)
        if cv2.waitKey(1) & 0xFF == ord('b'):
            buffer.set_once_wb()

def calculate_process(frame_queue: multiprocessing.Queue):
    objp = np.zeros((5 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:5, 0:8].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = 2.74 * objp   # 打印棋盘格一格的边长为2.74cm
    obj_points = []     # 存储3D点
    img_points = []     # 存储2D点
    
    cnt = 0 # 成功次数
    while True:
        img = frame_queue.get()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,gray = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (5, 8), None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))  
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
            cv2.drawChessboardCorners(img, (5, 8), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
            cv2.waitKey(1)
            cnt += 1
            print(cnt)
            if cnt >= 30: 
                break
            time.sleep(0.5)
    _, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    
    # 内参数矩阵
    Camera_intrinsic = {"mtx": mtx,"dist": dist,}
    print(Camera_intrinsic)

def main():
    frame_queue = multiprocessing.Queue(maxsize=1)
    process = [multiprocessing.Process(target=camera_process, args=(frame_queue, )),
               multiprocessing.Process(target=calculate_process, args=(frame_queue, )),]
    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == "__main__":
    main()