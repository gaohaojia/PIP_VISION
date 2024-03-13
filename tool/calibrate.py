import cv2
import numpy as np
import sys
import multiprocessing
import time
import yaml

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

def calculate_process(frame_queue: multiprocessing.Queue, end_pipe):
    objp = np.zeros((5 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:5, 0:8].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = 27.4 * objp   # 打印棋盘格一格的边长为27.4mm
    obj_points = []     # 存储3D点
    img_points = []     # 存储2D点
    
    cnt = 0 # 成功次数
    print("\r[WARN]Can't find the chessboard!", end='')
    while True:
        img = frame_queue.get()
        img = cv2.resize(img, (640, 480))
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
            print('\r[INFO]Progress:['+'#'*cnt + '-'*(30-cnt), end=']')
            if cnt >= 30: 
                print("")
                break
            time.sleep(0.5)
    _, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    
    # 内参数矩阵
    Camera_intrinsic = {"mtx": mtx,"dist": dist,}
    print(Camera_intrinsic)
    data = {}
    data['fx'] = float(Camera_intrinsic['mtx'][0][0])
    data['cx'] = float(Camera_intrinsic['mtx'][0][2])
    data['fy'] = float(Camera_intrinsic['mtx'][1][1])
    data['cy'] = float(Camera_intrinsic['mtx'][1][2])
    data['k1'] = float(Camera_intrinsic['dist'][0][0])
    data['k2'] = float(Camera_intrinsic['dist'][0][1])
    data['p1'] = float(Camera_intrinsic['dist'][0][2])
    data['p2'] = float(Camera_intrinsic['dist'][0][3])
    data['k3'] = float(Camera_intrinsic['dist'][0][4])
    with open('camera/config.yml', 'w', encoding='utf-8') as file:
        yaml.dump(data=data, stream=file, allow_unicode=True)
    end_pipe.send(1)

def main():
    frame_queue = multiprocessing.Queue(maxsize=1)
    end_pipe = multiprocessing.Pipe()
    process = [multiprocessing.Process(target=camera_process, args=(frame_queue, )),
               multiprocessing.Process(target=calculate_process, args=(frame_queue, end_pipe[1], )),]
    [p.start() for p in process]
    # [p.join() for p in process]

    end_pipe[0].recv()
    [p.terminate() for p in process]

if __name__ == "__main__":
    main()