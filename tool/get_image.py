import cv2
from camera import controller

try:
    buffer = controller.buffer()
    buffer.mvsdk_init()
except Exception as e:
    print(e)
    exit(0)

error_cnt = 0 # 错误次数
cnt = 0
while True:
    frame = buffer.get_frame()
    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == ord('j'):
        cnt += 1
        cv2.imwrite(f"save_images/{cnt}.jpg", frame)