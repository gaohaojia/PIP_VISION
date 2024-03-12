"""
该文件被main.py调用，借助官方sdk初始化摄像头。
"""
import numpy as np
import yaml

class buffer():
    def __init__(self):

        with open("camera/config.yml") as f:
            yml = yaml.full_load(f)
        
        # 相机矩阵
        self.camera_matrix = np.float32([
            [yml['fx'], 0,         yml['cx']],
            [0,         yml['fy'], yml['cy']],
            [0,         0,         1]
        ])

        # 畸变矩阵
        self.camera_dis = np.float32([
            yml['k1'], yml['k2'], yml['p1'], yml['p2'], yml['k3']
        ])

    def mvsdk_init(self):
        self._buffer = _MVBuffer()
    
    def get_frame(self):
        return self._buffer.get_frame()
    
    def set_once_wb(self):
        self._buffer.set_once_wb()
    
class _MVBuffer():
    
    def __init__(self):

        from camera import mvsdk

        self.mvsdk = mvsdk

        DevList = self.mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        assert nDev > 0, "No camera was found!"

        DevInfo = DevList[0]

        # 打开相机
        self.hCamera = 0
        try:
            self.hCamera = self.mvsdk.CameraInit(DevInfo, -1, -1)
        except self.mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message))

        # 获取相机特性描述
        cap = self.mvsdk.CameraGetCapability(self.hCamera)

        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if monoCamera:
            self.mvsdk.CameraSetIspOutFormat(self.hCamera, self.mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            self.mvsdk.CameraSetIspOutFormat(self.hCamera, self.mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # 相机模式切换成连续采集
        self.mvsdk.CameraSetTriggerMode(self.hCamera, 0)

        # 手动曝光，曝光时间10ms
        self.mvsdk.CameraSetAeState(self.hCamera, 0)
        self.mvsdk.CameraSetExposureTime(self.hCamera, 10 * 1000)

        # 设置为手动白平衡并进行一次白平衡
        self.mvsdk.CameraSetWbMode(self.hCamera, False)
        self.mvsdk.CameraSetOnceWB(self.hCamera)

        # 让SDK内部取图线程开始工作
        self.mvsdk.CameraPlay(self.hCamera)

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        FrameBufferSize = cap.sResolutionRange.iWidthMax* cap.sResolutionRange.iHeightMax *3

        # 分配RGB buffer，用来存放ISP输出的图像
        # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
        self.pFrameBuffer = self.mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
    
    def get_frame(self):
        """
        windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
        linux下直接输出正的，不需要上下翻转

        此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
        把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
        """
        pRawData, FrameHead = self.mvsdk.CameraGetImageBuffer(self.hCamera, 200)
        self.mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
        self.mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

        frame_data = (self.mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
                            1 if FrameHead.uiMediaType == self.mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
        return frame
    
    # 进行一次白平衡
    def set_once_wb(self):
        self.mvsdk.CameraSetOnceWB(self.hCamera)