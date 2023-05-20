"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import math
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision
import argparse
import cv2
import numpy
from sdk import mvsdk
import platform
import serial
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.1
check_fr=0
fr=[]
pre_x=0
pre_y=0
pre_time=0.1


def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
'''
def traned(self, data0):
    e = bin(data0)

    if len(e) < 18:

        ss = ''
        for i in range(18 - len(e)):
            ss = ss + '0'
        e = ss + e[2:]
        f = e[:8]
        h = e[8:]
        return [f, h]


'''

class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            print(f" f is {f}")
            engine = runtime.deserialize_cuda_engine(f.read())
            print(f'kewua{engine}')
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
    def traned(self,data0):
        b16s=(4 - len(hex(data0)[2:])) * '0' + hex(data0)[2:]
        #print(f'4545454545454545{b16s}')
        return [b16s[:2],b16s[2:]]
    def protectfr(self,all,friends):
        new=[]
        for i in all.numpy().tolist():
            if i not in (friends):
                new.append(i)
        return torch.tensor(new)
    def infer(self, input_image_path):
        start = time.time()
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        self.input_image_path = input_image_path
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])

        input_image, image_raw, origin_h, origin_w = self.preprocess_image(input_image_path
                                                                           )

        batch_origin_h.append(origin_h)
        batch_origin_w.append(origin_w)
        np.copyto(batch_input_image, input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())

        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        #end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        side1=time.time()
        print(f"side1\t{(side1-start)*1000:.3f}")
        result_boxes, result_scores, result_classid = self.post_process(
            output, origin_h, origin_w)

        #print(f"retboxes{result_boxes}")
        # Draw rectangles and labels on the original image
        try:
            ser.write(b'\x45')
        except:
            print("wrong open")#print(boxes[mindex])


        # er.write(b'\x64')
        # ser.write(hex(206).encode('utf-8'))

        color = 0
        friends=[]
        print(f'my recieve{ser.read()}')
        if ser.read() == b'\xff':
            color = 1  #blue
            friends=[0,1,2,3]

        elif ser.read() == b'\x00':
            color = 2   #red
            friends=[4,5,6,7]
        print(f"fr\t{friends}")
        global  check_fr,fr
        if check_fr==0 and len(friends)!=0:
            fr=friends
            check_fr=1
        friends_list=fr+[8,9,10,11]


        print(f"frdlist{friends_list}")
        exit_friends_boxes = []
        exit_friends_scores = []
        exit_friends_id = []
        for ii in range(len(result_classid)):
            if int(result_classid.numpy()[ii]) in friends_list:
                exit_friends_boxes.append(result_boxes[ii])
                exit_friends_scores.append(result_scores[ii])
                exit_friends_id.append(result_classid[ii])
        result_boxes = self.protectfr(result_boxes, exit_friends_boxes)
        result_scores = self.protectfr(result_scores, exit_friends_scores)
        result_classid = self.protectfr(result_classid, exit_friends_id)

        side2 = time.time()
        print(f"side2\t{(side2 - side1) * 1000:.3f}")
        #print(f"resultbox{result_boxes}")
        #print(f"resultscore{result_scores}")

        #print(f"resultclass{result_classid}")
        boxes = np.array(result_boxes)
        # print(f'boxes={boxes}')
        # print(boxes)
        inde = boxes.shape[0]

        # print(f"shape={boxes.shape[0]}")
        numlist = []
        # readata=ser.read
        for isb in range(inde):
            numlist.append(
                float(((boxes[isb][0] + boxes[isb][2]) / 2 - 320) ** 2 + ((boxes[isb][1] + boxes[isb][3]) - 240) ** 2))
        # print(numlist)
        # print(f"mindex={np.argmin(numlist)}")
        # print
        # if len(numlist)!=0:
        mindex = -1

        if len(numlist) == 0:
            mindex = -1
        else:
            mindex = np.argmin(numlist)
            #print(f"xxyy\t\t\t\t{int(boxes[mindex][0])}{int(boxes[mindex][1])}")

        try:
                #ser.write(b'\x45')
            global pre_x,pre_y,pre_time
            x_now=int((boxes[mindex][0]+boxes[mindex][2])/2)
            y_now=int((boxes[mindex][1]+boxes[mindex][3])/2)
            x_1, x_2 = self.traned((x_now))
            y_1, y_2 = self.traned((y_now))
            detax=x_now-pre_x
            detay=y_now-pre_y

            xx_1,xx_2=self.traned((int(pre_x+detax/2)))
            yy_1, yy_2 = self.traned((int(pre_y + detay / 2)))
            print("fgh")
            print(x_1, x_2,xx_1,xx_2)

            deta_dis=np.sqrt((detay**2+detax**2))
            speed_1,speed_2=self.traned(int(500*pre_time))
            pre_x=x_now
            pre_y=y_now


        except:
            print("wrong traning")
        side3 = time.time()
        print(f"side3\t{(side3 - side2) * 1000:.3f}")
        #print(numlist)
        tag_size = 0.05
        tag_size_half = 0.02
        half_Weight=[229/4,152/4]
        half_Height=[126/4,142/4]

        fx = 1056.4967111
        fy = 1056.6221413136
        cx = 657.4915775667
        cy = 508.2778608
        xxx=-0.392652606
        K = np.array([[fx, xxx, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float64)  # neican



        '''objPoints = np.array([[0, 0, 0],
                              [0, 52, 0],
                              [218, 0, 0],
                              [218, 52, 0]], dtype=np.float64)  # worldpoint'''
            #imgPoints = np.array([[608, 167], [514, 167], [518, 69], [611, 71]], dtype=np.float64)  # camerapoint
        cameraMatrix = K
        distCoeffs = None
        side4 = time.time()
        print(f"side4\t{(side4 - side3) * 1000:.3f}")
            #  print(box)
        if mindex != -1 :
            box = result_boxes[mindex]
            print(f"boxxx{box}")
            imgPoints = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], dtype=np.float64)
                #label = "{}:{:.2f}".format(categories[int(result_classid[mindex])], result_scores[mindex])
            if mindex%4==0:
                idn=0
            else:
                idn=1
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
                plot_one_box(box,image_raw,label="{}:{:.2f}".format(categories[int(result_classid[mindex])], result_scores[mindex]),)
            except:
                    '''666'''
            #print(categories)
                #label = "{}:{:.2f}".format(categories[int(result_classid[mindex])], result_scores[mindex])
                #print(f"label={label}")
                #ser.write((label.encode("gbk")))
                #ser.write((distance.encode("gbk")))
            rvec_matrix = cv2.Rodrigues(rvec)[0]
            proj_matrix = np.hstack((rvec_matrix, rvec))
            eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
            pitch, yaw, roll = str(int(eulerAngles[0])), str(int(eulerAngles[1])),str(int(eulerAngles[2]))
            distance=str(int(distance/10))
                #label=str(label)
            print(f'pryd{pitch},{yaw},{roll},{distance}')
                #ser.write((label.encode("UTF-8")))
                #ser.write('\t'.encode('gbk'))
            # dist_1, dist_2 = traned(int(distance))
            dis_1, dis_2 = self.traned((int(distance)))
        zero11,zero2=self.traned(0)
            # e = bin(int(240-boxes[mindex][1]) )

                #pi_1, pi_2 = self.traned((int(pitch)+180))

        try:
            #ser.write(b'\x45')
            ser.write(bytes.fromhex(dis_1))  # x-mid
            ser.write(bytes.fromhex(dis_2))
            ser.write(bytes.fromhex(xx_1))  # x-mid
            ser.write(bytes.fromhex(xx_2))
            ser.write(bytes.fromhex(yy_1))  # x-mid
            ser.write(bytes.fromhex(yy_2))
            ser.write(bytes.fromhex(speed_1))  # x-mid
            ser.write(bytes.fromhex(speed_2))

            # print(f"distance{bytes.fromhex(dis_1)}{bytes.fromhex(dis_2)}")
        except:
            #ser.write(b'\x45')
            ser.write(bytes.fromhex(zero11))  # x-mid
            ser.write(bytes.fromhex(zero11))
            ser.write(bytes.fromhex(zero11))  # x-mid
            ser.write(bytes.fromhex(zero11))
            ser.write(bytes.fromhex(zero11))  # x-mid
            ser.write(bytes.fromhex(zero11))
            ser.write(bytes.fromhex(zero11))  # x-mid
            ser.write(bytes.fromhex(zero11))
        ser.write(b'\x45')
        try:
            #ser.write(b'\x45')
            ser.write(bytes.fromhex(dis_1))  # x-mid
            ser.write(bytes.fromhex(dis_2))
            ser.write(bytes.fromhex(x_1))  # x-mid
            ser.write(bytes.fromhex(x_2))
            ser.write(bytes.fromhex(y_1))  # x-mid
            ser.write(bytes.fromhex(y_2))
            ser.write(bytes.fromhex(speed_1))  # x-mid
            ser.write(bytes.fromhex(speed_2))
        except:
            #ser.write(b'\x45')
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
        return image_raw, end - start


    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)

    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, input_image_path):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = input_image_path
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # to a torch Tensor
        pred = torch.Tensor(pred).cuda()
        # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4]
        # Get the classid
        classid = pred[:, 5]
        # Choose those boxes that score > CONF_THRESH
        si = scores > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        # Do nms
        indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()
        return result_boxes, result_scores, result_classid


class inferThread(threading.Thread):
    def __init__(self, yolov5_wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        #self.image_path_batch = image_path_batch
    def infer(self, frame):

        batch_image_raw, use_time = self.yolov5_wrapper.infer(frame)

       # for i, img_path in enumerate(self.image_path_batch):
      #      parent, filename = os.path.split(img_path)
       #     save_name = os.path.join('output', filename)
            # Save image
          #  cv2.imwrite(save_name, batch_image_raw[i])
      #  print('input->{}, time->{:.2f}ms, saving into output/'.format(self.image_path_batch, use_time * 1000))

        return batch_image_raw, use_time


class warmUpThread(threading.Thread):
    def __init__(self, yolov5_wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper

    def run(self):
        batch_image_raw, use_time = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image_zeros())
        print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))


if __name__ == "__main__":
    # load custom plugins
    ser = serial.Serial("/dev/ttyTHS0", 115200, timeout=0.0001)  # Linux系统使用com1口连接串行口
    print(ser.bytesize)
    print(ser.parity)
    print(ser.stopbits)
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        sys.exit()


    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    i = 0 if nDev == 1 else int(input("Select camera: "))
    DevInfo = DevList[i]
    print(DevInfo)

    # 打开相机
    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("CameraInit Failed({}): {}".format(e.error_code, e.message))


    # 获取相机特性描述
    cap = mvsdk.CameraGetCapability(hCamera)

    # 判断是黑白相机还是彩色相机
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    # 相机模式切换成连续采集
    mvsdk.CameraSetTriggerMode(hCamera, 0)

    # 手动曝光，曝光时间30ms
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)

    # 让SDK内部取图线程开始工作
    mvsdk.CameraPlay(hCamera)

    # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
    FrameBufferSize = cap.sResolutionRange.iWidthMax* cap.sResolutionRange.iHeightMax *3

    # 分配RGB buffer，用来存放ISP输出的图像
    # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', nargs='+', type=str, default="/home/pip/Desktop/yolov5+tensorrt/yolov5/buildc/best.engine", help='.engine path(s)')
    parser.add_argument('--save', type=int, default=0, help='save?')
    opt = parser.parse_args()
    PLUGIN_LIBRARY = "/home/pip/Desktop/yolov5+tensorrt/yolov5/builds/libmyplugins.so"
    engine_file_path = opt.engine
    print(f'enginepath:{engine_file_path}')
    ctypes.CDLL(PLUGIN_LIBRARY)

    # load coco labels
    '''
    categories=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']
            '''

    categories = ["armor1red", "armor3red", "armor4red", "armor5red", "armor1blue", "armor3blue", "armor4blue",
                  "armor5blue", "armor1grey", "armor3grey", "armor4grey", "armor5grey"]

    yolov5_wrapper = YoLov5TRT(engine_file_path)
    #cap = cv2.VideoCapture(0)
    datavount=0
    try:
        #thread1 = inferThread(yolov5_wrapper)
        #thread1.start()
        #thread1.join()
        while 1:
            begin = time.time()
            #global pre_time
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

            img, t = yolov5_wrapper.infer(frame)
            end2=time.time()
            end = time.time()

            pre_time = (end - begin)
            cv2.waitKey(1)
            #print("mid")
            cv2.imshow("result", img)


            end=time.time()
            
	 
            #print(f"time is{end*1000-begin*1000}ms\t\t{end2*1000-begin2*1000}ms")
    except:
        'g'
