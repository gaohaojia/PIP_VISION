# PIP_VISION 2.0 （已弃用，停止维护）
[![State-of-the-art Shitcode](https://img.shields.io/static/v1?label=State-of-the-art&message=Shitcode&color=7B5804)](https://github.com/trekhleb/state-of-the-art-shitcode)\
PIP战队算法组深度学习自瞄方案2.0仓库。

## 前言
2.0方案重构的 main.py 文件的代码，可读性、调试性、拓展性和稳定性更强。\
使用多进程方式，提高算法对硬件的利用率。

## 使用方式
### PC (x86_64)
```bash
pip3 install -r requirements.txt
cd model/build
cmake -DCMAKE_CUDA_ARCHITECTURES=75 ..
make
./yolov5_det -s best.wts best.engine n
```
### Jetson NX (armv8)
```bash
pip3 install -r requirements.txt
cd model/build
cmake ..
make -j6
./yolov5_det -s best.wts best.engine n
```

## 当前进度
### 已完成
- 代码初步框架的搭建。
- opencv 摄像头的调用。
- TensorRT 加速模式。
- 迈德相机调用。
- 友军保护。
- 测距
- 电控通讯。
### 进行中
#### 高优先度
#### 中优先度
- 海康相机调用。
- 云台解算。
- 跟踪预测。
- 反陀螺。
#### 低优先度
- 无 TensorRT 加速模式。
- OpenVINO 加速模式。

