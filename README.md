# PIP_VISION
PIP战队视觉组源码仓库。
## 前言
在这里推荐一个学习Git的[网站](https://www.liaoxuefeng.com/wiki/896043488029600)，不熟悉Git的同学可以学习了解，对咱们PIP战队和自己未来做大项目都有帮助。
## 注意事项
- 每次开始写代码之前，**一定要pull一下**，看看代码有没有更新，有更新的话要及时合并（merge），免得在老版本上写了很多代码。
- 在开始实现一个新功能的时候，**请新建立一个分支（Branch）**，每次干一定量的工作后记得及时commit和push。
- 完成一个新功能后，先测试新功能是否有bug，之后合并到dev分支下，**不要直接合并到main分支下**，待整个dev测试完成后，再将dev整体合并到main分支。
## 其他
- 想看学习报告和踩坑日志的可以从右上角切换到相应的分支（Branch）。

![示例图片](https://github.com/gaohaojia/PIP_VISION/blob/images/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87.png)
## 一起为国一奋斗！！！
## 版本修改
### 1.1.0
- 建立cam_conf文件夹，保存相机相关代码，并将mvsdk.py放入cam_conf文件夹内。
- 将detect.py文件拆分，拆为main.py、init_camera.py、yolov5TRT.py。将init_camera放入cam_conf文件夹下。
- 修改pyd.sh文件，使其可以自动获取文件目录。