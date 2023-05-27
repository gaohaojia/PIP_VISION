# Todo:
## 功能
- 完成电控通讯模块（可随时接收电控信号）
- 完成模型切换模块（分离风车预测与装甲板预测模型）
- 实现卡尔曼滤波
## Bug
- 当输入图像中存在多个装甲板时，大概率出现报错“The truth value of an array with more than one element is ambigous. Use a.any() or a.all()”，但对运行影响不大。