# 有关colab的使用
测试使用的是yolov7，权重为yolov7-tiny.pt进行的的测试
（yolov8搞了一下还没弄）

## 数据集准备
目前已有数据集是在本地用labelimg标注的
以后可以改成roboflow进行共同的标注
使用roboflow进行标注和数据处理（需要梯子）
可以自己先注册看一下处理一下数据集啥的

 1. 具体的话就是新建个项目，如果有xml文件就一起传上去这样他自己就显示标签了如果没有可以标注（这部分还没有试）
 2. 然后就是准备导出，可以按比例划分数据集，可以选择一下预处理和数据增广，最后在准备生成的时候选择你要训练的网络版本（目前yolov8和之前的5，7都支持）



这部分可以等学长决定一下用不用

## colab的debug和一些准备

### 修改的点

 1. data.yaml的文件路径是绝对路径
 2. 修改train.py的default参数
这里面有个cfg的参数在cfg的文件中找到training文件里面对应权重的yaml文件路径
 3. 修改yolov7-main/utils/google_utils.py中
 >file = Path(str(file).strip().replace("'", '').lower())
 >替换成
 >file = Path(str(file).strip().replace("'", ''))
4. 安装谷歌云盘的代码放在这个笔记本的开头连接云盘
> """首要的选个好梯子梯子太差经常断
第一次会要授权一下
有时候在云盘更改文件之后会看不到更改，可以多等一会或者重新连接一下或者新建一个笔记本"""
import os
from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/My Drive"
os.chdir(path)
os.listdir(path)
5. 虽然这个环境不需要自己创建环境colab已经安装了一些运行但是有一些东西是不满足yolo的要求需要自己去安装
>%cd /content/drive/MyDrive/colab-yolov7/yolov7-main
!pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ #进入YOLO文件夹之后安装依赖（如果已经安装好的符合就不用了）
6. 修改笔记本设置到gpu加速
>!nvidia-smi #查看分配的显卡，一般是特斯拉t4也可能是k80，根据自己用的程度（看脸）

不用的时候最好断开不然12小时就自动断了一段时间以内可能没法用好点的显卡
如果是自己写的神经网络应该可以直接用

7.运行程序
>!python ../XXX.py   #(最好是绝对路径)
### 注意事项

 1. 所有的文件路径最好都是绝对路径
 2. 尽量简介一下文件的层级，最好不使用相对路径，最好放在同一个文件夹下
 3. 最好不使用!python的语句传入参数可能会有问题修改train.py里面的default可以避免麻烦（这样就是失去灵活性了每次训练需要改一下）
 4. 如果手头宽松的话可以开个colab pro中终端，就可以用yolo命令训练就没这个麻烦了

### 常见错误

 1. File "/home/**/yolov7-main/utils/google_utils.py", line 26, in attempt_download assets = [x['name'] for x in response['assets']] # release assets KeyError: 'assets'
修改yolov7-main/utils/google_utils.py中
 >file = Path(str(file).strip().replace("'", '').lower())
 >替换成
 >file = Path(str(file).strip().replace("'", ''))
 2. Command ‘git tag‘ returned non-zero exit status ***
 weight权重没有读到也无法从github下载
 这个基本上没有啥好的办法，只能手动下载权重传上去，国内挂梯子也没用
 3. dataSet can not be found
 这个实际不一定是数据集没有读到，看第一个弹出的错误进行解决，最后的这个报错是最根本的问题所以

### 速度
目前没有跑太多的数据集只有一个70张自己的数据集
一分钟不到就跑完了（这速度也太恐怖了）
>10 epochs completed in 0.015 hours.

目前还没修改测试集但是也许咱们也不是太需要测试直接把权重放在设备上跑就行
