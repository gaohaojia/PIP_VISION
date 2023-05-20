# 视觉报告1
 23.1.9-23.1.20
 刘子赫
# 有关yolov8的部署
# 0. 环境（真的好麻烦）
（虽然这部分可能对于这个报告 来说没啥大用但是还是 写一下，一个是因为这环境就搞了一个星期还多，也写下来以后配置避雷）
## 软件包
这次yolo把所有的软件包集成成了一个ultralytics包直接pip就可以但是我最开始还是用的yolov5的配置方法所以一直有丢包，然后用yolov8里面的requirement写的安装也丢包（自己的requirement包丢包真的合适吗（也可能是我自己安丢了什么））后来看教程知道的直接安装ultralytics就行（然后就自己给自己埋雷）
## 虚拟环境
这部分可以说是历史遗留问题，之前在jupyter的环境配置就很乱所以这部分也带来了点麻烦。最开始使用的就是原先跑jupyter的pytorch的环境，yolo能跑但是显示没有cuda的参与用cpu跑，所以在调试的环境的这个地方花了很多时间。最后放弃了重新新建配置好几次（这里面也有各种的问题）。这部分配置好了之后用coco128数据集跑了一下能跑但是只能用cpu，如果训练自己的数据集就会报显存
## cuda
有关cuda的安装也是问题。在之前cuda需要单独安装但是有的教程并没有写cuda，有的教程说cuda已经包含在了pytorch的包里面，有的写了如何从英伟达那里安装cuda所以我也就一直没有搞清楚cuda为什么不在虚拟环境里面跑。尝试安装用pip命令安装cuda（对应版本也费了点时间）但是始终跑不起来。在从pytorch
安装带有cuda的pytorch之后无法import pytorch查看cuda情况。。最后在安装ultralytics包之后需要写在pytorch不能使用覆盖安装，同时pi命令可能会有问题需要用conda的命令。最后是成功安装了cuda。后来猜测可能是因为ultralytics包里面的pytorch是cpu版本的（这真的不是在坑人吗）也可能是因为ultralytics整合的pytorch但是没cuda（至于单独安装cuda为啥也用不了这就不清楚了）
## 终端的问题
yolov8的其中一个新特性就是可以是用命令行来训练不需要自己手动改参数所以最开始是直接在cmd里面输入命令但是没有成功。后在文件里面显示可以找到文件但是找不到标签，随后找了几个划分数据集的py程序。（还有一个网站叫Roboflow可以帮助标注数据集和划分，下载即可训练，看了一下介绍应该挺好用的但是我没尝试）。而在cmd进入虚拟环境以后还是不可以训练会有奇怪的报错，这部分也修改了很多次。最后在pycharm终端里面才可以训练，中间还有一些报错也修改了一下。（终端也许不需要activate环境还坑了我一把）

### 这里还有一个有关内核的问题
之前在jupyter经常内核挂掉所以加上了

>import os
os.environ[“KMP_DUPLICATE_LIB_OK”]=“TRUE”

这样一段代码
配置过程中有这样的报错
> OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized

加上了之前有的代码就解决了这个问题。（以前问题的的根源所在）
但是如果使用命令行来说可能每次都写一下有点不方便所以解决办法是吧虚拟环境下本身的那libiomp5md.dll删了（备份一下路径也备份一下就可以暂时性的解决这个问题（目前还没发现有啥问题）
## 新特性


目前是看的一些分析的csdn的文章与之前的（v5v7）的对比

具体的变化在：https://mp.weixin.qq.com/s/XK6UdMJ7pB-CK2P9tgt8xQ
至于具体的结构和损失函数啥的确实需要仔细的看源码肯定难度不小，所以我想得是用他训练完的精度看看能不能用在咱们的nx上面，或者说用不用换个yolo的版本再提高一下精度啥的

还有比如说可以用命令行训练，直接给了损失的图像啥的小细节
（虽然说这次还叫v8但是这结构也太复杂了）

 

## 一些其他的问题
### 有关num_worker的问题
这部分我倒是没遇到因为这个变量所导致进程报错的问题吗，但是很多人说不把这个变量改成0就跑不动。我确实是遇到了最后的报错是进程的问题但是不是因为这个引起的
### 有关cache的问题
这个问题是不能用cmd训练，pycharm会创建一个新的cache用于训练如果说是没有这个文件的话。cmd似乎不能在目标文件夹写入东西。（最开始用终端训练的时候也被坑了一把）
# 1. 实测（大概）
由于手头暂是没有太多的数据集所以只能用一下之前自己标注的70张左右的数据集（这个量肯定太少了），但是当时环境都没配置好所以也就没有要数据集（跑都跑不起来怎么搞实测啊）

至于官方的那个数据集手里也有但是不是很全（没有xml文件其实可以写个程序改一下但是没整）而且需要自己去拆分训练和验证集有点麻烦我也看不到原始数据的框图所以暂时没有拿这个训练（也怕1000多张训练不完）

也想着就是作为一个测试看看数据yolo自己的网络同时看看精度loss啥的

# 2.结果

具体结果在ultralytics-main\runs\detect\train14文件夹下（可能后续还有训练结果）
训练配置如下
>yolo task=detect mode=train model=yolov8s.pt data=E:/Desktop/ultralytics-main/data/report.yaml batch=10 epochs=7 imgsz=640 workers=1 device=0

第一次的训练没有权重只能拿官方给的yolo8s.pt训练多以很多训练验证集都成了数字识别但是最后的结果还是自己的标签

具体的训练过程在训练实录的txt里面（其中的一次）

## 精度
看了一下一半都不太准（肯定是因为数据少了而且batch也不多大）但是有些准的地方置信度是很高的（图和标签可能有点糊在一起到时候找点好的数据集去训练）

如果可以的话我会再拿训练好的best.pt去训练这里报告就先不改了如果有结果我单独发一下或者方压缩包里面吧

这里只是测试一下结果，原本想跟yolov5比较一下到那时一想这点数据谁也跑不出来什么所以就没有搞对比

## 速度
速度的话我也不知道怎么评价因为毕竟是自己的电脑性能肯定跟设备不一样具体如下
>Validating runs\detect\train14\weights\best.pt...
Ultralytics YOLOv8.0.9  Python-3.8.16 torch-1.13.1 CUDA:0 (NVIDIA GeForce RTX 3060 Laptop GPU, 6144MiB)
Fusing layers... 
Model summary: 168 layers, 11127906 parameters, 0 gradients, 28.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:05<00:00,  5.31s/it]
                   all          7         13      0.426        0.3      0.256      0.154
            armor3blue          7          2      0.463        0.5      0.511       0.31
            armor4grey          7          4     0.0616        0.5      0.295       0.23
            armor4blue          7          5      0.181        0.2      0.175     0.0582
             armor4red          7          2          1          0     0.0433     0.0178
Speed: 1.6ms pre-process, 35.8ms inference, 0.0ms loss, 3.9ms post-process per image
Saving runs\detect\train14\predictions.json...
Results saved to runs\detect\train14

如果按照其他的测评来说（基本上都是用的coco128的数据集）v8在速度上（特别是在小模型里面提升并没有很显著），精度也只在打的模型上面的有一定的提升

## 模型
这里也没做太多的测试只用了yolo8s.pt作为第一次的训练（后续可能用别的我这里暂时先不写）

官方还没给出模型的速度啥的目前只有评测的速度


# 3.可行性
## 有关tensorRT的加速
由于是新的模型所以目前还没有有关yolov8的教程（自己也摸索了一下但是配置环境真的给我整的崩溃了所以暂时没搞这个）（毕竟yolov8的改动还是挺大的而且配置跟之前不太一样所以我也没信心搞这玩意）但是既然是yolo也是英伟达应该可以搞加速

## 准确度
自己这里肯定是不准的所以很难说能不能用但是根据网上目前的测评来说，速度上与yolov7没有太大的差别，精度上也只是在大模型上买上面有一定的提升，小的模型的提升不大

# 总结
总之yolov8是可以尝试一下部署试试精度的，前提是需要有好的数据集，然后各种网络都试一下，有tensorRT更好就

大概报告就这样基本上吧想说的都说了，主要还是配置环境在磨人了一下进度就慢了。之后自己再试试用yolov8泡点别的数据集啥的看看。


### 一些reference

 1. https://blog.csdn.net/weixin_45921929/article/details/128673338
 2. https://blog.csdn.net/qq_37164776/article/details/126832303?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167414206616782428667128%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167414206616782428667128&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2
