# Water-Level-Recognition-With-OCR
基于OCR的“水位检测”项目
## TBD
- [ ] English version README

## 背景/需求
- 使用OCR方法，基于水位尺，读出水位数值，预期精度0.01。
## 环境需求
-   Windows or Linux
-   Python >= 3.5
-   tensorflow-gpu >= 1.14.0
-   cuda >= 10.0

## 代码目录结构
```
├── Hough(霍夫变换检测直线)
├── LSTM-OCR(基于LSTM模型的OCR方法，包含部分权重)
├── Number_detect_dataset(用于训练数字检测模型YOLO v3的数据集)
├── Post_Process_Result(OCR模型输出)
├── data_for_ocr_train(用于训练OCR模型的数据集)
├── demo(演示系统源码与可执行文件下载地址)
├── TPS(模糊文本识别模型)
├── creat_dataset(用于生成合成数据集，训练TPS模型)
```
## 项目技术方案
### 整体思路

![scheme_diagram](https://github.com/GaoKangYu/Water-Level-Recognition-With-OCR/blob/main/readme_fig/scheme_diagram.png)

- 调整摄像头至仅拍摄水位尺区域，使用训练好的目标检测模型检测数字区域（如图1左），根据bounding box的位置关系（主要基于Y轴，根据实际情况需要分别讨论(1)），确定检测区域1和检测区域2，其中，检测区域1代表最下方的刻度区数字，精度为0.1，经过OCR模型识别数字，得到值L2（数值）；检测区域2为最下方的较长数字，精度为1，经过OCR模型识别数字，得到值L1（数值），检测区域1和检测区域2由bounding box的宽度区分（如图1中）；另外，检测区域1的bounding box下边界至水面为检测区域三，通过霍夫变换检测直线数目，精度为0.01，得到值L3（直线条数）；最终，通过计算式L1-（10-L2·0.1）-0.25·L3(1)，获得最终水位（如图1右）。
- 其中（1）中的讨论主要指较长数字区域（L1值）和刻度区数字区域（L2值）的位置关系，实际数据中，二者并不总是呈现L1在L2上方的情况。所以需要分情况确定加减号，详见附录。

### 方案实施

![raw_data](https://github.com/GaoKangYu/Water-Level-Recognition-With-OCR/blob/main/readme_fig/raw_data.png)

- 为了减少环境干扰，首先对所有数据进行筛选并提取出感兴趣区域，该区域为方案的原始输入。

![roi_dataset](https://github.com/GaoKangYu/Water-Level-Recognition-With-OCR/blob/main/readme_fig/roi_dataset.png)

- 随后，开始标注数据，数字检测基于YOLO v3模型，标注为YOLO格式，类别仅有number一类。其中，每一个数字都应该被标注，被水浸没或环境遮挡的数字不标注，标注框覆盖数字但要尽可能的小，刻度区尽量不要包含到刻度，否则会影响到OCR模型的准确度。

![data_label](https://github.com/GaoKangYu/Water-Level-Recognition-With-OCR/blob/main/readme_fig/data_label.png)

![number_detect](https://github.com/GaoKangYu/Water-Level-Recognition-With-OCR/blob/main/readme_fig/number_detect.png)

- 训练完成后，使用目标检测模型（YOLO v3，其权重文件位于dist\detect.weights）检测其中的数字，根据bounding box的宽度划分为刻度区数字和较长数字，根据bounding box右下角点的Y轴分别确定检测区域2和检测区域1，为可辨识的最下方刻度区数字、较长数字，用于输入OCR模型。对于OCR模型，承接上一步的工作，使用数字检测模型检测所有数据，根据bounding box信息裁剪出所有的数字区域并另存为图片，作为OCR模型的训练集，经人工标注，形成数据集于文件夹LSTM-OCR-Test\tmp\lstm_ctc_data，共1673张。

![OCR_dataset](https://github.com/GaoKangYu/Water-Level-Recognition-With-OCR/blob/main/readme_fig/OCR_dataset.png)

- 为了保障运行速度，OCR模型采用LSTM模型，基于tensorflow框架，权重位于dist\ recognition系列。经训练，承接数字检测模型的输入，输出该区域的数值，具体地，由于输入区域本身存在较多噪点和污染，为了减少错误率，OCR模型并未学习小数点“.”的表示，因此对于例如“22.00”的区域会识别为“2200”，只需要添加部分后处理（除以100，保留两位有效数字）即可变成正确值。

![Hough Transform](https://github.com/GaoKangYu/Water-Level-Recognition-With-OCR/blob/main/readme_fig/Hough_Transform.png)

- 至此，综合（1）的分情况处理，已经完成了0.1的识别精度。最后0.01的精度基于霍夫变换，其结果如下，受参数和环境因素影响大，较为不稳定。

### 附录

- 主要符号释义：
L：水位值；

L1：刻度区数字（单个数字区域；即为0、1、2等）；

L2：非刻度区数字（长数字区域，同时也不是电话，二者的区分通过区域长度，经试验，刻度区数字平均长度30px，长数字区域平均长度65px，电话区域平均长度150px；即为30.00等）；

L3：刻度区数字上方直线至图片最下部区域直线条数，使用霍夫变换检测得到；

- 整体逻辑：

1.目标检测，使用yolov3对“数字”区域进行检测。[训练集规模为77张，约2400个目标，权重mAP为95.63%]；

2.数字OCR，使用LSTM+CTC对检测到的数字区域进行识别。[训练集规模为1673 张，使用检测模型在检测训练集上裁剪得到，权重acc 为83.30%，47300/50000轮，权重约1mb]；

3.区域筛选（找出需要识别的刻度区数字与非刻度区数字）：

a.刻度区数字：检测到的数字区域中长度（x坐标差）小于30px且左上角y坐标最大的，对应刻度区最下面的数字；识别结果记为L1；

b.非刻度区数字：检测到的数字区域中长度（x坐标差）大于45px小于100px且左上角y坐标最大的，对应非刻度区最下面的数字；识别结果记为L2；

4.水位计算（分情况）：

a.刻度区数字左上角Y坐标远大于（演示系统判断基准值为40px）非刻度区数字左上角Y坐标；即对应为通常情况（刻度区数字在非刻度区数字下面），此时L = L2 - 0.1 * (10-L1) - 0.2 * L3；

b.刻度区数字小于或接近（演示系统判断基准值为10px）非刻度区数字左上角Y坐标；即对应为刻度区数字在非刻度区上方或者附近，此时水面接近非刻度区，L = L2 - 0.2 * L3；

c.未检测到非刻度区数字，由于不知道尺的默认高度，目前无法判断其水位，演示系统也没有加入该类图片测试。

- 存在问题：

1.检测：

存在漏检、误检测情况，对于倾斜、弯曲的尺子检测区域很容易错误，正面的尺子准确率尚可；

可能原因分析：

1）训练数据过少（仅77张，但目标区域很多，所以呈现的效果还可以）；

2）训练的时候输入网络之前预先进行了resize（变化不大，但是此步骤在标注之后进行的，导致标注存在些许偏差），可能导致标注有偏差（不是很大）；

3）输入图片分辨率不是常规（yolo 默认为416x416版式）尺寸，为仅包含尺区域的竖版图片，与部署情况下的获取画面不一致，后续正式开发的时候需要更多数据、重新标注。

2.识别：

存在误识别情况，单个数字误识别情况相对较少，长数字常常位数都错误，测试图片中若出现少于4位的，都进行了补‘0’的修正，但不是所有的位数错误都错在少‘0’，此问题由于数据分布不均匀造成，1673 张数据中，多位数字的数据大约只有300张，导致单个数字训练充分，长数字不充分。

3.霍夫变换，参数是多次试验调试出来的，鲁棒性不算很好（水面也当做了直线，演示时做了-1的修正），后续要加入去干扰的图像处理算法，具体什么方法还需要实验。

### 演示系统
- 以下为演示系统效果展示（基于PyQt，CPU-only）
- 本系统设置了授权限时，如需运行需要手动更改系统时间为2020年10月1日以前
- 系统ui

![ui](https://github.com/GaoKangYu/Water-Level-Recognition-With-OCR/blob/main/readme_fig/ui.png)

- 选取图片

![load_img](https://github.com/GaoKangYu/Water-Level-Recognition-With-OCR/blob/main/readme_fig/load_img.png)

- 显示选取图片

![show_img](https://github.com/GaoKangYu/Water-Level-Recognition-With-OCR/blob/main/readme_fig/show_img.png)

- 识别结果

![result](https://github.com/GaoKangYu/Water-Level-Recognition-With-OCR/blob/main/readme_fig/result.png)

