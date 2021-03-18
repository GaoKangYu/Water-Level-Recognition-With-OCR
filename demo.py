#!python3
#pylint: disable=R, W0401, W0614, W0703
from ctypes import *
import random
import os
import cv2
import tensorflow as tf
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
import sys
import time
from tkinter import messagebox
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

start_ocr_flag = 0
img_path = ''
class WindowClass(QWidget):
    def __init__(self, parent=None):
        super(WindowClass, self).__init__(parent)
        self.setWindowTitle('水位检测简易演示系统')
        self.setMaximumSize(800, 600)
        self.setMinimumSize(800, 600)
        self.ui()
    def ui(self):
        self.btn_1=QPushButton("图片选取", self)
        self.btn_2=QPushButton("水位识别", self)
        self.ori = QLabel(self)
        self.pred_img = QLabel(self)
        self.btn_1.resize(100, 80)
        self.btn_1.move(0, 150)
        self.btn_2.resize(100, 80)
        self.btn_2.move(0, 300)
        self.ori.resize(160, 512)
        self.ori.move(200, 50)
        self.pred_img.resize(160, 512)
        self.pred_img.move(500, 50)
        self.btn_1.clicked.connect(self.select_img)
        self.btn_2.clicked.connect(self.start_ocr)
        self.show()

    def select_img(self):
        global start_ocr_flag
        global img_path
        start_ocr_flag = 0
        self.imageName, imgType = QFileDialog.getOpenFileName(self, "openImage", "*.jpg")
        # 判断图片路径是否为空
        if self.imageName != "":
            start_ocr_flag = 1
            img_path = self.imageName
            jpg = QtGui.QPixmap(self.imageName).scaled(self.ori.width(), self.ori.height())
            self.ori.setPixmap(jpg)  # 在label控件上显示选择的图片
            # ....省略一些对图片的操作


    def start_ocr(self):
        global start_ocr_flag
        if start_ocr_flag:
            print('Got img:', img_path)
            pred = performDetect(img_path)
            pred = cvimg_to_qtimg(pred)
            pred = QtGui.QPixmap(pred).scaled(self.pred_img.width(), self.pred_img.height())
            self.pred_img.setPixmap(pred)

tf.compat.v1.disable_eager_execution()

# 数据集
DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 分类数量
num_classes = len(DIGITS) + 1     # 数据集字符数+特殊标识符

# 图片大小，32 x 256
OUTPUT_SHAPE = (60, 24)

# LSTM网络层次
num_hidden = 128
num_layers = 2

# 数据集目录、模型目录
model_dir = './'


def cvimg_to_qtimg(cvimg):
    height, width, depth = cvimg.shape
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    qtimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)

    return qtimg

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "detect_cpu.dll")
    print(winNoGPUdll)
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # print(os.environ.keys())
            # print("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find `"+winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

def network_width(net):
    return lib.network_width(net)

def network_height(net):
    return lib.network_height(net)

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def imagecrop(image, box):
    crop_image = []
    if box != [[0, 0], [0, 0], [0, 0], [0, 0]]:
        xs = [x[1] for x in box]
        ys = [x[0] for x in box]
        crop_image = image[min(xs):max(xs), min(ys):max(ys)]
    return crop_image

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    """
    Performs the meat of the detection
    """
    #pylint: disable= C0321
    im = load_image(image, 0, 0)
    if debug: print("Loaded image")
    ret = detect_image(net, meta, im, thresh, hier_thresh, nms, debug)
    free_image(im)
    if debug: print("freed image")
    return ret

def detect_image(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    num = c_int(0)
    if debug: print("Assigned num")
    pnum = pointer(num)
    if debug: print("Assigned pnum")
    predict_image(net, im)
    letter_box = 0
    #predict_image_letterbox(net, im)
    #letter_box = 1
    if debug: print("did prediction")
    #dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)
    if debug: print("Got dets")
    num = pnum[0]
    if debug: print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug: print("did sort")
    res = []
    if debug: print("about to range")
    for j in range(num):
        if debug: print("Ranging on "+str(j)+" of "+str(num))
        if debug: print("Classes: "+str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug: print("Class-ranging on "+str(i)+" of "+str(meta.classes)+"= "+str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug: print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug: print("did sort")
    free_detections(dets, num)
    if debug: print("freed detections")
    return res


netMain = None
metaMain = None
altNames = None

def performDetect(imagePath, thresh= 0.25, configPath = "./detect.cfg", weightPath = "./detect.weights", metaPath= "./detect.data", initOnly= False):
    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    global metaMain, netMain, altNames #pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `"+os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `"+os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `"+os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    if initOnly:
        print("Initialized detector")
        return None
    if not os.path.exists(imagePath):
        raise ValueError("Invalid image path `"+os.path.abspath(imagePath)+"`")
    # Do the detection
    img = cv2.imread(imagePath)
    ori_img = cv2.imread(imagePath)
    detections = detect(netMain, metaMain, imagePath.encode("ascii"), thresh)
    y_single_button = 0
    single_point_button = [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0]
    ]
    single_point_button_1 = (0, 0)
    single_point_button_2 = (0, 0)
    y_mult_button = 0
    mult_point_button = [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0]
    ]
    mult_point_button_1 = (0, 0)
    mult_point_button_2 = (0, 0)
    L = 0.0
    L_test = 0.0
    L_1 = 0
    L_2 = ''
    L_3 = random.randint(3, 5)
    for detection in detections:
        bounds = detection[2]
        yExtent = int(bounds[3])
        xEntent = int(bounds[2])
        xCoord = int(bounds[0] - bounds[2] / 2)
        yCoord = int(bounds[1] - bounds[3] / 2)
        boundingBox = [
            [xCoord, yCoord],
            [xCoord, yCoord + yExtent],
            [xCoord + xEntent, yCoord + yExtent],
            [xCoord + xEntent, yCoord]
        ]
        # 获取数字区域的长度，即X2-X1的情况，获取一个判断基准值，大于单个数字，小于超长数字
        # 实验测试可得：
        # 单个数字length：小于35
        # 多个数字length：大于45，小于80
        # 超长数字length：大于120
        box_length = boundingBox[2][0] - boundingBox[0][0]
        # 根据Y轴情况找出最下面的单个数字bbox
        if box_length < 35:
            if y_single_button < boundingBox[0][1]:
                y_single_button = boundingBox[0][1]
                single_point_button = boundingBox
                single_point_button_1 = (boundingBox[0][0], boundingBox[0][1])
                single_point_button_2 = (boundingBox[2][0], boundingBox[2][1])
        # 根据Y轴情况找出最下面的长数字bbox
        if 45 < box_length < 80:
            if y_mult_button < boundingBox[0][1]:
                y_mult_button = boundingBox[0][1]
                mult_point_button = boundingBox
                mult_point_button_1 = (boundingBox[0][0], boundingBox[0][1])
                mult_point_button_2 = (boundingBox[2][0], boundingBox[2][1])

    single_ocr_area = imagecrop(img, single_point_button)
    if single_ocr_area == []:
        L_1 = 0
        img = cv2.putText(img, 'No single_ocr_area', (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (42, 42, 165), 2)
    elif single_ocr_area != []:
        single_result = ocr_predict(single_ocr_area)
        if single_result == '':
            L_1 = 0
            cv2.rectangle(img, single_point_button_1, single_point_button_2, (255, 0, 0), 2)
            img = cv2.putText(img, 'Null', single_point_button_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (255, 255, 255), 2)
        else:
            L_1 = int(single_result)
            if L_1 > 9:
                L_1 = 0
                cv2.rectangle(img, single_point_button_1, single_point_button_2, (255, 0, 0), 2)
                img = cv2.putText(img, 'Null', single_point_button_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (255, 255, 255), 2)
            else:
                cv2.rectangle(img, single_point_button_1, single_point_button_2, (255, 0, 0), 2)
                img = cv2.putText(img, str(single_result), single_point_button_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (255, 255, 255), 2)

    mult_ocr_area = imagecrop(img, mult_point_button)
    if mult_ocr_area == []:
        img = cv2.putText(img, 'Level: null', (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                          (42, 42, 165), 2)
        img = cv2.putText(img, 'No multi_ocr_area', (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (42, 42, 165), 2)
    elif mult_ocr_area != []:
        mult_result = ocr_predict(mult_ocr_area)
        if mult_result == '':
            cv2.rectangle(img, mult_point_button_1, mult_point_button_2, (255, 0, 0), 2)
            img = cv2.putText(img, 'Null', mult_point_button_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (255, 255, 255), 2)
            img = cv2.putText(img, 'Level: Null', (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                              (42, 42, 165), 2)
        else:
            L_2 = mult_result
            L_2 = float(L_2)
            L_test = L_2 - (10 - float(L_1)) * 0.1 - L_3 * 0.02
            cv2.rectangle(img, mult_point_button_1, mult_point_button_2, (255, 0, 0), 2)
            L_2 = float(L_2)
            img = cv2.putText(img, "{:.2f}".format(L_2), mult_point_button_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (255, 255, 255), 2)
            L_test = float(L_test)
            img = cv2.putText(img, 'Level:' + "{:.2f}".format(L_test), (8, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                              (42, 42, 165), 2)
            print('done!')


    return img

def get_train_model():
    # 输入
    inputs = tf.compat.v1.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]])

# 稀疏矩阵
    targets = tf.compat.v1.sparse_placeholder(tf.int32)

# 序列长度 [batch_size,]
    seq_len = tf.compat.v1.placeholder(tf.int32, [None])

# 定义LSTM网络
    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    outputs, _ = tf.compat.v1.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)
    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    outputs = tf.reshape(outputs, [-1, num_hidden])
    W = tf.Variable(tf.compat.v1.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")
    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

# 转置矩阵
    logits = tf.transpose(logits, (1, 0, 2))

    return logits, inputs, targets, seq_len, W, b

def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result

# 序列编码转换
def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = DIGITS[spars_tensor[1][m]]
        decoded.append(str)
    return decoded

def ocr_predict(image):
# 获取网络结构
    tf.compat.v1.reset_default_graph()
    logits, inputs, targets, seq_len, W, b = get_train_model()
    decoded, log_prob = tf.compat.v1.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    saver = tf.compat.v1.train.Saver()
    with tf.name_scope("fw_side"), tf.compat.v1.variable_scope("fw_side", reuse=tf.compat.v1.AUTO_REUSE):
      with tf.compat.v1.Session() as sess:
          with tf.device('/cpu:0'):
              # 加载模型
              saver.restore(sess, tf.train.latest_checkpoint(model_dir))
              # 图像预处理
              # print(OUTPUT_SHAPE)
              image = cv2.resize(image, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]), 3)
              image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
              pred_inputs = np.zeros([1, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
              pred_inputs[0, :] = np.transpose(image.reshape((OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))
              pred_seq_len = np.ones(1) * OUTPUT_SHAPE[1]
              # 模型预测
              pred_feed = {inputs: pred_inputs, seq_len: pred_seq_len}
              dd, log_probs = sess.run([decoded[0], log_prob], pred_feed)
              # 识别结果转换
              detected_list = decode_sparse_tensor(dd)[0]
              detected_text = ''
              for d in detected_list:
                  detected_text = detected_text + d
    return detected_text



if __name__ == "__main__":
    localtime = time.localtime(time.time())
    today = int(time.strftime("%Y%m%d"))
    if today > 20201001:
        messagebox.showerror("Error", "本系统的使用授权已过期！")
        sys.exit()
    else:
        app = QApplication(sys.argv)
        win = WindowClass()
        win.show()
        sys.exit(app.exec_())
