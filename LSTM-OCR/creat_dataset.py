import os
import random
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# 数据集，可根据需要增加英文或其它字符
DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 分类数量
num_classes = len(DIGITS) + 1     # 数据集字符数+特殊标识符

# 图片大小，32 x 256
OUTPUT_SHAPE = (60, 24)

# 学习率
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 100
LEARNING_RATE_DECAY_FACTOR = 0.9
MOMENTUM = 0.9

# LSTM网络层次
num_hidden = 128
num_layers = 2

# 训练轮次、批量大小
num_epochs = 50000
BATCHES = 10
BATCH_SIZE = 32
TRAIN_SIZE = BATCHES * BATCH_SIZE

# 数据集目录、模型目录
data_dir = './tmp/lstm_ctc_data/'
test_data_dir = './tmp/test_lstm_ctc_data/'
model_dir = './tmp/lstm_ctc_model/'
# 生成椒盐噪声
def img_salt_pepper_noise(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.randint(0,1)==0:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

# 随机生成不定长图片集
def gen_text(cnt):
    # 设置文字字体和大小
    font_path = './fonts/Arial.ttf'
    font_size = 30
    font = ImageFont.truetype(font_path,font_size)
    c = 1
    for i in range(cnt):
        # 随机生成1到5位的不定长数字
        rnd = random.randint(1, 5)
        text = ''
        for j in range(rnd):
            text = text + DIGITS[random.randint(0, len(DIGITS) - 1)]
# 生成图片并绘上文字
        img=Image.open('./tmp/background/bg.jpg')
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
        draw=ImageDraw.Draw(img)
        draw.text((1,1),text,font=font,fill='black')
        img=np.array(img)
# 随机叠加椒盐噪声并保存图像
        img = img_salt_pepper_noise(img, float(random.randint(1,10)/100.0))
        print('生成中，目前第' + str(c) + '张，剩余' + str(cnt - c) + '张')
        c = c+1
        cv2.imwrite(test_data_dir + text + '_' + str(i+1) + '.jpg',img)


# 序列转为稀疏矩阵
# 输入：序列
# 输出：indices非零坐标点，values数据值，shape稀疏矩阵大小
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape

# 稀疏矩阵转为序列
# 输入：稀疏矩阵
# 输出：序列
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


# 将文件和标签读到内存，减少磁盘IO
def get_file_text_array():
    file_name_array=[]
    text_array=[]
    for parent, dirnames, filenames in os.walk(data_dir):
        file_name_array=filenames
    for f in file_name_array:
        text = f.split('_')[0]
        text_array.append(text)
    return file_name_array,text_array

# 获取训练的批量数据
def get_next_batch(file_name_array,text_array,batch_size=64):
    inputs = np.zeros([batch_size, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
    codes = []

# 获取训练样本
    for i in range(batch_size):
        index = random.randint(0, len(file_name_array) - 1)
        image = cv2.imread(data_dir + file_name_array[index])
        image = cv2.resize(image, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]), 3)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        text = text_array[index]
# 矩阵转置
        inputs[i, :] = np.transpose(image.reshape((OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))
        # 标签转成列表
        codes.append(list(text))
# 标签转成稀疏矩阵
    targets = [np.asarray(i) for i in codes]
    sparse_targets = sparse_tuple_from(targets)
    seq_len = np.ones(inputs.shape[0]) * OUTPUT_SHAPE[1]
    return inputs, sparse_targets, seq_len

def get_train_model():
    # 输入
    inputs = tf.compat.v1.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]])

# 稀疏矩阵
    targets = tf.compat.v1.sparse_placeholder(tf.int32)

# 序列长度 [batch_size,]
    seq_len = tf.compat.v1.placeholder(tf.int32, [None])

# 定义LSTM网络
    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    #stack = tf.nn.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)      # old
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


# 准确性评估
# 输入：预测结果序列 decoded_list ,目标序列 test_targets
# 返回：准确率
def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)

# 正确数量
    true_numer = 0

# 预测序列与目标序列的维度不一致，说明有些预测失败，直接返回
    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return

# 比较预测序列与结果序列是否一致，并统计准确率
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    accuracy = true_numer * 1.0 / len(original_list)
    print("Test Accuracy:", accuracy)

    return accuracy


def train():
    # 获取训练样本数据
    file_name_array, text_array = get_file_text_array()

# 定义学习率
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    # 获取网络结构
    logits, inputs, targets, seq_len, W, b = get_train_model()

# 设置损失函数
    loss = tf.compat.v1.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

# 设置优化器
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    decoded, log_prob = tf.compat.v1.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    init = tf.compat.v1.global_variables_initializer()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session() as session:
        session.run(init)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=10)
        for curr_epoch in range(num_epochs):
            train_cost = 0
            train_ler = 0
            for batch in range(BATCHES):
                # 训练模型
                train_inputs, train_targets, train_seq_len = get_next_batch(file_name_array, text_array, BATCH_SIZE)
                feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
                b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _ = session.run(
                    [loss, targets, logits, seq_len, cost, global_step, optimizer], feed)

# 评估模型
                if steps > 0 and steps % REPORT_STEPS == 0:
                    test_inputs, test_targets, test_seq_len = get_next_batch(file_name_array, text_array, BATCH_SIZE)
                    test_feed = {inputs: test_inputs,targets: test_targets,seq_len: test_seq_len}
                    dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
                    report_accuracy(dd, test_targets)

                    # 保存识别模型
                    save_path = saver.save(session, model_dir + "lstm_ctc_model.ctpk",global_step=steps)

                c = b_cost
                train_cost += c * BATCH_SIZE

            train_cost /= TRAIN_SIZE
            # 计算 loss
            train_inputs, train_targets, train_seq_len = get_next_batch(file_name_array, text_array, BATCH_SIZE)
            val_feed = {inputs: train_inputs,targets: train_targets,seq_len: train_seq_len}
            val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)
            #log = "{} Epoch {}/{}, steps = {}, train_cost = {:.3f}, val_cost = {:.3f}"
            print('Epoch:' + str(curr_epoch + 1,) + ';' + 'steps:' + str(steps) + '/' + str(num_epochs) + ';' + 'train_cost:' + str(train_cost) + ';' + 'val_cost:' + str(val_cost))


# LSTM+CTC 文字识别能力封装
# 输入：图片
# 输出：识别结果文字
def predict(image_file_path):
# 获取网络结构
    #print(str(image))
    logits, inputs, targets, seq_len, W, b = get_train_model()
    decoded, log_prob = tf.compat.v1.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        # 加载模型
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        # 图像预处理
        image = cv2.imread(image_file_path, 1)
        print(OUTPUT_SHAPE)
        image = cv2.resize(image, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]), 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        pred_inputs = np.zeros([1, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
        pred_inputs[0, :] = np.transpose(image.reshape((OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))
        pred_seq_len = np.ones(1) * OUTPUT_SHAPE[1]
        # 模型预测
        pred_feed = {inputs: pred_inputs,seq_len: pred_seq_len}
        dd, log_probs = sess.run([decoded[0], log_prob], pred_feed)
        # 识别结果转换
        detected_list = decode_sparse_tensor(dd)[0]
        detected_text = ''
        for d in detected_list:
            detected_text = detected_text + d

    return detected_text

if __name__ == '__main__':
    #gen_text(15)
    train()
    #image_file_path = './test/3000_0183.jpg'
    #result = predict(image_file_path)
    #origin = image_file_path.split('/')[2].split('_')[0]
    #print('实际值为：', origin)
    #cv2.namedWindow("ori", 0)
    #img = cv2.imread(image_file_path)
    #cv2.imshow('ori', img)
    #print('预测值为：' + result)
    #cv2.waitKey(0)