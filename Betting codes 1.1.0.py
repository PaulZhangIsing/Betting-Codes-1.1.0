import pandas as pd
import numpy as np
import tensorflow as tf


#just import data first using pandas
#taking first 100000 lines as learning data, called it data
tf.set_random_seed(1)
rawdata = pd.read_csv('hdp_2017-07-01_6.csv', delimiter=',')
data = rawdata.head(100000)

# Technical Parameters
corner_h = data.ix[:, "corner_h"]
corner_c = data.ix[:, "corner_c"]
home_red = data.ix[:, "home_red"]
guest_red = data.ix[:, "guest_red"]
ycard_c = data.ix[:, "ycard_c"]
ycard_h = data.ix[:, "ycard_h"]
penalty_h = data.ix[:, "penalty_h"]
penalty_c = data.ix[:, "penalty_c"]
shootontarget_h = data.ix[:, "shootontarget_h"]
shootontarget_c = data.ix[:, "shootontarget_c"]
shootmiss_h = data.ix[:, "shootmiss_h"]
shootmiss_c = data.ix[:, "shootmiss_c"]
attack_h = data.ix[:, "attack_h"]
attack_c = data.ix[:, "attack_c"]
dangerousattack_h = data.ix[:, "dangerousattack_h"]
dangerousattack_c = data.ix[:, "dangerousattack_c"]
control_h = data.ix[:, "control_h"]
control_c = data.ix[:, "control_c"]

# betting Parameters
ior_RH = data.ix[:, "ior_RH"]
ior_RC = data.ix[:, "ior_RC"]
change_since_last_ior_RH = data.ix[:, "change_since_last_ior_RH"]
change_since_last_ior_RH_percent = data.ix[:, "change_since_last_ior_RH_percent"]
change_since_last_ior_RC = data.ix[:, "change_since_last_ior_RC"]
change_since_last_ior_RC_percent = data.ix[:, "change_since_last_ior_RC_percent"]
slope = data.ix[:, "slope"]
slope4 = data.ix[:, "slope4"]
trend4 = data.ix[:, "trend4"]
pankou = data.ix[:, "pankou"]
lastPankou = data.ix[:, "lastPankou"]
ior_RH_max_1min = data.ix[:, "ior_RH_max_1min"]
ior_RH_min_1min = data.ix[:, "ior_RH_min_1min"]
ior_RH_change_1min = data.ix[:, "ior_RH_change_1min"]
ior_RH_change_1min_rate = data.ix[:, "ior_RH_change_1min_rate"]
ior_RH_change_1min_startend = data.ix[:, "ior_RH_change_1min_startend"]
ior_RC_max_1min = data.ix[:, "ior_RC_max_1min"]
ior_RC_min_1min = data.ix[:, "ior_RC_min_1min"]
ior_RC_change_1min = data.ix[:, "ior_RC_change_1min"]
ior_RC_change_1min_rate = data.ix[:, "ior_RC_change_1min_rate"]
ior_RC_change_1min_startend = data.ix[:, "ior_RC_change_1min_startend"]
water_change_count_1min = data.ix[:, "water_change_count_1min"]
sign_change_count_1min = data.ix[:, "sign_change_count_1min"]
pankou_change_count_1min = data.ix[:, "pankou_change_count_1min"]
ior_RH_max_5min = data.ix[:, "ior_RH_max_5min"]
ior_RH_min_5min = data.ix[:, "ior_RH_min_5min"]
ior_RH_change_5min = data.ix[:, "ior_RH_change_5min"]
ior_RH_change_5min_rate = data.ix[:, "ior_RH_change_5min_rate"]
ior_RH_change_5min_startend = data.ix[:, "ior_RH_change_5min_startend"]
ior_RC_max_5min = data.ix[:, "ior_RC_max_5min"]
ior_RC_min_5min = data.ix[:"ior_RC_min_5min"]
ior_RC_change_5min = data.ix[:, "ior_RC_change_5min"]
ior_RC_change_5min_rate = data.ix[:, "ior_RC_change_5min_rate"]
ior_RC_change_5min_startend = data.ix[:, "ior_RC_change_5min_startend"]
water_change_count_5min = data.ix[:, "water_change_count_5min"]
sign_change_count_5min = data.ix[:, "sign_change_count_5min"]
pankou_change_count_5min = data.ix[:, "pankou_change_count_5min"]
ior_RH_max_10min = data.ix[:, "ior_RH_max_10min"]
ior_RH_min_10min = data.ix[:, "ior_RH_min_10min"]
ior_RH_change_10min = data.ix[:, "ior_RH_change_10min"]
ior_RH_change_10min_rate = data.ix[:, "ior_RH_change_10min_rate"]
ior_RH_change_10min_startend = data.ix[:, "ior_RH_change_10min_startend"]
ior_RC_max_10min = data.ix[:, "ior_RC_max_10min"]
ior_RC_min_10min = data.ix[:, "ior_RC_min_10min"]
ior_RC_change_10min = data.ix[:, "ior_RC_change_10min"]
ior_RC_change_10min_rate = data.ix[:, "ior_RC_change_10min_rate"]
ior_RC_change_10min_startend = data.ix[:, "ior_RC_change_10min_startend"]
water_change_count_10min = data.ix[:, "water_change_count_10min"]
sign_change_count_10min = data.ix[:, "sign_change_count_10min"]
pankou_change_count_10min = data.ix[:, "pankou_change_count_10min"]
ior_RH_max_15min = data.ix[:, "ior_RH_max_15min"]
ior_RH_min_15min = data.ix[:, "ior_RH_min_15min"]
ior_RH_change_15min = data.ix[:, "ior_RH_change_15min"]
ior_RH_change_15min_rate = data.ix[:, "ior_RH_change_15min_rate"]
ior_RH_change_15min_startend = data.ix[:, "ior_RH_change_15min_startend"]
ior_RC_max_15min = data.ix[:, "ior_RC_max_15min"]
ior_RC_min_15min = data.ix[:, "ior_RC_min_15min"]
ior_RC_change_15min = data.ix[:, "ior_RC_change_15min"]
ior_RC_change_15min_rate = data.ix[:, "ior_RC_change_15min_rate"]
ior_RC_change_15min_startend = data.ix[:, "ior_RC_change_15min_startend"]
water_change_count_15min = data.ix[:, "water_change_count_15min"]
sign_change_count_15min = data.ix[:, "sign_change_count_15min"]
pankou_change_count_15min = data.ix[:, "pankou_change_count_15min"]
ior_RH_max_30min = data.ix[:, "ior_RH_max_30min"]
ior_RH_min_30min = data.ix[:, "ior_RH_min_30min"]
ior_RH_change_30min = data.ix[:, "ior_RH_change_30min"]
ior_RH_change_30min_rate = data.ix[:, "ior_RH_change_30min_rate"]
ior_RH_change_30min_startend = data.ix[:, "ior_RH_change_30min_startend"]
ior_RC_max_30min = data.ix[:, "ior_RC_max_30min"]
ior_RC_min_30min = data.ix[:, "ior_RC_min_30min"]
ior_RC_change_30min = data.ix[:, "ior_RC_change_30min"]
ior_RC_change_30min_rate = data.ix[:, "ior_RC_change_30min_rate"]
ior_RC_change_30min_startend = data.ix[:, "ior_RC_change_30min_startend"]
water_change_count_30min = data.ix[:, "water_change_count_30min"]
sign_change_count_30min = data.ix[:, "sign_change_count_30min"]
pankou_change_count_30min = data.ix[:, "pankou_change_count_30min"]

# outcome for the prediction
win_or_lose_1 = data.ix[:, "win_or_lose_1"]
win_or_lose_2 = data.ix[:, "win_or_lose_2"]

# betting parametres



# hyperparameters
lr = 0.001
training_iters = 200000000
batch_size = 100000  # 一共120000个数据， 准备拿500组训练，100组用来验证

n_inputs = 5 # 一次有95个向量被输入，一共是18个技术参数+ 77个博彩参数
n_steps = 5
n_hidden_units = 200  # 200个神经元
n_classes = 3  # 3种不同的结果 胜负平
display_step = 50

# Define Input as X vector and output as Y vector
X_in = np.array([corner_h, corner_c, home_red, guest_red, ycard_h,
                 ycard_c, penalty_h, penalty_c, shootontarget_c, shootontarget_h, shootmiss_c, shootmiss_h,
                 attack_c, attack_h, dangerousattack_c, dangerousattack_h, control_h, control_c,ior_RH,ior_RC,change_since_last_ior_RH,
                 change_since_last_ior_RH_percent,change_since_last_ior_RC,change_since_last_ior_RC_percent,
                 slope])
X_in = X_in.reshape([batch_size,n_steps,n_inputs])


y1 = np.array([win_or_lose_1],dtype= np.float32)
print(y1)
print('y1 shape is :',y1.shape)

#enable one hot
for i in range (1,100000):
    if y1[0,i] == 1:
        y1[0,i] = 1
    elif y1[0,i] == -1:
        y1 [0,i] = 2
    else:
        y1[0,i] = 0

Y_in = tf.one_hot(y1,depth= 3)

Y_in = tf.transpose(Y_in, [1, 0, 2])

print('X_in shape is :', X_in.shape)
print(Y_in)

# Define x and y as inputs.

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights

weights = {
    # (18,200)
    "in": tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (200,3)
    "out": tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {

    # (200, )

    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),

    # (3, )

    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))

}


def RNN(X, weights, biases):
    # hidden layer for input to cell

    ########################################



    # transpose the inputs shape from

    # X ==> (100000 batch * 5 steps, 19 inputs)

    X = tf.reshape(X, [-1, n_inputs])

    # into hidden

    # X_in = ( 50 batch * 200 steps, 200 hidden)

    X_in = tf.matmul(X, weights['in']) + biases['in']

    # X_in ==> (500 batch, 200 steps, 200 hidden)

    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell

    ##########################################



    # basic LSTM Cell.

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:

        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    else:

        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)

    # lstm cell is divided into two parts (c_state, h_state)

    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results

    #############################################

    # results = tf.matmul(final_state[1], weights['out']) + biases['out']


    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:

        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs

    else:

        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


pred = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from

    # 2017-03-02 if using tensorflow >= 0.12

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:

        init = tf.initialize_all_variables()

    else:

        init = tf.global_variables_initializer()

    sess.run(init)

    step = 0

    while step * batch_size < training_iters:

        batch_xs = X_in

        batch_ys = Y_in

        batch_ys =tf.reshape(Y_in,[batch_size,n_classes]).eval()


        print('batch_xs shape is:',batch_xs.shape)







        sess.run([train_op], feed_dict={

            x: batch_xs,

            y: batch_ys,

        })

        if step % 2 == 0:
            print(sess.run(accuracy, feed_dict={

                x: batch_xs,

                y: batch_ys,

            }))

        step += 1

        # plotting

