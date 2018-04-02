import tensorflow as tf
import numpy as np
import sklearn.metrics as sk

IMG_SIZE_PX = 50
SLICE_COUNT = 20
n_classes = 2
batch_size = 10


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,2,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,2,32,64])),
               'W_fc':tf.Variable(tf.random_normal([54080,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])

    conv1 = maxpool3d(conv1)
    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

much_data = np.load('muchdata-50-50-20.npy')
train_data = much_data[:-160]
validation_data = much_data[-40:]



def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        successful_runs = 0
        total_runs = 0

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for data in train_data:
                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    pass
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
        print('Done. Finishing accuracy:')
        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
        print('fitment percent:',successful_runs/total_runs)


        y_p = tf.argmax(prediction, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]})
        print('y_pred:',y_pred)

        print("validation accuracy:", val_accuracy)
        y_true = []
        for i in validation_data:
            y_true.append(np.argmax(i[1]))

        print ("Precision", sk.precision_score(y_true, y_pred))
        print ("Recall", sk.recall_score(y_true, y_pred))
        print ("f1_score", sk.f1_score(y_true, y_pred))
        print ("confusion_matrix")
        print (sk.confusion_matrix(y_true, y_pred))
        fpr, tpr, tresholds = sk.roc_curve(y_true, y_pred)
        print (fpr)
        print (tpr)
        print (tresholds)



x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8

train_neural_network(x)