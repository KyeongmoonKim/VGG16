import tensorflow as tf
import network as net
import data_csv as data
import configure as cf
import numpy as np
import utils as ut
import csv
'''BATCH_SIZE = 32
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001
DROP_OUT = 0.5
CLS_NUM = 102
ITER = 50000
FINE_TUNING = True'''

print("BATCH_SIZE : " + str(cf.BATCH_SIZE))
print("LEARNING_RATE : " + str(cf.LEARNING_RATE))
print("DROP_OUT : " + str(cf.DROP_OUT))
print("CLS_NUM : " + str(cf.CLS_NUM))
print("ITER : " + str(cf.ITER))
print("FINE_TUNING : " + str(cf.FINE_TUNING))
#with tf.device('/GPU:0')
fd = open(cf.TES_LOG, 'w', newline='')
wr = csv.writer(fd)

cnt_matrix = []
for i in range(0, cf.CLS_NUM):
    cnt_matrix.append([0]*cf.CLS_NUM)

#test_sz = 290
ckpt_path = cf.TES_CKPT_PATH 
sess = tf.Session()
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
answers = tf.placeholder(tf.float32, [None, cf.CLS_NUM])
train_mode = tf.placeholder(tf.bool)

test_sz = 0
f_temp = open(cf.TES_CSV, 'r')
rdr_temp = csv.reader(f_temp)
for line in rdr_temp:
  test_sz = test_sz + 1
f_temp.close()
print("test sz : " + str(test_sz))
dt = data.Data(cf.TES_IMG_PATH , cf.TES_CSV)
vgg = net.Network(ckpt_path, trainable=False, fine_tuning=False)
vgg.build(images, train_mode)

#cnt = [0]*cf.CLS_NUM
#ans = [0]*cf.CLS_NUM
    
#graph finish 

sess.run(tf.global_variables_initializer())

var_list = vgg.build_tensors()
#print(var_list)
loader = tf.train.Saver(var_list)

loader.restore(sess, ckpt_path)
print("start testing")
end = int(test_sz/cf.BATCH_SIZE)
ret = []
for i in range(0, end):
    im_list, la_list = dt.get_batch()
    #print(im_list.shape)
    #print(la_list.shape)
    r1, r2 = sess.run([vgg.prob, vgg.fc8], feed_dict={images: im_list, train_mode:False})
    l, clsnum = la_list.shape 
    for j in range(0, l):
      lab = np.argmax(la_list[j])
      cnt_matrix[lab][np.argmax(r1[j])] = cnt_matrix[lab][np.argmax(r1[j])] + 1 
    ret.append(ut.get_accuracy(r1, la_list))
    print(str(i)+" batch accuracy : "+str(ret[i]))

answers_sum = []
answers_count = []
for i in range(0, cf.CLS_NUM):
    answers_sum.append(sum(cnt_matrix[i]))
    answers_count.append(cnt_matrix[i][i])
acc = sum(answers_count)/sum(answers_sum)
print(answers_sum)
print(answers_count)
print(acc) 
for i in range(0, cf.CLS_NUM):
    wr.writerow(cnt_matrix[i])
fd.close()