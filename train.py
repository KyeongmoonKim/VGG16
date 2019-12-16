import tensorflow as tf
import network as net
import data_csv as data
import configure as cf
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
log = cf.TRA_LOG
f = open(log,'w')
wr = csv.writer(f)

#ckpt_path = './vgg16.ckpt'
ckpt_path = cf.TRA_CKPT_PATH
sess = tf.Session()
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
answers = tf.placeholder(tf.float32, [None, cf.CLS_NUM])
train_mode = tf.placeholder(tf.bool)

dt = data.Data(cf.TRA_IMG_PATH, cf.TRA_CSV, oversampling=cf.OVER)
vgg = net.Network(ckpt_path, fine_tuning=cf.FINE_TUNING, hard=cf.HARD)
vgg.build(images, train_mode)


    
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=answers, logits=vgg.fc8)
loss = tf.reduce_mean(cross_entropy) 
optimizer = tf.train.AdamOptimizer(cf.LEARNING_RATE).minimize(loss)
#graph finish

sess.run(tf.global_variables_initializer())

var_list = vgg.build_tensors()
#print(var_list)
loader = tf.train.Saver(var_list)

loader.restore(sess, ckpt_path)
print("start training")
for i in range(0, cf.ITER):
    if(i%cf.SAVE_ITER == 0)and(i!=0):
        print("trying to saving...")
        vgg.save_model(sess, i)
    im_list, la_list = dt.get_batch()
    r1, r2 = sess.run([optimizer, loss], feed_dict={images: im_list, answers: la_list, train_mode:True})
    line = [i, r2]
    wr.writerow(line)
    print("iteration "+ str(i)+" loss : " + str(r2))
    
f.close()
    
    