
import numpy as np
import tensorflow as tf
import time
import cv2
import math

from get_data import *


tf.set_random_seed(1)
np.random.seed(1)
tf.keras.backend.clear_session()

num_classes = 43
lr = 0.01 
epochs = 20
batch_size = 64

with tf.device('/gpu:0'): # run on GPU
    x_initializer = tf.contrib.layers.xavier_initializer()
    x_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3]) 
    y_placeholder = tf.placeholder(tf.float32, shape=[None, num_classes])
    is_training = tf.placeholder(tf.bool, shape=()) # for Batch Normalization layer
    
    
    def mn_v2_reziduu(intrare, expand, squeeze, ksize, strides, padding, reziduu):
        
        # --------------- 1x1 -> 3x3 -> 1x1
        k1x1_1 = 1
        c1x1_1 = int(intrare.get_shape()[3])
        h1x1_1 = expand
        
        W1x1_1 = tf.Variable(x_initializer([1,1,int(intrare.get_shape()[3]),expand]))
        b1x1_1 = tf.Variable(x_initializer([expand]))
        conv1x1_1 = tf.nn.relu6(tf.compat.v1.layers.batch_normalization(
                    tf.add(tf.nn.conv2d(intrare, W1x1_1, strides=[1,1,1,1], padding="SAME"), b1x1_1), 
                    training=is_training, momentum=0.9, trainable=True))
        
        # conv dw 3x3   BN + relu
        pc1x1_1 =  (k1x1_1*k1x1_1*c1x1_1+1)*h1x1_1
        cc1x1_1 = (k1x1_1*k1x1_1*c1x1_1+1)*h1x1_1*conv1x1_1.get_shape()[1]*conv1x1_1.get_shape()[2]

        
        k3x3 = ksize
        c3x3 = int(conv1x1_1.get_shape()[3])
        h3x3 = 1
        
        W3x3 = tf.Variable(x_initializer([ksize,ksize,int(conv1x1_1.shape[3]),1]))
        b3x3 = tf.Variable(x_initializer([expand]))
        conv3x3 = tf.nn.relu6(tf.compat.v1.layers.batch_normalization(
                    tf.add(tf.nn.depthwise_conv2d(conv1x1_1, W3x3, strides=[1,strides,strides,1], padding=padding), b3x3), 
                    training=is_training, momentum=0.9, trainable=True))
        
        # conv pw 1x1   BN
        pc3x3 =  (k3x3*k3x3*c3x3+1)*h3x3
        cc3x3 = (k3x3*k3x3*c3x3+1)*h3x3*conv3x3.get_shape()[1]*conv3x3.get_shape()[2]

        k1x1_2 = 1
        c1x1_2 = int(conv3x3.get_shape()[3])
        h1x1_2 = squeeze
        
        W1x1_2 = tf.Variable(x_initializer([1,1,int(conv3x3.get_shape()[3]),squeeze]))
        b1x1_2 = tf.Variable(x_initializer([squeeze]))
        conv1x1_2 = tf.compat.v1.layers.batch_normalization(
                    tf.add(tf.nn.conv2d(conv3x3, W1x1_2, strides=[1,1,1,1], padding="SAME"), b1x1_2) + reziduu, 
                    training=is_training, momentum=0.9, trainable=True)
        pc1x1_2 =  (k1x1_2*k1x1_2*c1x1_2+1)*h1x1_2
        cc1x1_2 = (k1x1_2*k1x1_2*c1x1_2+1)*h1x1_2*conv1x1_2.get_shape()[1]*conv1x1_2.get_shape()[2]

        m1_pc = pc1x1_1 + pc3x3 + pc1x1_2
        m1_cc = cc1x1_1 + cc3x3 + cc1x1_2
        print (' param', m1_pc, ' conn', m1_cc)
        
        return conv1x1_2
    
    def mn_v1(intrare, expand, strides, padding):
        
        # conv dw 3x3   BN + relu
        shape_c3x3 = [3,3,int(intrare.get_shape()[3]),1] 
        W3x3 = tf.Variable(x_initializer(shape_c3x3))
        b3x3 = tf.Variable(x_initializer(shape_c3x3[-1:]))
        c3x3 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(
                    tf.add(tf.nn.depthwise_conv2d(intrare, W3x3, strides=[1,strides,strides,1], padding=padding), b3x3), 
                    training=is_training, momentum=0.9, trainable=True))
        
        # conv pw 1x1   BN 
        shape_c1x1 = [1,1,int(c3x3.get_shape()[3]),expand] 
        W1x1 = tf.Variable(x_initializer(shape_c1x1))
        b1x1 = tf.Variable(x_initializer(shape_c1x1[-1:]))
        c1x1 = tf.nn.relu(tf.compat.v1.layers.batch_normalization(
                    tf.add(tf.nn.conv2d(c3x3, W1x1, strides=[1,1,1,1], padding="SAME"), b1x1), 
                    training=is_training, momentum=0.9, trainable=True))
        return c1x1
    

    # --------------------------------- C1 -----------------------------------------------------
    k1 = 3
    c1 = 3
    h1 = 64
    
    c1_W = tf.Variable(x_initializer([ 3, 3, 3, 64 ]))
    c1_b = tf.Variable(x_initializer([64]))
    conv1 = tf.add(tf.nn.conv2d(x_placeholder, c1_W, strides=[1,1,1,1], padding="VALID"), c1_b)
    c1_bn = tf.compat.v1.layers.batch_normalization(conv1, training=is_training, momentum=0.9, trainable=True)
    c1_bn_act = tf.nn.relu(c1_bn)
    pc1 =  (k1*k1*c1+1)*h1
    cc1 = (k1*k1*c1+1)*h1*int(c1_bn_act.get_shape()[1])*int(c1_bn_act.get_shape()[2])
    print('  param', pc1, '  conn', cc1 )
    print("\nC1:", c1_bn_act.get_shape())

    
    # ------------------------------- R1 ---------------------------------------
    r1 = mn_v1(intrare=c1_bn_act, expand=128, strides=1, padding="VALID")
    # ------------------------------ C2 - separabila ------------------------------------------------
    m1_s1 = mn_v2_reziduu(intrare=c1_bn_act, expand=128*2, squeeze=128, ksize=3, strides=1, padding="VALID", reziduu=r1) 
    print('m1_s1 + r', m1_s1.get_shape())
       
    
    # ------------------------------- R2 ---------------------------------------
    r2 = mn_v1(intrare=m1_s1, expand=256, strides=2, padding="SAME")   
    # ------------------------------ C2 - separabila ------------------------------------------------
    m2_s1 = mn_v2_reziduu(intrare=m1_s1, expand=256*2, squeeze=256, ksize=3, strides=2, padding="SAME", reziduu=r2) 
    print('m2_s1 + r', m2_s1.get_shape())
    
    
    # ------------------------------- R3 ---------------------------------------
    r3 = mn_v1(intrare=m2_s1, expand=256, strides=1, padding="VALID")
    # ------------------------------ C1 - separabila ------------------------------------------------
    m3_s1 = mn_v2_reziduu(intrare=m2_s1, expand=256*2, squeeze=256, ksize=3, strides=1, padding="VALID", reziduu=r3) 
    print('m3_s1 + r', m3_s1.get_shape())
    
            
    # ------------------------------- R4 ---------------------------------------
    r4 = mn_v1(intrare=m3_s1, expand=384, strides=1, padding="VALID")
    # ------------------------------ C1 - separabila ------------------------------------------------
    m4_s1 = mn_v2_reziduu(intrare=m3_s1, expand=384*2, squeeze=384, ksize=3, strides=1, padding="VALID", reziduu=r4) 
    print('m4_s1 + r', m4_s1.get_shape())   
    
    
    # ------------------------------- R5 ---------------------------------------
    r5 = mn_v1(intrare=m4_s1, expand=256, strides=2, padding="SAME")
    # ------------------------------ C5 - separabila ------------------------------------------------
    m5 = mn_v2_reziduu(intrare=m4_s1, expand=256*2, squeeze=256, ksize=3, strides=2, padding="SAME", reziduu=r5) 
    print('m5 + r', m5.get_shape()) 
    
    # ------------------------------- R5 ---------------------------------------
    r6 = mn_v1(intrare=m5, expand=256, strides=1, padding="VALID")
    # ------------------------------ C final - separabila ------------------------------------------------
    m6 = mn_v2_reziduu(intrare=m5, expand=256*2, squeeze=256, ksize=3, strides=1, padding="VALID", reziduu=r6) 
    print('m6 + r', m6.get_shape()) 
    
    # ------------------------------ C final - separabila ------------------------------------------------
    m7 = mn_v2_reziduu(intrare=m6, expand=256*2, squeeze=256, ksize=3, strides=1, padding="VALID", reziduu=0) 
    print('m7 + r', m7.get_shape()) 

    ### ----------------------------------- FLATTEN ---------------------------------------
    size = m7.get_shape()
    num_f = size[1:].num_elements()   
    
    
    embedding_p = tf.reshape(m7, [-1, num_f])
        
    weights_n = tf.Variable(x_initializer([num_f, num_classes]))
       
    embedding_norm = tf.norm(embedding_p, axis=1, keep_dims=True)  
    embedding = tf.divide(embedding_p, embedding_norm)
    
    weights_norm = tf.norm(weights_n, axis=0, keep_dims=True) 
    weights = tf.divide(weights_n, weights_norm)
    
    
    m = 0.5
    s = 64.
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    
    mm = sin_m * m
    
    threshold = math.cos(math.pi - m)
    
    cos_t = tf.matmul(embedding, weights)
    s_cos_t = tf.multiply(s, cos_t)
    
    cos_t2 = tf.square(cos_t)
    sin_t2 = tf.subtract(1., cos_t2)
    
    sin_t = tf.sqrt(sin_t2)
    cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m))
    cond_v = cos_t - threshold
    cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)
    keep_val = s*(cos_t - mm)
    cos_mt_temp = tf.where(cond, cos_mt, keep_val)
    mask = y_placeholder
    inv_mask = tf.subtract(1., mask)
    
    arcface = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask))
    
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arcface, labels=tf.argmax(y_placeholder, axis=1)))
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=0.1).minimize(loss=cost)
    
    y_pred = tf.argmax(tf.nn.softmax(cos_t), axis=1)
    
    acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, tf.argmax(y_placeholder, axis=1)), tf.float32))
    
    
    ## -------------------------------- FULLY CONNECTED - OUTPUT ------------------------------------
    pc_out = (256+1)*43
    print('  param', pc_out, '  conn', pc_out )
   

    # ------------------- SESIUNE -------------------------
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    num_batches_train = X_train.shape[0] // batch_size # numar de batchuri pentru train 
    num_batches_test = X_test.shape[0] // batch_size # numar de batchuri pentru test 
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops_n = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)


    with tf.control_dependencies(update_ops):
        
        t_train_start = time.time()
        
        ac_antrenare = np.empty(epochs)
        
        for e in range(epochs):
            
            start_optimizare = time.time()
            
            X_train[:], Y_train[:] = shuffle(X_train, Y_train)
            
            # ------------------------------ ANTRENARE ---------------------------------------------
            for i in range(num_batches_train):
                
                x_batch = X_train[i*batch_size:(i+1)*batch_size, :]
                y_batch = Y_train[i*batch_size:(i+1)*batch_size, :]
                
                feed_dict_train = {x_placeholder:x_batch, y_placeholder:y_batch, is_training:True}            
                sess.run([optimizer, update_ops_n], feed_dict=feed_dict_train)
            # -------------------------------------------------------------------------------------- 
                    
            
            # ------------------------------ ACURATETE ANTRENARE -----------------------------------
            acc_train_total = 0
            for i in range(num_batches_train):
                
                x_batch = X_train[i*batch_size:(i+1)*batch_size, :]
                y_batch = Y_train[i*batch_size:(i+1)*batch_size, :]
                
                feed_dict_train = {x_placeholder:x_batch, y_placeholder:y_batch, is_training:False}
                acc_train_batch = sess.run(acc, feed_dict=feed_dict_train)
                acc_train_total += acc_train_batch
                
            acc_train_total /= num_batches_train
            ac_antrenare[e] = acc_train_total
            #---------------------------------------------------------------------------------------
    
                          
            end_epoca = time.time()
            t_epoca = end_epoca - start_optimizare
            
            print('Epoca: ' + str(e) +
                  ' Ac.ant: ' + '{0:.4f}'.format(acc_train_total * 100) + ' %  ' +
                    # ' Ac.val: ' + '{0:.4f}'.format(acc_val_total * 100) + ' %  ' +
                  " Timp: " + '{0:.4f}'.format(t_epoca) + ' s')
            #---------------------------- SFARSIT EPOCA -------------------------------------------------
            
        t_train_stop = time.time()   
        timp_train = t_train_stop - t_train_start
        
        # ------------------------------ ACURATETE TESTARE ---------------------------------------------
        t_test_strat = time.time()
        acc_test_total = 0
        for i in range(num_batches_test):
            t_test_strat = time.time()
                
            x_batch = X_test[i*batch_size:(i+1)*batch_size, :]
            y_batch = Y_test[i*batch_size:(i+1)*batch_size, :]
                    
            feed_dict_test = {x_placeholder:x_batch, y_placeholder:y_batch,  is_training:False}
            acc_test_batch = sess.run(acc, feed_dict=feed_dict_test)
            acc_test_total += acc_test_batch
                
        acc_test_total /= num_batches_test
        t_test_stop = time.time()
        
        timp_test = t_test_stop - t_test_strat
        
        
        print('Acuratete antrenare: ', '{0:.4f}'.format(acc_train_total * 100), ' %')
        print('Acuratete testare: ', '{0:.4f}'.format(acc_test_total * 100), ' %')
        print('timp antrenare: ', timp_train)
        print('timp testare: ', timp_test)