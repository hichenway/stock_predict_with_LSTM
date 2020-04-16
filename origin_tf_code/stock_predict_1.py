#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

f=open('dataset_1.csv')  
df=pd.read_csv(f)     
data=np.array(df['max'])
data=data[::-1]      

#plt.figure()
#plt.plot(data)
#plt.show()
normalize_data=(data-np.mean(data))/np.std(data)  
normalize_data=normalize_data[:,np.newaxis]       


time_step=20      
rnn_unit=10
lstm_layers=2
batch_size=60     
input_size=1      
output_size=1     
lr=0.0006         
data_x,data_y=[],[]
for i in range(len(normalize_data)-time_step-1):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    data_x.append(x.tolist())
    data_y.append(y.tolist())

train_rate = 0.8
length = len(data_x)
train_x = data_x[:int(length*train_rate)]
train_y = data_y[:int(length*train_rate)]
test_x = data_x[int(length*train_rate):]
test_y = data_y[int(length*train_rate):]

X=tf.placeholder(tf.float32, [None,time_step,input_size])    
Y=tf.placeholder(tf.float32, [None,time_step,output_size])   

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }


def lstm(batch):      
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  
    input_rnn=tf.matmul(input,w_in)+b_in    # [?,10]
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])     # [?,20,10]
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)]*lstm_layers)
    # cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch,dtype=tf.float32)  # [60,10]
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    #[60,20,10],[60,10]
    output=tf.reshape(output_rnn,[-1,rnn_unit])
    # [1200,10]
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


def train_lstm():
    global batch_size
    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(batch_size)
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(1): #We can increase the number of iterations to gain better result.
            step=0
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                
                if step%10==0:
                    print("Number of iterations:",i," loss:",loss_)
                    print("model_save",saver.save(sess,'model_save1\\modle.ckpt'))
                    #I run the code on windows 10,so use  'model_save1\\modle.ckpt'
                    #if you run it on Linux,please use  'model_save1/modle.ckpt'
                step+=1
        print("The train has finished")
train_lstm()



def prediction():
    with tf.variable_scope("sec_lstm",reuse=tf.AUTO_REUSE):
        pred,_=lstm(1)    
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        saver.restore(sess, 'model_save1\\modle.ckpt') 
        #I run the code in windows 10,so use  'model_save1\\modle.ckpt'
        #if you run it in Linux,please use  'model_save1/modle.ckpt'

        # test
        predict_test = []
        for i in range(len(test_x)):
            test_pred = sess.run(pred,feed_dict={X:[test_x[i]]})
            predict_test.append(test_pred[-1])
        print(np.shape(test_pred),type(test_pred))
        print(test_pred[-1])
        # print(test_y[0],type(test_y[0]))

        # predict
        prev_seq=train_x[-1]
        predict=[]
        for i in range(100):
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            predict.append(next_seq[-1])
            prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
        
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.show()
        
prediction()
