import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np

weight_decay = 0.0005
momentum = 0.9
CLASSES_NUM = 12
init_learning_rate = 0.1
cardinality = 8 # how many split ?
blocks = 3 # res_block ! (split + transition)
depth = 32 # out channel

"""
So, the total number of layers is (3*blokcs)*residual_layer_num + 2
because, blocks = split(conv 2) + transition(conv 1) = 3 layer
and, first conv layer 1, last dense layer 1
thus, total number of layers = (3*blocks)*residual_layer_num + 2
"""





class SE_ResNeXt():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_SEnet(x)
        self.reduction_ratio = 4

    def conv_layer(self,input, filter, kernel, stride, padding='SAME', layer_name="conv"):
        with tf.name_scope(layer_name):
            network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                       padding=padding)
            return network

    def Global_Average_Pooling(self,x):
        return global_avg_pool(x, name='Global_avg_pooling')

    def Average_pooling(self,x, pool_size=[2, 2], stride=2, padding='SAME'):
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    def Batch_Normalization(self,x, training, scope):
        with arg_scope([batch_norm],
                       scope=scope,
                       updates_collections=None,
                       decay=0.9,
                       center=True,
                       scale=True,
                       zero_debias_moving_mean=True):
            return tf.cond(training,
                           lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                           lambda: batch_norm(inputs=x, is_training=training, reuse=True))

    def Relu(self,x):
        return tf.nn.relu(x)

    def Sigmoid(self,x):
        return tf.nn.sigmoid(x)

    def Concatenation(self,layers):
        return tf.concat(layers, axis=3)

    def Fully_connected(self,x, units=CLASSES_NUM, layer_name='fully_connected'):
        with tf.name_scope(layer_name):
            return tf.layers.dense(inputs=x, use_bias=False, units=units)

    def first_layer(self, x, scope):
        with tf.name_scope(scope) :
            x = self.conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name=scope+'_conv1')
            x = self.Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = self.Relu(x)

            return x

    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope) :
            x = self.conv_layer(x, filter=depth, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = self.Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = self.Relu(x)

            x = self.conv_layer(x, filter=depth, kernel=[3,3], stride=stride, layer_name=scope+'_conv2')
            x = self.Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = self.Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = self.conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = self.Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            # x = Relu(x)

            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name) :
            layers_split = list()
            for i in range(cardinality) :
                splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return self.Concatenation(layers_split)

    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :


            squeeze = self.Global_Average_Pooling(input_x)

            excitation = self.Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = self.Relu(excitation)
            excitation = self.Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = self.Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation

            return scale

    def residual_layer(self, input_x, out_dim, layer_num, res_block=blocks):
        # split + transform(bottleneck) + transition + merge
        # input_dim = input_x.get_shape().as_list()[-1]

        for i in range(res_block):
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1

            x = self.split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))
            x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=self.reduction_ratio, layer_name='squeeze_layer_'+layer_num+'_'+str(i))

            if flag is True :
                pad_input_x = self.Average_pooling(input_x)
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]]) # [?, height, width, channel]
            else :
                pad_input_x = input_x

            input_x = self.Relu(x + pad_input_x)

        return input_x


    def Build_SEnet(self, input_x):
        # only cifar10 architecture

        input_x = self.first_layer(input_x, scope='first_layer')

        x = self.residual_layer(input_x, out_dim=64, layer_num='1')
        x = self.residual_layer(x, out_dim=128, layer_num='2')
        x = self.residual_layer(x, out_dim=256, layer_num='3')

        x = self.Global_Average_Pooling(x)
        x = flatten(x)

        x = self.Fully_connected(x, layer_name='final_fully_connected')
        return x