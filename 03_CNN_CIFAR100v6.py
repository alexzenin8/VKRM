


# ## Модель пятой нейронной сети для CIFAR100
#

# In[1]:

from __future__ import print_function
from __future__ import absolute_import

# Если используется  tensorflow, установить порядок измерений изображений
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")





import time
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
np.random.seed(2017) 
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout, merge, Lambda
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import warnings

from keras.optimizers import RMSprop
from keras.optimizers import Nadam
from keras import initializers
from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers.advanced_activations import PReLU
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.layers import  Input
from keras.utils import plot_model
from keras import callbacks

# Ограничение ресурсов, потребляемых картой GPU

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.874
set_session(tf.Session(config=config))

# Импорт датасета CIFAR100

from keras.datasets import cifar100
(train_features, train_labels), (test_features, test_labels) = cifar100.load_data(label_mode='fine')
num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  train_features.shape
num_classes = len(np.unique(train_labels))
print ("Number of training samples: %d"%train_features.shape[0])
print ("Number of test samples: %d"%test_features.shape[0])
print ("Image rows: %d"%train_features.shape[1])
print ("Image columns: %d"%train_features.shape[2])
print ("Number of classes: %d"%num_classes)


# Предобработка

train_features = train_features.astype('float32')/255
test_features = test_features.astype('float32')/255
# convert class labels to binary class labels
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)


# Функция построения истории обучения

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


# Функция, вычисляющая точность

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)




def do_scale(x, scale):
    y = scale * x 
    return y 

# Функция сверточного слоя с батч-нормализацией и активацией

def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1), 
              bias=False, activ_fn='elu', normalize=True):
    """
    Utility function to apply conv + BN. 
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    if not normalize:
        bias = True
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      border_mode=border_mode,
                      bias=bias,init='he_normal')(x)
    if normalize:
        x = BatchNormalization(axis=channel_axis)(x)
    if activ_fn:
        x = Activation(activ_fn)(x)
    return x


# Функция блока сверток 32х32

def block32(input, scale=1.0, activation_fn='elu'):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    shortcut = input

    tower_conv = conv2d_bn(input, 32, 1, 1, activ_fn=activation_fn)

    tower_conv1_0 = conv2d_bn(input, 32, 1, 1, activ_fn=activation_fn)
    tower_conv1_1 = conv2d_bn(tower_conv1_0, 32, 3, 3, activ_fn=activation_fn)

    tower_conv2_0 = conv2d_bn(input, 32, 1, 1, activ_fn=activation_fn)
    tower_conv2_1 = conv2d_bn(tower_conv2_0, 48, 3, 3, activ_fn=activation_fn)
    tower_conv2_2 = conv2d_bn(tower_conv2_1, 64, 3, 3, activ_fn=activation_fn)

    mixed = merge([tower_conv, tower_conv1_1, tower_conv2_2], mode='concat', concat_axis=channel_axis)

    up = conv2d_bn(mixed, 320, 1, 1, activ_fn=False, normalize=False)

    up = Lambda(do_scale, output_shape=K.int_shape(up)[1:], arguments={'scale':scale})(up)

    net = merge([shortcut, up], mode='sum')

    if activation_fn:
        net = Activation(activation_fn)(net)
    return net

# Функция блока сверток 16х16

def block16(input, scale=1.0, activation_fn='elu'):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    shortcut = input

    tower_conv = conv2d_bn(input, 192, 1, 1, activ_fn=activation_fn)

    tower_conv1_0 = conv2d_bn(input, 128, 1, 1, activ_fn=activation_fn)
    tower_conv1_1 = conv2d_bn(tower_conv1_0, 160, 1, 7, activ_fn=activation_fn)
    tower_conv1_2 = conv2d_bn(tower_conv1_1, 192, 7, 1, activ_fn=activation_fn)

    mixed = merge([tower_conv, tower_conv1_2], mode='concat', concat_axis=channel_axis)

    up = conv2d_bn(mixed, 1088, 1, 1, activ_fn=False, normalize=False)

    up = Lambda(do_scale, output_shape=K.int_shape(up)[1:], arguments={'scale':scale})(up)

    net = merge([shortcut, up], mode='sum')

    if activation_fn:
        net = Activation(activation_fn)(net)
    return net

# Функция блока сверток 7х7

def block8(input, scale=1.0, activation_fn='elu'):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    shortcut = input

    tower_conv = conv2d_bn(input, 192, 1, 1, activ_fn=activation_fn)

    tower_conv1_0 = conv2d_bn(input, 192, 1, 1, activ_fn=activation_fn)
    tower_conv1_1 = conv2d_bn(tower_conv1_0, 224, 1, 3, activ_fn=activation_fn)
    tower_conv1_2 = conv2d_bn(tower_conv1_1, 256, 3, 1, activ_fn=activation_fn)

    mixed = merge([tower_conv, tower_conv1_2], mode='concat', concat_axis=channel_axis)

    up = conv2d_bn(mixed, 2080, 1, 1, activ_fn=False, normalize=False)

    up = Lambda(do_scale, output_shape=K.int_shape(up)[1:], arguments={'scale':scale})(up)

    net = merge([shortcut, up], mode='sum')

    if activation_fn:
        net = Activation(activation_fn)(net)
    return net


# Инициализация констант


input_img = Input(shape=(3, 32, 32))
channel_axis = 1
num_classes=100
dropout_keep_prob=0.8
    


## Первый параллельный блок сверток

tower_conv = conv2d_bn(input_img, 96, 1, 1)

tower_conv1_0 = conv2d_bn(input_img, 48, 1, 1)
tower_conv1_1 = conv2d_bn(tower_conv1_0, 64, 5, 5)

tower_conv2_0 = conv2d_bn(input_img, 64, 1, 1)
tower_conv2_1 = conv2d_bn(tower_conv2_0, 96, 3, 3)
tower_conv2_2 = conv2d_bn(tower_conv2_1, 96, 3, 3)

tower_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input_img)
tower_pool_1 = conv2d_bn(tower_pool, 64, 1, 1)

net = merge([tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], mode='concat', concat_axis=channel_axis)

# 10 x block32

for idx in xrange(10):
    net = block32(net, scale=0.17)


## Первый уменьшающий размерность блок сверток


tower_conv = conv2d_bn(net, 384, 3, 3, subsample=(2,2), border_mode='valid')

tower_conv1_0 = conv2d_bn(net, 256, 1, 1)
tower_conv1_1 = conv2d_bn(tower_conv1_0, 256, 3, 3)
tower_conv1_2 = conv2d_bn(tower_conv1_1, 384, 3, 3, subsample=(2,2), border_mode='valid')

tower_pool = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(net)

net = merge([tower_conv, tower_conv1_2, tower_pool], mode='concat', concat_axis=channel_axis)

# 20 x block16

for idx in xrange(20):
    net = block16(net, scale=0.10)


## Второй уменьшающий размерность блок сверток

tower_conv = conv2d_bn(net, 256, 1, 1)
tower_conv_1 = conv2d_bn(tower_conv, 384, 3, 3, subsample=(2,2), border_mode='valid')

tower_conv1 = conv2d_bn(net, 256, 1, 1)
tower_conv1_1 = conv2d_bn(tower_conv1, 288, 3, 3, subsample=(2,2), border_mode='valid')

tower_conv2 = conv2d_bn(net, 256, 1, 1)
tower_conv2_1 = conv2d_bn(tower_conv2, 288, 3, 3)
tower_conv2_2 = conv2d_bn(tower_conv2_1, 320, 3, 3, subsample=(2,2), border_mode='valid')

tower_pool = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(net)

net = merge([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], mode='concat', concat_axis=channel_axis)

    # 9 x block7

for idx in xrange(9):
    net = block8(net, scale=0.20)
net = block8(net, activation_fn=False)


# Завершающая последовательность слоев

net = conv2d_bn(net, 1536, 1, 1)
net = AveragePooling2D((7,7), border_mode='valid')(net)

net = Flatten()(net)
net = Dropout(dropout_keep_prob)(net)

predictions = Dense(output_dim=num_classes, activation='softmax')(net)
    
    

# Задания алгоритма градиентного спуска

optim = Nadam(lr=0.009,epsilon=1.0)        
model=Model(input_img,predictions, name='inception_resnetv2')

model.load_weights("/home/zenin/weights100v6.69-1.09.hdf5")

# Компиляция модели

model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

# Вывод на экран информации о модели

model.summary()


#Создание генератора аугментированных изображений

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zoom_range=0.2, 
                             horizontal_flip=True,
                             rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             fill_mode='nearest'
                             )
datagen.fit(train_features, seed=2017)

# Создание callbacks,TensorBoard - пишет логи и строит графики, checkpointer - сохранение весов, early_stopper -ранняя остановка, lr_reducer - уменьшение шага обучение в определенных условиях.

tensorboardcall=callbacks.TensorBoard(log_dir='/home/zenin/logs100v6',write_graph=True, write_images=True,
                                      embeddings_freq=50, embeddings_layer_names=None, embeddings_metadata=None)

checkpointer = callbacks.ModelCheckpoint(filepath="/home/zenin/weights100v6.{epoch:02d}-{val_loss:.2f}.hdf5",monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=5)

early_stopper   = callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)

lr_reducer      = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                    cooldown=0, patience=6, min_lr=0.5e-6)
start = time.time()

# Старт обучения

model_info = model.fit_generator(datagen.flow(train_features, train_labels, batch_size = 40),
                                 samples_per_epoch = train_features.shape[0], epochs = 200,
                                 validation_data = (test_features, test_labels),initial_epoch=70, callbacks=[tensorboardcall,checkpointer,early_stopper,lr_reducer])
end = time.time()

#Вывод результатов на экран

print ("Model took %0.2f seconds to train"%(end - start))
print ("Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model))

