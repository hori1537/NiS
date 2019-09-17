
#© 2019 Horie Yuki
#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()

#print(device_lib.list_local_devices())

# coding: UTF-8
import keras.models
from keras.models import Model
from keras.models import Sequential

#from keras.models import model_from_json

from keras.layers import Dense, Input, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import MaxPool2D #MAxPool2D equal MaxPooling2D

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import os, random


def chk_mkdir(paths):
    for path_name in paths:
        print(path_name)
        if os.path.exists(path_name) == False:
            os.mkdir(path_name)
    return


batch_size=32
input_image_size = 224

base_dir        =r'C:\Users\3ken\Desktop\data_science\pictures\actors'
#base_dir        =r'drive/My Drive/actors'
theme_name      = os.path.basename(os.path.dirname(base_dir))

train_dir       = base_dir + r'\data\train'
validation_dir  = base_dir + r'\data\validation'
test_dir        = base_dir + r'\data\test'
display_dir     = base_dir + r'\data\display'


paths = [base_dir + os.sep + 'model',
         base_dir + os.sep + 'best_model',
         base_dir + os.sep + 'model'+ os.sep + 'checkpoint',
         base_dir + os.sep + 'model'+ os.sep + 'csvlogger',
            ]

chk_mkdir(paths)

label=os.listdir(test_dir)
n_categories = len(label)

print(label)
print(n_categories)

n_epochs = 300
net_type = 'xception'
#net_type = 'mobilenet'

file_name = net_type +'_'  + theme_name +'_category_' + str(n_categories) + '_' + str(n_epochs) + 'eps_'

def create_original_model():

    model = Sequential() # model = keras.models.Sequential()と等価

    model.add(Conv2D(32,3,input_shape=(input_image_size,input_image_size,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32,3))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(64,3))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(1.0))

    model.add(Dense(n_categories, activation='softmax'))

    return model

def create_xception():
    base_model = Xception(
        include_top = False,
        weights = "imagenet",
        input_shape = None
    )

    #add new layers instead of FC networks

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation = 'relu')(x)
    predictions = Dense(n_categories, activation = 'softmax')(x)

    # ネットワーク定義
    model = Model(inputs = base_model.input, outputs = predictions)
    print("{}層".format(len(model.layers)))


    #108層までfreeze
    for layer in model.layers[:108]:
        layer.trainable = False

        # Batch Normalization の freeze解除
        if layer.name.startswith('batch_normalization'):
            layer.trainable = True
        if layer.name.endswith('bn'):
            layer.trainable = True

    #109層以降、学習させる
    for layer in model.layers[108:]:
        layer.trainable = True
    return model

def create_mobilenet():
    base_model = MobileNet(
    include_top = False,
    weights = "imagenet",
    input_shape = None
    )   

    # 全結合層の新規構築
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation = 'relu')(x)
    predictions = Dense(n_categories, activation = 'softmax')(x)

    # ネットワーク定義
    model = Model(inputs = base_model.input, outputs = predictions)
    print("{}層".format(len(model.layers)))

    # 72層までfreeze
    for layer in model.layers[:72]:
        layer.trainable = False

        # Batch Normalization の freeze解除
        if "bn" in layer.name:
            layer.trainable = True

    #73層以降、学習させる
    for layer in model.layers[72:]:
        layer.trainable = True

    return model


def create_vgg16():
    base_model = VGG16(
    include_top = False,
    weights = "imagenet",
    input_shape = None
    )

    # 全結合層の新規構築
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation = 'relu')(x)
    predictions = Dense(n_categories, activation = 'softmax')(x)

    # ネットワーク定義
    model = Model(inputs = base_model.input, outputs = predictions)
    print("{}層".format(len(model.layers)))

        # 17層までfreeze
    for layer in model.layers[:17]:
        layer.trainable = False

    # 18層以降、学習させる
    for layer in model.layers[17:]:
        layer.trainable = True

    return model

if net_type == 'xception':
    model = create_xception()
elif net_type == 'mobilenet':
    model = create_mobilenet()
elif net_type == 'original':
    model = create_original_model()
elif net_type == 'vgg16':
    model = create_vgg16()
else:
    model = create_vgg16()

# layer.trainableの設定後にcompile
model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ["accuracy"]
)

model.summary()

train_datagen=ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=90,
    vertical_flip=True,
    horizontal_flip=True,
    height_shift_range=0.5,
    width_shift_range=0.5,
    channel_shift_range=5.0,
    brightness_range=[0.3,1.0],
    fill_mode='nearest'
    )

validation_datagen=ImageDataGenerator(rescale=1.0/255)

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(input_image_size,input_image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator=validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(input_image_size,input_image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)


model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(base_dir + os.sep + 'model'+ os.sep + 'checkpoint' + os.sep + net_type + '_model_{epoch:02d}_{val_loss:.2f}.h5'),
    monitor='val_loss',
    period=10,
    verbose=1)

print(net_type)
hist=model.fit_generator(train_generator,
                         epochs=n_epochs,
                         verbose=1,
                         validation_data=validation_generator,
                         callbacks=[model_checkpoint, CSVLogger(base_dir + os.sep + 'model'+ os.sep + 'csvlogger' + os.sep  + file_name+'.csv')])



#load model and weights
#model = keras.models.load_model(base_dir + 'model/' +file_name + '_' + str(round(score[1],2))+ '.h5')

#model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])


#data generate
test_datagen=ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=90,
    vertical_flip=True,
    horizontal_flip=True,
    height_shift_range=0.5,
    width_shift_range=0.5,
    channel_shift_range=5.0,
    brightness_range=[0.3,1.0],
    fill_mode='nearest'
    )

test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(input_image_size,input_image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

#evaluate model
score=model.evaluate_generator(test_generator)
print('\n test loss:',score[0])
print('\n test_acc:',score[1])

#save weights
model.save(base_dir + os.sep + 'model'+ os.sep +file_name + '_' + str(round(score[1],2))+ '.h5')

#predict model and display images
files=os.listdir(display_dir)

n_display = min(49, len(files))
img=random.sample(files,n_display)

plt.figure(figsize=(10,10))

for i in range(n_display):
    temp_img=load_img(os.path.join(display_dir,img[i]),target_size=(input_image_size,input_image_size))
    plt.subplot(5,7,i+1)
    plt.imshow(temp_img)
    #Images normalization
    temp_img_array=img_to_array(temp_img)
    temp_img_array=temp_img_array.astype('float32')/255.0
    temp_img_array=temp_img_array.reshape((1,input_image_size,input_image_size,3))
    #predict image
    img_pred=model.predict(temp_img_array)
    #print(str(round(max(img_pred[0]),2)))
    plt.title(label[np.argmax(img_pred)] + str(round(max(img_pred[0]),2)))
    #eliminate xticks,yticks
    plt.xticks([]),plt.yticks([])


plt.show()
