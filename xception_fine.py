
# coding: UTF-8
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import CSVLogger

from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import os,random
from keras.preprocessing.image import img_to_array, load_img
import os
import keras.models

from keras.callbacks import ModelCheckpoint

batch_size=32
train_dir=r'C:\Users\3ken\Desktop\NiS\data\train'
validation_dir=r'C:\Users\3ken\Desktop\NiS\data\validation'
test_dir=r'C:\Users\3ken\Desktop\NiS\data\test'
display_dir=r'C:\Users\3ken\Desktop\NiS\data\display'

label=os.listdir(test_dir)
#['homura','kyoko','madoka','mami','sayaka']


n_categories = len(label)

print(label)
print(n_categories)

n_epochs = 450
file_name='xception_NiS_' + str(n_categories) + '_' + str(n_epochs) + 'eps_'

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
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator=validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)


model_checkpoint = ModelCheckpoint(
    filepath=os.path.join('model', 'model_{epoch:02d}_{val_loss:.2f}.h5'),
    monitor='val_loss',
    period=10,
    verbose=1)

'''    
hist=model.fit_generator(train_generator,
                         epochs=n_epochs,
                         verbose=1,
                         validation_data=validation_generator,
                         callbacks=[model_checkpoint, CSVLogger(file_name+'.csv')])
'''




#batch_size=32

#load model and weights

model = keras.models.load_model('best_model/' + 'model_450_0.51.h5')
model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

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
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

#evaluate model
score=model.evaluate_generator(test_generator)
print('\n test loss:',score[0])
print('\n test_acc:',score[1])

#save weights
model.save('model/' +file_name + '_' + str(round(score[1],2))+ '.h5')

#predict model and display images
files=os.listdir(display_dir)

n_display = min(49, len(files))
img=random.sample(files,n_display)

plt.figure(figsize=(10,10))

for i in range(n_display):
    temp_img=load_img(os.path.join(display_dir,img[i]),target_size=(224,224))
    plt.subplot(5,7,i+1)
    plt.imshow(temp_img)
    #Images normalization
    temp_img_array=img_to_array(temp_img)
    temp_img_array=temp_img_array.astype('float32')/255.0
    temp_img_array=temp_img_array.reshape((1,224,224,3))
    #predict image
    img_pred=model.predict(temp_img_array)
    #print(str(round(max(img_pred[0]),2)))
    plt.title(label[np.argmax(img_pred)] + str(round(max(img_pred[0]),2)))
    #eliminate xticks,yticks
    plt.xticks([]),plt.yticks([])


plt.show()