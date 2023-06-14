import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# 定义学习率衰减函数
def lr_decay(epoch):
    initial_lr = 0.0012
    if epoch < 20:
        return initial_lr
    else:
        return initial_lr * tf.math.exp(0.008 * (20 - epoch))

#加载MNIST数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 定义回调函数
checkpoint = ModelCheckpoint(filepath='model_weights.h5', 
                             save_best_only=True,
                             save_weights_only=True, 
                             monitor='val_accuracy', 
                             mode='max', 
                             verbose=1, 
                             period=1)

# 数据增强
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1,
                             fill_mode='nearest')

# 定义学习率衰减回调函数
lr_scheduler = LearningRateScheduler(lr_decay)

# 创建模型
model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.6))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型并添加回调函数和学习率衰减回调函数
history = model.fit(datagen.flow(x_train, y_train, batch_size=512), epochs=60, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpoint, lr_scheduler])

# 评估模型
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])