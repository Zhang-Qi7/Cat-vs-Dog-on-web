from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications import Xception
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import Model
from tensorflow.python.keras.preprocessing.image import img_to_array

from tensorflow.python.keras.layers import (    
    Flatten,    
    Dense,    
    AveragePooling2D,    
    Dropout
)
from tensorflow.python.keras.callbacks import (    
    EarlyStopping,    
    ModelCheckpoint,    
    LearningRateScheduler
)
import tensorflow as tf

tf.flags.DEFINE_integer('batch_size', 8, 'batch size, default: 4')
tf.flags.DEFINE_integer('img_size', 224, 'square images acquired')
tf.flags.DEFINE_integer('epochs', 50, 'epochs, default: 10')
FLAGS = tf.flags.FLAGS


model = Xception(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), include_top=False, weights='imagenet')

x = model.output
x = AveragePooling2D(pool_size=(2, 2))(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.1)(x)
x = Flatten()(x)
x = Dense(2, activation='softmax', kernel_regularizer=l2(.0005))(x)

model = Model(inputs=model.inputs, outputs=x)
opt = SGD(lr=0.0001, momentum=.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255, rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, horizontal_flip=False)
train_generator = train_datagen.flow_from_directory('train/', target_size=(FLAGS.img_size, FLAGS.img_size), shuffle=True, batch_size=FLAGS.batch_size, class_mode='categorical',)
valid_generator = valid_datagen.flow_from_directory('valid/', target_size=(FLAGS.img_size, FLAGS.img_size), shuffle=True, batch_size=FLAGS.batch_size, class_mode='categorical',)


earlystop = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
checkpoint = ModelCheckpoint("model-weights/xception_checkpoint.h5", monitor="val_loss", mode="min", save_best_only=True, verbose=1)

history = model.fit_generator(train_generator,epochs=FLAGS.epochs,callbacks=[earlystop, checkpoint], validation_data=valid_generator)
model.save("model-weights/xception.h5")
