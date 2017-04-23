from keras.models import Model
import resnetsmall
from keras.optimizers import Nadam
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import ModelCheckpoint
import os
from keras.preprocessing.image import ImageDataGenerator

def get_session(gpu_fraction=0.4):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

img_width, img_height = 96, 96

train_data_dir = '/data/dataset-96-red2/train'
validation_data_dir = '/data/dataset-96-red2/validation'
nb_train_samples = 50864
nb_validation_samples = 12828


nb_filters=16
model = resnetsmall.ResnetBuilder.build_resnet_c((3, 96, 96), 2,nb_filters)

nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        zoom_range=0.2,
        rotation_range=180,
        vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode='categorical')
checkpoint =ModelCheckpoint("models/hyperres/60kset/custom3.{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]

model.fit_generator(
        train_generator,
        samples_per_epoch=2048,
        nb_epoch=2000,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=callbacks_list
        )
print(model.summary())
