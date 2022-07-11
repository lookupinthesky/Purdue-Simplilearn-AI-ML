import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow import keras
import os
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array,ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l1, l2, l1_l2
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from livelossplot import PlotLossesKerasTF



@tf.function
def soft_f1_macro(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels

    return macro_cost

@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


def create_model():


    model = Sequential([
        Conv2D(64,kernel_size=(3,3), activation = 'relu',padding='same',input_shape=(224,224,3)),   #512
        Conv2D(64,kernel_size=(3,3), activation = 'relu',padding='same'),
        MaxPooling2D(pool_size=(2,2),strides=2),                                                    #256
        BatchNormalization(),
        Conv2D(128,kernel_size=(3,3), activation = 'relu',padding='same' ),
        Conv2D(128,kernel_size=(3,3), activation = 'relu',padding='same' ),
        MaxPooling2D(pool_size=(2,2),strides=2),                                                    #128
        BatchNormalization(),
        Conv2D(256,kernel_size=(3,3), activation = 'relu',padding='same' ),
        Conv2D(256,kernel_size=(3,3), activation = 'relu',padding='same' ),
#         Conv2D(256,kernel_size=(3,3), activation = 'relu',padding='same'),
        MaxPooling2D(pool_size=(2,2),strides=2),                                                    #64
        BatchNormalization(),
        Conv2D(512,kernel_size=(3,3), activation = 'relu',padding='same',kernel_regularizer=l2(1) ),
        Conv2D(512,kernel_size=(3,3), activation = 'relu',padding='same',kernel_regularizer=l2(1) ),
        MaxPooling2D(pool_size=(2,2),strides=2),                                                    #32
        BatchNormalization(),
#         Conv2D(1024,kernel_size=(3,3), activation = 'relu',padding='same'),
        #Conv2D(1024,kernel_size=(3,3), activation = 'relu',padding='same'),
#         MaxPooling2D(pool_size=(2,2),strides=2),                                                    #16
#         BatchNormalization(),
        Flatten(),
        Dense(1024, activation='relu',kernel_regularizer=l1(0.001) ),
        Dropout(0.2),
        Dense(1024,activation='relu',kernel_regularizer=l1(0.001)),
        Dense(4,activation='softmax')        
        ])

    return model


def get_training_data(train_dir):
    X_train = np.load(os.path.join(train_dir,'X','trainX.npy'))
    y_train = np.load(os.path.join(train_dir,'y','trainy.npy'))
    return X_train, y_train
    
def get_validation_data(val_dir):
    X_val = np.load(os.path.join(val_dir,'X','valX.npy'))
    y_val = np.load(os.path.join(val_dir,'y','valy.npy'))
    return X_val, y_val


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm_model_dir', type=str, default=os.environ.get('SM_MODEL_DIR')) #/opt/ml/model/
    parser.add_argument('--training', type=str, default=os.environ.get('SM_CHANNEL_TRAINDATA')) #as defined in fit
    parser.add_argument('--validation',type=str, default=os.environ.get('SM_CHANNEL_VALDATA')) # as defined in fit
    parser.add_argument("--checkpoint_path",type=str,default="/opt/ml/checkpoints",help="Path where checkpoints will be saved.") #automatically syncs with s3
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current_host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__=="__main__":

    # Define accelerated strategy to run on any Cloud environment - Kaggle, Colab and Sagemaker
    # If TPU is available use it, else use Mirrored Strategy

    try:
        # tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR'] # uncomment for colab
        #tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address) # TPU detection # uncomment for colab
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection | uncomment for Kaggle, Sagemaker
        tf.config.experimental_connect_to_cluster(tpu) 
        tf.tpu.experimental.initialize_tpu_system(tpu) 

        # Alternatively
        # tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

        strategy = tf.distribute.experimental.TPUStrategy(tpu) 
        # Going back and forth between TPU and host is expensive.
        # Better to run 128 batches on the TPU before reporting back.
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  
        print("Number of accelerators: ", strategy.num_replicas_in_sync)

    except ValueError:
        print('TPU failed to initialize.')
        print("Utilizing GPUs using Mirrored Strategy")
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))



    
    args, unknown = _parse_args()
    train_images, train_labels = get_training_data(args.training)
    val_images, val_labels = get_validation_data(args.validation)

    # Normalize the images to [0, 1] range.
    train_images = train_images / np.float32(255)
    val_images = val_images / np.float32(255)

    # Batch the input data
    BUFFER_SIZE = train_images.shape[0]
    BATCH_SIZE_PER_REPLICA = 16
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync  

    # Create Datasets from the batches
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    # Create Distributed Datasets from the datasets
    # Commenting because : use experimental_data_distribute only for custom training
    # train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    # test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


    with strategy.scope():
        model = create_model()
        model.compile(optimizer=Adam(learning_rate=1e-6), loss='categorical_crossentropy', metrics = ['accuracy'])

    steps_per_epoch = BUFFER_SIZE//GLOBAL_BATCH_SIZE
    
    checkpoint_path = args.checkpoint_path
    #checkpoint_path = os.path.join(args.sm_model_dir, 'v0', 'dr_model.h5') 

    # If above doesn't work try this
#     checkpoint_dir = './checkpoints'
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#     checkpoint_path = os.path.join(checkpoint_dir, 'dr_model.h5')

#     print(checkpoint_path)

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')




    early_stopping = EarlyStopping(monitor='val_accuracy',patience=10, mode='max')
    plotlosseskeras = PlotLossesKerasTF()
#     callbacks = [checkpoint,early_stopping,plotlosseskeras] <- Plotlosseskeras doesn't render on sagemaker logs so omit
    callbacks = [checkpoint]
    epochs = 2 # using only 2 epochs because GPU aren't available

    history = model.fit(train_dataset, validation_data = val_dataset, epochs=epochs, callbacks=callbacks)
    
    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        model.save(args.sm_model_dir, 'dr_model.h5')

    # if plotlosseskeras doesn't work try this

    # plt.subplots(2,2,1)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')

    # plt.subplots(2,2,2)
    # plt.plot(history.history['f1_macro'])
    # plt.plot(history.history['val_f1_macro'])
    # plt.title('F1 Score')
    # plt.ylabel('F1 Score')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')

    # plt.subplots(2,2,3)
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')

    # plt.show()



    

    


    

    
    







    











