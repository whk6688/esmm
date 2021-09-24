import os
import time
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Reshape, Lambda, concatenate, dot, add, Multiply
from keras.layers import Dropout, GaussianDropout, multiply, SpatialDropout1D, BatchNormalization
from keras.models import Model
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.utils import shuffle
import datetime
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.models import  save_model,load_model

class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_CTR_auc")*logs.get("val_multiply_1_auc")
        if np.greater(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)




os.environ['CUDA_VISIBLE_DEVICES'] = '1'
date_type={'airQuality' : np.int16, 'area' : np.int16,  'ctr' : np.int16,  'cvr' : np.int16}
df_chunk=pd.read_csv("/data/home/wanghk/data6/train_1.csv",sep = ';',dtype=date_type, chunksize=100000)

res_chunk=[]
for chunk in df_chunk:
    res_chunk.append(chunk)
train_df=pd.concat(res_chunk)
train_size=len(train_df)
print("load data end", train_size, datetime.datetime.now())
train_df = shuffle(train_df)
print(train_df)
sparse_features = ['airQuality','area']

dense_features = ['roomPrice']

ctr = train_df['ctr'].values
cvr = train_df['cvr'].values

cols = sparse_features+dense_features
def base_model():
    field_input = []
    field_embedding = []
    dense_input=[]
    for cat in cols:
        if cat in sparse_features:
            input = Input(shape=(1,), name=cat)
            field_input.append(input)
            nums = train_df[cat].max() + 3
            # ffm embeddings
            x_embed = Embedding(nums, 8, input_length=1, trainable=True)(input)
            x_reshape = Reshape((10,))(x_embed)
            field_embedding.append(x_reshape)
        else:
            dense = Input(shape=[1])  # None*1
            field_input.append(dense)
            dense_input.append(dense)


    embed_layer_1 = concatenate(field_embedding+dense_input, axis=-1)


    #######dnn layer##########

    embed_layer = Dense(256)(embed_layer_1)
    #embed_layer = BatchNormalization()(embed_layer)
    embed_layer = Activation('relu')(embed_layer)
    embed_layer = Dropout(rate=0.2)(embed_layer)

    embed_layer = Dense(128)(embed_layer)
    #embed_layer = BatchNormalization()(embed_layer)
    embed_layer = Activation('relu')(embed_layer)
    embed_layer = Dropout(rate=0.2)(embed_layer)

    embed_layer = Dense(64)(embed_layer)
    #embed_layer = BatchNormalization()(embed_layer)
    embed_layer = Activation('relu')(embed_layer)
    embed_layer = Dropout(rate=0.2)(embed_layer)

    embed_layer = Dense(1)(embed_layer)

    embed_layer_cvr = Dense(256)(embed_layer_1)
    #embed_layer_cvr = BatchNormalization()(embed_layer_cvr)
    embed_layer_cvr = Activation('relu')(embed_layer_cvr)
    embed_layer_cvr = Dropout(rate=0.2)(embed_layer_cvr)

    embed_layer_cvr = Dense(128)(embed_layer_cvr)
    #embed_layer_cvr = BatchNormalization()(embed_layer_cvr)
    embed_layer_cvr = Activation('relu')(embed_layer_cvr)
    embed_layer_cvr = Dropout(rate=0.2)(embed_layer_cvr)

    embed_layer_cvr = Dense(64)(embed_layer_cvr)
    #embed_layer_cvr = BatchNormalization()(embed_layer_cvr)
    embed_layer_cvr = Activation('relu')(embed_layer_cvr)
    embed_layer_cvr = Dropout(rate=0.2)(embed_layer_cvr)

    embed_layer_cvr = Dense(1)(embed_layer_cvr)


    ctr_predictions = Activation('sigmoid',name="CTR")(embed_layer)
    cvr_predictions = Activation('sigmoid',name="CVR")(embed_layer_cvr)
    prop = Multiply()([ctr_predictions,cvr_predictions])

    # ctr_predictions = tf.layers.dense(embed_layer,1, activation='sigmoid', name="CTR")
    # cvr_predictions = tf.layers.dense(embed_layer_cvr, 1, activation='sigmoid', name="CTR")
    # prop = tf.multiply(ctr_predictions, cvr_predictions, name="ctcvr_output")

    ctcvr_model = Model(
        inputs=field_input,
        outputs=[ctr_predictions, prop])

    opt = Adam(0.001)
    ctcvr_model.compile(optimizer=opt, loss=["binary_crossentropy", "binary_crossentropy"],
                        loss_weights=[1.0, 1.0],
                        metrics=[auc])
    return ctcvr_model
# training########################################


x_train = train_df[cols].values
x_train = list(x_train.T)

#es = EarlyStopping(monitor='val_auc',mode='max',patience=3)
#checkpoint = ModelCheckpoint( file_path, save_weights_only=True, verbose=1, save_best_only=True)
model = base_model()
model.fit(x_train, [ctr,cvr],
          batch_size=10000,
          epochs=20,
          validation_split=0.2, verbose=2, shuffle=True, callbacks=[EarlyStoppingAtMinLoss(3)])

model.save('esmm_model.h5')
test_df=pd.read_csv("/data/home/wanghk/data6/test_1.csv",sep = ';',dtype=date_type)
x_test = test_df[cols].values
x_test = list(x_test.T)
pred_ans = model.predict(x_test, batch_size=1000)
print("test ctr-AUC", round(roc_auc_score(test_df["ctr"].values, pred_ans[0]), 4))
print("test cvr-AUC", round(roc_auc_score(test_df["cvr"].values, pred_ans[1]), 4))
print("end:", datetime.datetime.now())

