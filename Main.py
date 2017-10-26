# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import Models
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(2017)
img_rows = 75
img_cols = 75
color_type = 3
batch_size = 128
nb_epoch = 150
split = 0.2

path = '../Statoil/'
Train_data = pd.read_json(path +"/train.json")
Test_data = pd.read_json(path +"/test.json")



# Train data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in Train_data["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in Train_data["band_2"]])
Train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis], ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)
Target_train = np.array(Train_data["is_iceberg"])
print Target_train.shape
print("Totla Xtrain:", Train.shape)

# Test data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in Test_data["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in Test_data["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis], ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)
print("Total Xtest:", X_test.shape)
print "\n"
i = 0
while(i < 5):
    print "Fold number : ", i+1
    X_train, X_validate, y_train, y_validate = train_test_split(Train, Target_train, test_size=split,
                                                                                random_state=i)

    model = Models.CNN_NCouches(img_rows, img_cols, color_type)
    checkpointer = ModelCheckpoint(filepath="weightsOil"+str(i)+".hdf5", verbose=1, save_best_only=True)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_split=0.2, shuffle=True, callbacks=[checkpointer])

    # Make predictions
    prediction = model.predict(X_test, verbose=1)
    submit_Statoil = pd.DataFrame({'id': Test_data["id"], 'is_iceberg': prediction.flatten()})
    submit_Statoil.to_csv("./Statoil_submission"+str(i+1)+".csv", index=False)
    i += 1










