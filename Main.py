# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import Models
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold

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

Results = []
evaluation = []

# Load images
def load_data(data):
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_2"]])
    return np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis], ((x_band1 + x_band1) / 2)[:, :, :, np.newaxis]], axis=-1)

# Train data
Train = load_data(Train_data)
Target_train = np.array(Train_data["is_iceberg"])
print Target_train.shape
print("Totla Xtrain:", Train.shape)

# Test data
X_test = load_data(Test_data)
print("Total Xtest:", X_test.shape)
print "\n"
i = 0
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(Train):
    X_train, X_validation = Train[train_index], Train[test_index]
    y_train, y_validation = Target_train[train_index], Target_train[test_index]
    print "X_train : ", X_train.shape
    print "X_validate : ", X_validation.shape
    print "Non_Iceberg image examples for training : ", len(y_train) - np.count_nonzero(y_train)
    print "Iceberg image examples for training: ", np.count_nonzero(y_train)
    print "Non_Iceberg image examples for validation : ", len(y_validation) - np.count_nonzero(y_validation)
    print "Iceberg image examples for validation: ", np.count_nonzero(y_validation)
    model = Models.CNN_NCouches(img_rows, img_cols, color_type)
    checkpointer = ModelCheckpoint(filepath="./Weights/weightsOil" + str(i + 1) + ".hdf5", verbose=1,
                                   save_best_only=True)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_split=0.2, shuffle=True, callbacks=[checkpointer])
    model.load_weights(filepath="./Weights/weightsOil" + str(i + 1)+".hdf5")    # Make predictions
    prediction = model.predict(X_test, verbose=1)
    Results.append(prediction.flatten())
    print "\n"
    evaluation.append(model.evaluate(X_validation, y_validation))
    print(evaluation[i])
    print "\n"
    i += 1

print "\n"
print "Mean loss : ", (evaluation[0][0] + evaluation[1][0] + evaluation[2][0] + evaluation[3][0] + evaluation[4][0])/5
print "Mean Accuracy : ", (evaluation[0][1] + evaluation[1][1] + evaluation[2][1] + evaluation[3][1] + evaluation[4][1])/5
Results = np.asarray(Results)
prediction = (Results[0] + Results[1] +  Results[2] +  Results[3] + Results[4]) / 5

submit_Statoil = pd.DataFrame({'id': Test_data["id"], 'is_iceberg': prediction.flatten()})
submit_Statoil.to_csv("./submission/statoil_KFoldMergeWithChannel.csv", index=False)









