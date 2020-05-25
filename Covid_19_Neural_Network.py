import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import keras
from keras import regularizers
from sklearn.preprocessing import StandardScaler
import pickle
import time
import matplotlib.pyplot as plt


def readData(csvFile):
    dataFrame = pd.read_csv(csvFile, sep=',', encoding="ISO-8859-1", error_bad_lines=False)
    return dataFrame


def encodeData(dataFrame, column):
    dataList = dataFrame[column].to_list()
    labelEncoder = LabelEncoder()
    labelEncoder.fit(dataList)
    encodedData = labelEncoder.transform(dataList)
    dataFrame[column+"Num"] = encodedData
    return encodedData

#
# Example: createStats(df, ['SRC_CITY', 'DEST_CITY'], 'ACTUAL_TRANSIT_TIME (DAYS)')
# keyColumns Example: ['SRC_CITY', 'DEST_CITY']
# column Example: 'ACTUAL_TRANSIT_TIME (DAYS)'
# df_by_keys[1][0] should be the key: 'SRC_CITY' and 'DEST_CITY' pair : ('BEDFORD', 'ASHBURN')
def createStats(df, keyColumns, column):
    df_by_keys = df.groupby(keyColumns, as_index=False)
    col_dict = {'count' : column+'_COUNT',
            'min' : column+'_MIN',
            'max' : column + '_MAX',
            'mean' : column + '_MEAN',
            'std' : column + '_STD'}

    stats = df_by_keys[column].agg(list(col_dict.keys()))
    # Fill NaN with zeros.  This could happen with std and variance calculations
    stats = stats.fillna(0)
    stats = stats.rename(columns = col_dict)

#    merged_df = pd.merge(df, stats, on=keyColumns)
    df = df.set_index(keyColumns)
    merged_df = df.join(stats, on=keyColumns)
    col_dict_values = list(col_dict.values())
    merged_df = merged_df.reset_index()
    df = df.reset_index()
    df[col_dict_values] = merged_df[col_dict_values]

    return df

def toDateTime(df, column):
    df[column] = pd.to_datetime(df[column], format='%m/%d/%y')
    # df[column + '_yearNum'] = df[column].dt.year
    df[column + '_dayofyearNum'] = (df[column].dt.year - 2020) * 365 + df[column].dt.dayofyear
    df = df.sort_values(by=["date", "region", "state"])
    return df

def encodeRegionAndState(dataFrame):
    encodeData(dataFrame, 'region')
    encodeData(dataFrame, 'state')
    dataFrame = dataFrame.drop(['region', 'state'], axis=1)
    return dataFrame


def build_model(n_hidden=1, n_neurons=128, learning_rate=3e-4, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    # optimizer = keras.optimizers.SGD(lr=learning_rate)
    # optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
    optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False, clipnorm=1.)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', 'mape', 'mse'])
    return model

def lstm_train(X_train, y_train):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
                            input_shape=[None, 1]),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10))
    ])

    model.compile(loss="mse", optimizer="adam", metrics=['mae', 'mape', 'mse'])
    history = model.fit(X_train, y_train, epochs=20,
                        validation_split=0.1, input_shape=[X_train.shape[1]])
    return model, history, scaler


def train(X_train, y_train):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    model = build_model(n_hidden=5, n_neurons=16, learning_rate=3e-3, input_shape=[X_train.shape[1]])
    history = model.fit(X_train, y_train, epochs=500, batch_size=16, validation_split=0.1, workers=4, verbose=2)
    return model, history, scaler

def evaluate(X_test, y_test, scaler):
    # Transform the test data using the scaler that is used for testing
    X_test = scaler.transform(X_test)
    history_eval = model.evaluate(X_test, y_test, batch_size=1, verbose=1)
    print("Test Results:")
    for i in range(len(model.metrics_names)):
        print("%s: %.2f" % (model.metrics_names[i], history_eval[i]))
    return history_eval

def plotHistory(pltFile, history):
    # Plot training & validation accuracy values
    plt.plot(history.history['mape'])
    plt.plot(history.history['mse'])
    plt.title('Mean Absolute Percent Error and mean square error')
    plt.ylabel('Mean Absolute Percent Error')
    plt.xlabel('Epoch')
    plt.legend(['MAPE', 'MSE'], loc='upper left')
    # plt.show()
    plt.savefig(pltFile+"_MAPE.png")

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['LOSS', 'VAL_LOSS'], loc='upper left')
    plt.savefig(pltFile+"_LOSS.png")

def predict(model, scaler, X_test):
    X_test = scaler.transform(X_test)
    return model.predict(X_test)


def saveModel(model, modelFile):
    pickle.dump(model, open(modelFile, 'wb'))


def loadModel(modelFile):
    return pickle.load(open(modelFile, "rb"))

#
# This is the main code that calls other functions
#########

# Read column header data
folder = "./"

# Read the training data and convert the text data to numbers and date time text to date time fields
csvFile = folder + "brazil_covid19_revised.csv"
dataFrame = readData(csvFile)

# Change to dateTime format and Sort by date, region, state
dataFrame = toDateTime(dataFrame, 'date')
df = encodeRegionAndState(dataFrame)

# Select the feature columns and the label column
X = df.loc[:, df.columns != 'cases']
y = np.ravel(df.loc[:, df.columns == 'cases'])

testDate = '2020-05-05'
splitSize = X[(X['date'] <= testDate)].shape[0]

# Drop the date, new cases and deaths fields
X = X.drop(['date', 'newCases', 'deaths'], axis=1)


X_train_orig, X_test_orig = dataFrame[:splitSize], dataFrame[splitSize:]
y_train_orig, y_test_orig = y[:splitSize], y[splitSize:]

X_train, X_test = X[:splitSize], X[splitSize:]
y_train, y_test = y[:splitSize], y[splitSize:]

# Train and test the model
start = time.time()
model, history, scaler = train(X_train, y_train)
elapsed = time.time() - start
print ("Runtime = %.2f Seconds" % elapsed)

# Evaluate the model using test data
history_eval = evaluate(X_test, y_test, scaler)

# Perform prediction using test data
y_train_predicted = predict(model, scaler, X_train)
y_test_predicted = predict(model, scaler, X_test)
y_train_predicted = pd.DataFrame(y_train_predicted)
y_test_predicted = pd.DataFrame(y_test_predicted)
y_predicted = pd.concat([y_train_predicted, y_test_predicted])

y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)

y_original = pd.concat([y_train_df, y_test_df])

X_train_orig = X_train_orig.reset_index()
X_train_orig = X_train_orig.drop('index', axis=1)
X_train_orig['ML_TYPE'] = X_train_orig['ML_TYPE'] = 'TRAIN'

X_test_orig = X_test_orig.reset_index()
X_test_orig = X_test_orig.drop('index', axis=1)
X_test_orig['ML_TYPE'] = X_test_orig['ML_TYPE'] = 'TEST'

X_final_df = pd.concat([X_train_orig,X_test_orig])
X_final_df = X_final_df.reset_index()
X_final_df = X_final_df.drop('index', axis=1)

y_predicted = y_predicted.reset_index();
y_predicted = y_predicted.drop('index', axis=1)

y_original = y_original.reset_index()
y_original = y_original.drop('index', axis=1)

X_final_df['Y_PREDICTED'] = y_predicted
X_final_df['Y_ORIGINAL'] = y_original

#add the metric columns
abs_error = np.abs((y_predicted - y_original))
abs_percent_error = abs_error * 100 / y_original
squared_error = np.square(abs_error)
root_squared_error = np.sqrt(squared_error)
X_final_df['ABS_ERROR'] = abs_error
X_final_df['ABS_PERCENT_ERROR'] = abs_percent_error
X_final_df['SQUARED_ERROR'] = squared_error
X_final_df['ROOT_SQUARED_ERROR'] = root_squared_error

# Drop the numeric columns
numeric_columns = list(filter(lambda x: x.find("Num") >= 0, X_final_df.columns))
X_final_df = X_final_df.drop(numeric_columns, axis=1)

# write the predicted file
predictedFile = folder + "Covid19_Predicted.csv"
X_final_df.to_csv(predictedFile)

pd.DataFrame(history.history).to_csv(folder+"Covid19_History.csv")
# Plot the results

pltFile = folder + "Covid19"
plotHistory(pltFile, history)
