import pandas as pd
import joblib

# Saving data to csv files(single data file)
def Save_Csv(file, filename):
    filename += '.csv'
    filename = "Data/" + filename
    file.to_csv(filename, sep=',', index=False, encoding='utf-8')
    print("Saved File: ", filename)

# Loading data from csv files(single data file)
def Read_Data(filename):
    filename = "Data/" + filename + '.csv'
    print("Loading File: ", filename)
    return pd.read_csv(filename, sep=',')

def Save_Model(model, filename):
    filename = "Trained/" + filename + '.pkl'
    joblib.dump(model, filename)
    print("Saved Model: ", filename)

# Loading a saved KNN model
def Load_Model(filename):
    filename = "Trained/" + filename + '.pkl'
    print("Loading Model: ", filename)
    return joblib.load(filename)

# Loading data from csv files(4 data files)
def Loading_Data():
    print("Loading Data!")
    X_train = pd.read_csv("Data/X_train", sep=',')
    X_test = pd.read_csv("Data/X_test", sep=',')
    y_train = pd.read_csv("Data/y_train", sep=',')
    y_test = pd.read_csv("Data/y_test", sep=',')
    print("Loading Completed!")

    return X_train, X_test, y_train, y_test

# Saving data to csv files(4 data files)
def Saving_Data(X_train, X_test, y_train, y_test):
    print("Saving Data!")
    X_train.to_csv("X_train", sep=',', index=False, encoding='utf-8')
    X_test.to_csv("X_test", sep=',', index=False, encoding='utf-8')
    y_train.to_csv("y_train", sep=',', index=False, encoding='utf-8')
    y_test.to_csv("y_test", sep=',', index=False, encoding='utf-8')
    print("Saving Completed!")
