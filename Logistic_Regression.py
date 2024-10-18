import Data
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Feature Scaling
Scaler = StandardScaler()

def Data_Summary(data):
    print("Data Summary for file: ", data)
    print("=====================================")
    print("First 5 rows")
    print(data.describe())
    print("=====================================")
    print("Data types")
    print(data.info())
    print("=====================================")
    print("Data count")
    print(data.count())
    print("=====================================")
    print("Missing values")
    print(data.isnull().sum())
    print("=====================================")
    print("Data shape")
    print(data.shape)
    print("=====================================")
    print("Unique values in each column")
    print(data.nunique())
    print("=====================================")

    # describe the last column
    print("Last column description")
    # print(data.iloc[:, -1].describe())
    # # Visualize the last column
    # temp_data = data.drop(data.index[0])
    # plt.figure(figsize=(7, 6))
    # plt.bar(temp_data.iloc[:, -1].unique(), temp_data.iloc[:, -1].value_counts())
    # plt.show(block=True)

    print("---------------------------------------------------------------------------------")


# Saving the predictions and passenger id to csv file
def save_kaggle_style_csv(y_pred, test_data, filename):
    print("Saving Kaggle Style CSV")
    df = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})
    filename = "Data/" + filename + ".csv"
    df.to_csv(filename, index=False)

def kaggle_test(log_reg, X_test):
    print("Kaggle Test Started")

    # Predicting the test data for Kaggle
    y_pred = log_reg.predict(X_test)

    print("Kaggle Test Completed")
    return y_pred

def logistik_Regression(X_train, y_train):
    print("Logistic Regression Started")
    X_train = Scaler.fit_transform(X_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Testing the model
    print("Testing Accuracy: ", log_reg.score(X_test, y_test))
    return log_reg


def Data_Processing(train_data, train=True):
    # Delete unwanted columns
    train_data = train_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Fare"], axis=1)
    # train_data = train_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked", "Fare"], axis=1)

    # Label Encoding
    for column in train_data.columns:
        if train_data[column].dtype == 'object':
            train_data[column] = train_data[column].astype('category')
            train_data[column] = train_data[column].cat.codes

    # Fill missing Age values with mean
    train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
    # Fill missing Embarked values the most frequent value
    train_data["Embarked"].fillna(train_data["Embarked"].mode()[0], inplace=True)

    # ****************************** Note: Fit X_train only and transform X_test only ******************************

    if train:
        # Feature Scaling (fit only)
        X_train = train_data.drop("Survived", axis=1)
        X_train = Scaler.fit_transform(X_train)
        y_train = train_data["Survived"]

        return X_train, y_train
    else:
        X_train = Scaler.transform(train_data)

        return X_train


if __name__ == "__main__":
    train_data = Data.Read_Data("train")

    # Data_Summary(train_data)

    X_train, y_train = Data_Processing(train_data)

    log_reg = logistik_Regression(X_train, y_train)

    # Save the model
    Data.Save_Model(log_reg, "Logistic_Regression")

    # Predicting the test data for Kaggle
    test_data = Data.Read_Data("test")
    X_test = Data_Processing(test_data, train=False)
    kaggle_csv = kaggle_test(log_reg, X_test)
    kaggle_csv = [kaggle_csv, test_data]
    save_kaggle_style_csv(kaggle_csv[0], kaggle_csv[1], "Logistic_Regression(without Pclass and without age mean and with Embarked)")