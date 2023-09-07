import Data
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Saving the predictions and passenger id to csv file
def save_kaggle_style_csv(y_pred, test_data, filename):
    print("Saving Kaggle Style CSV")
    df = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})
    filename = "Data/" + filename + ".csv"
    df.to_csv(filename, index=False)

def kaggle_test(log_reg):
    print("Kaggle Test Started")
    test_data = Data.Read_Data("test")
    test_data_modified = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Fare"], axis=1)
    #test_data_modified = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked", "Fare"], axis=1)

    # Label Encoding
    test_data_modified.replace(["male", "female"], [0, 1], inplace=True)
    test_data_modified.replace(["S", "C", "Q"], [0, 1, 2], inplace=True)

    # Fill missing Age values with mean
    test_data_modified["Age"].fillna(test_data_modified["Age"].mean(), inplace=True)
    test_data_modified["Embarked"].fillna(test_data_modified["Embarked"].mean(), inplace=True)

    # Feature Scaling (transform only)
    X_test = Scaler.transform(test_data_modified)

    # Predicting the test data for Kaggle
    y_pred = log_reg.predict(X_test)

    print("Kaggle Test Completed")
    return y_pred, test_data

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

if __name__ == "__main__":
    train_data = Data.Read_Data("train")

    # Delete unwanted columns
    train_data = train_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Fare"], axis=1)
    #train_data = train_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked", "Fare"], axis=1)

    # Label Encoding
    # male = 0, female = 1
    train_data.replace(["male", "female"], [0, 1], inplace=True)
    train_data.replace(["S", "C", "Q"], [0, 1, 2], inplace=True)

    # Check for missing values
    updated_data = train_data.dropna(axis=0)

    # Splitting the data
    X_train = train_data.drop(["Survived"], axis=1)
    y_train = train_data["Survived"]
    # Feature Scaling
    Scaler = StandardScaler()

    # Fill missing Age values with mean
    X_train["Age"].fillna(X_train["Age"].mean(), inplace=True)
    X_train["Embarked"].fillna(X_train["Embarked"].mean(), inplace=True)
    # ********************************* Note: Fit X_train only and transform X_test only *********************************

    log_reg = logistik_Regression(X_train, y_train)

    # Save the model
    Data.Save_Model(log_reg, "Logistic_Regression")

    # Predicting the test data for Kaggle
    kaggle_csv = kaggle_test(log_reg)
    save_kaggle_style_csv(kaggle_csv[0], kaggle_csv[1], "Logistic_Regression(without Pclass and without age mean and with Embarked)")