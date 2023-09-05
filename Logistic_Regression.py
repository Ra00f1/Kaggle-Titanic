import Data
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def kaggle_test(log_reg):
    print("Kaggle Test Started")
    test_data = Data.Read_Data("test")
    test_data_modified = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked", "Fare"], axis=1)

    # Label Encoding
    test_data_modified.replace(["male", "female"], [0, 1], inplace=True)

    # Fill missing Age values with mean
    test_data_modified["Age"].fillna(test_data_modified["Age"].mean(), inplace=True)

    # Feature Scaling (transform only)
    X_test = Scaler.transform(test_data_modified)

    # Predicting the test data for Kaggle
    y_pred = log_reg.predict(X_test)

    # Saving the predictions and passenger id to csv file
    df = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})
    df.to_csv('Data/Logistic_Regression.csv', index=False)
    print("Kaggle Test Completed")

def logistik_Regression(X_train, y_train):
    print("Logistic Regression Started")
    X_train = Scaler.fit_transform(X_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Saving the model
    Data.Save_Model(log_reg, "Logistic_Regression")

    # Testing the model
    print("Testing Accuracy: ", log_reg.score(X_test, y_test))
    return log_reg

if __name__ == "__main__":
    train_data = Data.Read_Data("train")

    # Delete unwanted columns
    train_data = train_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked", "Fare"], axis=1)

    # Label Encoding
    # male = 0, female = 1
    train_data.replace(["male", "female"], [0, 1], inplace=True)

    # Splitting the data
    X_train = train_data.drop(["Survived"], axis=1)
    y_train = train_data["Survived"]

    # Feature Scaling
    Scaler = StandardScaler()

    # Fill missing Age values with mean
    X_train["Age"].fillna(X_train["Age"].mean(), inplace=True)

    # Note: Fit X_train only and transform X_test only

    log_reg = logistik_Regression(X_train, y_train)
    # Predicting the test data for Kaggle
    kaggle_test(log_reg)







