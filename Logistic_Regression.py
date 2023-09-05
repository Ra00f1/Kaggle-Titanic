from sklearn.model_selection import train_test_split
import Data
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

if __name__ == "__main__":
    train_data = Data.Read_Data("train")
    test_data = Data.Read_Data("test")

    # Delete unwanted columns
    train_data = train_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked", "Fare"], axis=1)

    # Label Encoding
    # male = 0, female = 1
    train_data.replace(["male", "female"], [0, 1], inplace=True)

    X_train = train_data.drop(["Survived"], axis=1)
    y_train = train_data["Survived"]

    # Logistic Regression
    Scaler = StandardScaler()

    # Fill missing Age values with mean
    # Test with both this and deleted row one
    X_train["Age"].fillna(X_train["Age"].mean(), inplace=True)

    # Note: Fit X_train only and transform X_test only
    X_train = Scaler.fit_transform(X_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Saving the model
    Data.Save_Model(log_reg, "Logistic_Regression")

    # Testing the model
    print("Testing Accuracy: ", log_reg.score(X_test, y_test))

    # Predicting the test data for Kaggle
    test_data_modified = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked", "Fare"], axis=1)

    # Label Encoding
    test_data_modified.replace(["male", "female"], [0, 1], inplace=True)
    # Fill missing Age values with mean
    test_data_modified["Age"].fillna(test_data_modified["Age"].mean(), inplace=True)

    X_test = Scaler.transform(test_data_modified)

    # Predicting the test data for Kaggle
    y_pred = log_reg.predict(X_test)

    # Saving the predictions and passenger id to csv file
    df = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})
    df.to_csv('Data/Logistic_Regression.csv', index=False)






