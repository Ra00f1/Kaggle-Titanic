import pandas as pd
import matplotlib.pyplot as plt

def Read_Data(filename):
    filename = "Data/" + filename + ".csv"
    df = pd.read_csv(filename)
    return df

if __name__ == "__main__":
    train_data = Read_Data("train")

    #plt_sex = plt.bar(train_data['Survived'], train_data['Sex'])
    #plt_sex.title("Sex vs Survived")
    #plt_sex.xlable('Sex')
    #plt_sex.ylable('Survived')

    #plt_age = plt.scatter(train_data['Age'], train_data['Survived'])
    #plt_age.title("Age vs Survived")
    #plt_age.xlable('Age')
    #plt_age.ylable('Survived')
#
    plt_pclass = plt.bar(train_data['Pclass'], train_data['Survived'])
    #plt_pclass.title("Pclass vs Survived")
    #plt_pclass.xlable('Pclass')
    #plt_pclass.ylable('Survived')
#
    #plt_sibps = plt.scatter(train_data['SibSp'], train_data['Survived'])
    #plt_sibps.title("SibSp vs Survived")
    #plt_sibps.xlable('SibSp')
    #plt_sibps.ylable('Survived')
#
    #plt_parch = plt.scatter(train_data['Parch'], train_data['Survived'])
    #plt_parch.title("Parch vs Survived")
    #plt_parch.xlable('Parch')
    #plt_parch.ylable('Survived')

    plt.show()
