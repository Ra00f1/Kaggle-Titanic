import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def Read_Data(filename):
    filename = "Data/" + filename + ".csv"
    df = pd.read_csv(filename)
    return df

if __name__ == "__main__":
    train_data = Read_Data("train")
    print(train_data.head())
    print(train_data["Survived"].value_counts())
    #print(train_data["SibSp"].value_counts())
    #print(train_data["Parch"].value_counts())

    f = plt.figure(1)
    sns.barplot(x='Survived', y='Age', data=train_data, hue='Sex')
    f.show()

    g = plt.figure(2)
    sns.barplot(x='Survived', y='Pclass', data=train_data, hue='Sex')
    g.show()

    h = plt.figure(3)
    sns.barplot(x='Survived', y='SibSp', data=train_data, hue='Sex')
    h.show()

    y = plt.figure(3)
    sns.barplot(x='Survived', y='Parch', data=train_data, hue='Sex')
    y.show()
