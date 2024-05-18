from sklearn import datasets
import pandas as pd 

def prepare():
    # Load dataset.
    data = datasets.load_iris()
    # print(data)

    # https://stackoverflow.com/questions/55346510/how-do-i-write-scikit-learn-dataset-to-csv-file.
    df = pd.DataFrame(data=data['data'], columns = data['feature_names'])
    df.head(10)
    df.tail(10)
    df.to_csv('iris.txt', sep = ',', index = False)

if __name__ == "__main__":
    prepare()
