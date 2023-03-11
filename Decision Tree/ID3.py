import pandas as pd

# 读取数据
Iris = pd.read_csv(r'D:\\jiajian\\code\\GitHub&Study\\The-second-volume-of-junior-year\\Decision Tree\\iris.data', header=None)
print(Iris)

# 数据预处理
Iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
Iris['class'] = Iris['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
print(Iris)