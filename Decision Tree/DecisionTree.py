import pandas as pd

# 读取数据
Iris = pd.read_csv(r'D:\\jiajian\\code\\GitHub&Study\\The-second-volume-of-junior-year\\Decision Tree\\iris.data', header=None)
print(Iris)

# 数据预处理
Iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
Iris['class'] = Iris['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
print(Iris)
 
# 数据集划分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Iris.iloc[:, :-1], Iris.iloc[:, -1], test_size=0.3, random_state=0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# 训练模型
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

# 模型评估
from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# 可视化
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

# 保存模型
import joblib
joblib.dump(clf, 'iris.pkl')

# 加载模型
clf = joblib.load('iris.pkl')
y_pred = clf.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
