import warnings
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pydotplus
from io import StringIO, BytesIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


sns.set()
warnings.filterwarnings('ignore')
np.random.seed = 17


# 一定是两个特征，才能用
def plot_decision_area(X, y, model):
    if len(X.shape) != 2:
        return None
    if X.shape[1] != 2:
        return None
    xmin, xmax = X[:, 0].min()-1, X[:, 0].max()+1
    ymin, ymax = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    pred = model.predict(np.c_[xx.flat, yy.flat]).reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, pred, cmap="summer")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="black")
    plt.show()


def plot_tree(tree, feature_names, **kwargs):
    dot_data = StringIO()
    export_graphviz(tree, feature_names=feature_names, out_file=dot_data, filled=True, **kwargs)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    plt.figure(figsize=(10, 6))
    img= plt.imread(BytesIO(graph.create_png()), "png")
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def get_clf_result(X,y, estimator, clf_name, test_size=0.0, ):
    X_train = X
    y_train = y
    X_holdout = X
    y_holdout = y

    if test_size != 0.0:
        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=test_size, random_state=17)
    estimator.fit(X_train, y_train)

    plot_decision_area(X_train, y_train, estimator)

    if "tree" in clf_name:
        feature_names = ["x"+str(i) for i in range(X_train.shape[1])]
        plot_tree(estimator, feature_names)

    pred = estimator.predict(X_holdout)
    print(clf_name + "predicted:", accuracy_score(y_holdout, pred))


# train_data = np.r_[np.random.normal(size=(100, 2)), np.random.normal(size=(100, 2), loc=2)]
# train_labels = np.r_[np.zeros(100), np.ones(100)]
# X = train_data
# y = train_labels

# data = pd.DataFrame({'Age': [17, 64, 18, 20, 38, 49, 55, 25, 29, 31, 33],
#                      'Loan Default': [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1]})
# X = data['Age'].values.reshape(-1, 1)
# y= data['Loan Default'].values

# data2 = pd.DataFrame({'Age':  [17, 64, 18, 20, 38, 49, 55, 25, 29, 31, 33],
#                       'Salary': [25, 80, 22, 36, 37, 59, 74, 70, 33, 102, 88],
#                       'Loan Default': [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1]})
# X = data2[['Age', 'Salary']].values
# y = data2['Loan Default'].values

df = pd.read_csv('telecom_churn.csv')
df['International plan'] = pd.factorize(df['International plan'])[0]
df['Voice mail plan'] = pd.factorize(df['Voice mail plan'])[0]
df['Churn'] = df['Churn'].astype('int')
states = df['State']
y = df['Churn']
df.drop(['State', 'Churn'], axis=1, inplace=True)
X = df.values

tree = DecisionTreeClassifier(max_depth=5, random_state=17)
knn = KNeighborsClassifier(n_neighbors=10)
get_clf_result(X, y, tree, clf_name="tree", test_size=0.3)
get_clf_result(X, y, knn, clf_name="knn", test_size=0.3)