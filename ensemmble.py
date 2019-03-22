import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
data = pd.read_csv(r'C:\Users\kaushal\Desktop\SET\diabetes.csv')
max_skinthickness = data.SkinThickness.max()
data = data[data.SkinThickness!=max_skinthickness]
    #create a helper function
def replace_zero(df, field, target):
    mean_by_target = df.loc[df[field] != 0, [field, target]].groupby(target).mean()
    data.loc[(df[field] == 0)&(df[target] == 0), field] = mean_by_target.iloc[0][0]
    data.loc[(df[field] == 0)&(df[target] == 1), field] = mean_by_target.iloc[1][0]

    # run the function
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    replace_zero(data, col, 'Outcome')

# split data
X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)

# load algorithms
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

# helper functions
def train_clf(clf, X_train, y_train):
    return clf.fit(X_train, y_train)


def pred_clf(clf, features, target):
    y_pred = clf.predict(features)
    return f1_score(target.values, y_pred, pos_label=1)


def train_predict(clf, X_train, y_train, X_test, y_test):
    train_clf(clf, X_train, y_train)

    print("F1 score for training set is: {:.4f}".format(pred_clf(clf, X_train, y_train)))
    print("F1 score for testing set is: {:.4f}\n".format(pred_clf(clf, X_test, y_test)))

# load algorithms
gbc = GradientBoostingClassifier(random_state=0)
print("{}:".format(gbc))
train_predict(gbc, X_train, y_train, X_test, y_test)


from sklearn.metrics import accuracy_score

params = {'max_depth':9, 'subsample':0.5, 'learning_rate':0.01, 'min_samples_leaf':1, 'random_state':0}
gbc = GradientBoostingClassifier(n_estimators=290, **params)
clf_ = gbc.fit(X_train, y_train)
y_pred = clf_.predict(X_test)
print('Accuracy is {}'.format(accuracy_score(y_test,y_pred )))
train_predict(gbc, X_train, y_train, X_test, y_test)