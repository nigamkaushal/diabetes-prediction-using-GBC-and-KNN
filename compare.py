import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
data = pd.read_csv(r'C:\Users\kaushal\Desktop\SET\diabetes.csv')
max_skinthickness = data.SkinThickness.max()
data = data[data.SkinThickness!=max_skinthickness]
acc=[]

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_train, y_train, test_size = 0.3, random_state=100)


knn = KNeighborsClassifier(n_neighbors=8)
clf_ = knn.fit(X_train, y_train)
y_pred = clf_.predict(X_test)
a=acc.append(accuracy_score(y_test,y_pred ))

#confusion matrix
results = confusion_matrix(y_test, y_pred)
print('KNN')
print('Confusion Matrix :')
print(results)

print('Report : ')
print(classification_report(y_test, y_pred))


params = {'max_depth':9, 'subsample':0.5, 'learning_rate':0.01, 'min_samples_leaf':1, 'random_state':0}
gbc = GradientBoostingClassifier(n_estimators=290, **params)
clf_ = gbc.fit(X_train, y_train)
y_pred = clf_.predict(X_test)
a=acc.append(accuracy_score(y_test,y_pred ))

#Bar graph
plt.bar(['KNN', 'Gradient Boosting Classifier'], acc, color=['green', 'brown'], label='Accuracy')
plt.ylabel('Accuracy Score')
plt.xlabel('Algortihms')
plt.show()

#confusion matrix
results = confusion_matrix(y_test, y_pred)
print('Gradient Boosting')
print('Confusion Matrix :')
print(results)

print('Report : ')
print(classification_report(y_test, y_pred))