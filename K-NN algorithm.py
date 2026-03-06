from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
X = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

correct = 0
wrong = 0

for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        correct += 1
    else:
        wrong += 1

print("Correct Predictions:",correct)
print("Wrong Predictions:",wrong)