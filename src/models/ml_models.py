from sklearn.naive_bayes import GaussianNB

def gaussiannb(X_train, y_train, X_test):
     classifier = GaussianNB()
     classifier.fit(X_train, y_train)
     y_pred = classifier.predict(X_test)
     return y_pred
