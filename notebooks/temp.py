import pandas as pd

df = pd.read_csv(r'C:\Users\33017\Desktop\wine_predictions\notebooks\tokenized_wine_data.csv')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#count_vect = CountVectorizer()
tfidf = TfidfTransformer()
cv = CountVectorizer().fit_transform(df.tokenized_sent)
bagofwords = tfidf.fit_transform(cv)

data = df.values
y = data[:,3]
#from sklearn.preprocessing import LabelBinarizer
#encoder = LabelBinarizer()
#y = encoder.fit_transform(y)

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

svd = TruncatedSVD(200)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(bagofwords)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#X_train, y_train, X_test, y_test = X_train[:5000], y_train[:5000], X_test[:1000], y_test[:1000]


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
results = {}
for clf_type in ["Random-Forest", "Gradient-Boosting"]:

    acc_results = []  # accuracy
    if clf_type is "Random-Forest":
        n_estimators = [200, 500]
    elif clf_type is "Gradient-Boosting":                                      
        n_estimators = [50, 100]
    
    for n_estimator in n_estimators:
        print("train....... (num trees = {:d} in {:s})".format(n_estimator, clf_type))
        if clf_type is "Random-Forest":
            clf = RandomForestClassifier(n_estimators=n_estimator)
        elif clf_type is "Gradient-Boosting":
            clf = GradientBoostingClassifier(n_estimators=n_estimator, max_depth=2)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_results.append(100 * accuracy_score(y_test, y_pred))
        print("done.")
    results[clf_type] = (n_estimators, acc_results)
    

import matplotlib.pyplot as plt
for clf_type in ["Random-Forest", "Gradient-Boosting"]:
    plt.plot(results[clf_type][0], results[clf_type][1])
    plt.xlabel("number of trees")
    plt.ylabel("model accuracy")

