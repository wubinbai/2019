import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()

def print_faces(images, target, top_n):
    fig = plt.figure(figsize=(12,12))
    fig.subplots_adjust(left=0, right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
    for i in range(top_n):
        p=fig.add_subplot(20,20,i+1,xticks=[],yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        p.text(0,14,str(target[i]))
        p.text(0,60,str(i))

from sklearn.svm import SVC
svc_1 = SVC(kernel = 'linear')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=0)

from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem

def evaluate_cross_validation(clf, X, y, K):
    cv= KFold(len(y), K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print("Mean score: {0:.3f}(+/-{1:.3f})".format(np.mean(scores), sem(scores)))

evaluate_cross_validation(svc_1, X_train, y_train, 5)
