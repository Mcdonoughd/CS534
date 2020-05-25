#Clustering Tests by Daniel Mcdonough
import matplotlib.pyplot as plt
from copy import deepcopy
import random
from sklearn.datasets import load_digits
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# rep matrix is more moreless the same as the confusion matrix
def get_rep_matrix(labels,pred_labels,k=10):
    rep_matrix = np.zeros((k,k))
    num_points = len(pred_labels)
    for i in range(num_points):
        y = labels[i]
        x = pred_labels[i]
        rep_matrix[y][x] += 1
    for i in range(len(rep_matrix)):
        print("Best representative of cluster ",i," is ",np.argmax(rep_matrix[i]))
    return rep_matrix

def main():
    digits = load_digits()

    print("K-means Representation is in Problem 1. \n")

    X = digits.data

    n_samples, n_features = X.shape

    labels = digits.target
    # pick a ratio for splitting the digits list
    # into a training and a validation set.
    training_size = int(10) # ratio for affinity is n = clusters
    training = X[:training_size]
    validation = X[training_size:]
    #affinty propagation requires training data
    clustering = AffinityPropagation(preference=10).fit(training)
    print("\nAffinity Propagation: \n")


    validation_labels = labels[training_size:]
    cluster = clustering.predict(validation)
    get_rep_matrix(validation_labels, cluster, 10)
    con_matrix = confusion_matrix(validation_labels, cluster)
    accuracy = metrics.fowlkes_mallows_score(validation_labels, cluster)


    print("Confusion Matrix:\n", con_matrix)
    print("Accuracy: ", accuracy)







    clustering = AgglomerativeClustering(n_clusters=10).fit(X)
    print("\nAgglomerative Clustering: \n")

    get_rep_matrix(labels, clustering.labels_, 10)
    con_matrix = confusion_matrix(labels, clustering.labels_)
    accuracy = metrics.fowlkes_mallows_score(labels, clustering.labels_)

    print("Confusion Matrix:\n", con_matrix)
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    main()