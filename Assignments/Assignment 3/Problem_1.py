#Kmeans (NDimentional by Daniel McDonough)
import matplotlib.pyplot as plt
import random
import math
from sklearn.datasets import load_digits
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
#%matplotlib inline

#generates random hex values
def get_hex(i):
    r = lambda: random.randint(256, 257)
    while r() + i > 255:
        r = lambda: random.randint(20, 200)+i
        color = '#{:02x}{:02x}{:02x}'.format(r(),r(),r())
    return  color


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

# generate random points
def gen_rand_points():
    data = []
    for i in range(1000):
        item_x = random.uniform(0, 1000)
        item_y = random.uniform(0, 1000)
        data.append((item_x,item_y))
    return data


# Euclidian Distance between two N-dimensional points
def eucldist(p0, p1):
    dist = 0.0
    for i in range(0, len(p0)):
        dist += (p0[i] - p1[i]) ** 2
    return math.sqrt(dist)


def init_centroids(datapoints,k):
    # Randomly Choose Centers for the Clusters
    cluster_centers = []
    for i in range(0, k):
        new_cluster = []
        # for i in range(0,d):
        #    new_cluster += [random.randint(0,10)]
        cluster_centers += [random.choice(datapoints)]
    return cluster_centers


# K-Means Algorithm
def kmeans(k, datapoints,output_type,labels=None):
    # d - Dimensionality of Datapoints
    d = len(datapoints[0])

    # Limit our iterations
    Max_Iterations = 1000
    iteration = 0 #iteration counter

    cluster = [0] * len(datapoints) #have an array for clusters
    prev_cluster = [-1] * len(datapoints)
    # have an array of old clusters to check if
    # it's the same as the new one so we can stop


    #Randomly place centroids
    cluster_centers = init_centroids(datapoints,k)

    #this is used to prevent calculating the distance an empty cluster vector
    force_recalculation = False

    #do K-means
    while (cluster != prev_cluster) or (iteration > Max_Iterations) or (force_recalculation):

        prev_cluster = list(cluster)
        force_recalculation = False
        iteration += 1

        # Update Point's Cluster Alligiance
        for p in range(0, len(datapoints)):
            min_dist = float("inf")

            # Check min_distance against all centers
            for c in range(0, len(cluster_centers)):

                dist = eucldist(datapoints[p], cluster_centers[c])

                if (dist < min_dist):
                    min_dist = dist
                    cluster[p] = c  # Reassign Point to new Cluster

        # Update Cluster's Position
        for k in range(0, len(cluster_centers)):
            new_center = [0] * d
            members = 0
            for p in range(0, len(datapoints)):
                if (cluster[p] == k):  # If this point belongs to the cluster
                    for j in range(0, d):
                        new_center[j] += datapoints[p][j]
                    members += 1

            for j in range(0, d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members)

                    # This means that our initial random assignment was poorly chosen
                # Change it to a new datapoint to actually force k clusters
                else:
                    new_center = random.choice(datapoints)
                    force_recalculation = True
                    print("Forced Recalculation...")

            cluster_centers[k] = new_center

    # print("======== Results ========")
    # print("Clusters", cluster_centers)
    # print("Iterations", iteration)
    # print("Labels", cluster)

    #if not getting the digit set print the graph
    if output_type != "2":
        fig, ax = plt.subplots()
        for i in range(k+1):
            points = np.array([datapoints[j] for j in range(len(datapoints)) if cluster[j] == i]) #plot the points to a corresponding cluster
            ax.scatter(points[:, 0], points[:, 1], s=5, c=get_hex(i), label=("Cluster " + str(i + 1)))
        cluster_centers= np.array(cluster_centers) #just to make getting the columns easier
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=20, c='#050505', label="Centroid") #plot all the centroids

        plt.legend(loc=3)
        plt.title('Cluster Data K-Means')
        plt.ylabel('Width')
        plt.xlabel('Length')
        plt.show()

    elif output_type == "2":
        get_rep_matrix(labels, cluster, 10)
        con_matrix = confusion_matrix(labels, cluster)
        accuracy = metrics.fowlkes_mallows_score(labels, cluster)
        #todo claculate confusion matrix
        print("Confusion Matrix:\n", con_matrix)
        print("Accuracy: ",accuracy)

def main():
    inputdata = input("Enter:\n 0 for random data\n 1 for cluster_data.txt\n 2 for digits data set\n")

    if inputdata == "0":
        data = gen_rand_points()
        k = int(input("Enter number of wanted clusters\n"))
        if k == 0 or k >= len(data):
            print("Invalid K \n")
            exit(1)
        kmeans(k, data, inputdata)

    elif inputdata == "1":
        fileobject = np.loadtxt("./cluster_data.txt")  # get the data
        k = int(input("Enter number of wanted clusters\n"))
        fileobject = np.delete(fileobject, 0, 1) #delete the index count
        if k == 0 or k >= len(fileobject):
            print("Invalid K \n")
            exit(1)
        kmeans(k, fileobject,inputdata)

    elif inputdata == "2":

        k = 10  # K - Number of Clusters

        digits = load_digits()

        fileobject = digits.data

        n_samples, n_features = fileobject.shape

        labels = digits.target

        #print(fileobject.shape)
        print("n_digits: %d, \t n_samples %d, \t n_features %d" % (k, n_samples, n_features))

        # pick a ratio for splitting the digits list
        # into a training and a validation set.
        training_size = int(len(fileobject) * 0.25)
        validation = fileobject[:training_size]
        training = fileobject[training_size:]


        kmeans(k, fileobject,inputdata,labels)
    else:
        print("input not valid")
        exit(1)

# main
if __name__ == "__main__":
    main()