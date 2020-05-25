import matplotlib.pyplot as plt
import numpy as np


def graphThings(classes, ratio=True):
    count = []
    if len(classes[0]) == 0:
        pass
    for c in classes:
        counts = c['class'].value_counts()
        total = np.sum(counts)
        if not ratio:
            total = 1
        if 'cancer' in counts.index and 'healthy' in counts.index:
            count.append([counts['healthy'] / total, counts['cancer'] / total])
        elif 'healthy' in counts.index:
            arr = [counts['healthy'] / total, 0]
            count.append(arr)
        elif 'cancer' in counts.index:
            arr = [0, counts['cancer'] / total]
            count.append(arr)

    count = sorted(count, key=lambda x: x[1])
    index = np.arange(len(classes))
    bar_width = 0.35
    _, ax = plt.subplots()
    ax.bar(index, list(map(lambda x: x[0], count)), bar_width, color='r', label='Healthy')
    ax.bar(index + bar_width, list(map(lambda x: x[1], count)), bar_width, color='g', label='Damaged')
    ax.set_title("Healthy count vs Damaged count by cluster")
    plt.xticks(())
    plt.xlabel("Cluster Number")
    plt.ylabel("Count")
    ax.legend()
    plt.show()




