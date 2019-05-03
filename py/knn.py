from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

data = np.array([
    [1, 4],
    [1, 3],
    [2, 5],
    [4, 2]]
)
kelas = [0, 0, 1, 1]
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(data, kelas)
kelasBaru = knn.predict([[3, 2]])
print(kelasBaru)
validasi = knn.predict(data)
print(confusion_matrix(kelas, validasi))