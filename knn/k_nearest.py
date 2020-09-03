
import sklearn
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn import linear_model, preprocessing



import numpy as np

from os import listdir
from os.path import isfile, join

def process_data(path_to_folder='training_data/'):

    onlyfiles = [f for f in listdir(path_to_folder) if isfile(join(path_to_folder, f))]

    array_of_files = []
    for files in onlyfiles:
        f = np.load(path_to_folder + files)
        for pose in f:
            array_of_files.append(pose)

    np.save('files_combined.npy', array_of_files)


def create_knn(path_to_file='files_combined.npy'):

    f = np.load(path_to_file)

    y, X = create_training_data(f)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    return knn

def create_training_data(file_of_poses):
    y, X = [], []

    for p in file_of_poses:
        if p[0][0] == 0:
            y.append(False)
        else: 
            y.append(True)

        flatten_kp = p[1]

        for kp in p[2:]:
            flatten_kp = np.append(flatten_kp, kp)

        X.append(flatten_kp)

    return y, X

if __name__ == '__main__':
    process_data()