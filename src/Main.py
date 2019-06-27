import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from src.DataSet import DataSet
from src.Extraction import Extraction


def export_to_file(path, ids, answers):
    output_file = open(path, "w+")
    output_file.write("file,species\n")
    for i in range(0, len(ids)):
        output_file.write(ids[i] + "," + answers[i] + "\n")
    output_file.close()


if __name__ == "__main__":
    # print("Extract features...")
    # Extraction.process_train_images("D:\\Projects\\LeafRecognition\\data\\NewData\\train",
    #                                 "D:\\Projects\\LeafRecognition\\data\\train_data_centroid_new.csv")
    # Extraction.process_test_images("D:\\Projects\\LeafRecognition\\data\\NewData\\test",
    #                                "D:\\Projects\\LeafRecognition\\data\\test_data_centroid_new.csv")

    print("Reading train data...")
    train_data = DataSet("train_data")
    train_data.read_data_from_file("D:\\Projects\\LeafRecognition\\data\\train_data_centroid_new.csv")
    train_features = train_data.get_features()
    train_labels = train_data.get_labels()

    # print("Cross Validation...")
    # print(" Split...")
    # skf = StratifiedKFold(n_splits=10, shuffle=True)

    print("Training...")
    clf = RandomForestClassifier(n_estimators=4096, criterion="entropy", n_jobs=-1)
    clf.fit(train_features, train_labels)

    print("Reading test data...")
    test_data = DataSet("test_data")
    test_data.read_data_from_file("D:\\Projects\\LeafRecognition\\data\\test_data_centroid_new.csv")
    test_features = test_data.get_features()

    print("Classifying...")
    answers = clf.predict(test_features)

    print("Writing results...")
    export_to_file("D:\\Projects\\LeafRecognition\\data\\submission.csv", test_data.get_filenames(), answers)
