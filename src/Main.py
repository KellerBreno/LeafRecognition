from sklearn.ensemble import RandomForestClassifier

from src.DataSet import DataSet


def export_to_file(path, ids, answers):
    output_file = open(path, "w+")
    output_file.write("file,species\n")
    for i in range(0, len(ids)):
        output_file.write(ids[i] + "," + answers[i] + "\n")
    output_file.close()


if __name__ == "__main__":
    # Map namefiles from kaggle in dataset
    # Map.map("D:\\Projects\\LeafRecognition\\data\\Original\\test",
    #             "D:\\Projects\\LeafRecognition\\data\\Nonsegmented")

    # Rename segmentated images using the previous map
    # Map.rewrite_train_data("D:\\Projects\\LeafRecognition\\data\\gabarito\\ScentlessMayweed.csv",
    #                            "D:\\Projects\\LeafRecognition\\data\\Segmented\\Scentless Mayweed",
    #                            "D:\\Projects\\LeafRecognition\\data\\NewData\\train\\Scentless Mayweed")
    # Map.rewrite_test_data("D:\\Projects\\LeafRecognition\\data\\gabarito\\test.csv",
    #                           "D:\\Projects\\LeafRecognition\\data\\Segmented",
    #                           "D:\\Projects\\LeafRecognition\\data\\NewData\\test")

    # print("Extract features...")
    # Extraction.process_train_images("D:\\Projects\\LeafRecognition\\data\\NewData\\train",
    #                                 "D:\\Projects\\LeafRecognition\\data\\train_data_resized.csv")
    # Extraction.process_test_images("D:\\Projects\\LeafRecognition\\data\\NewData\\test",
    #                                "D:\\Projects\\LeafRecognition\\data\\test_data_resized.csv")

    print("Reading train data...")
    train_data = DataSet("train_data")
    train_data.read_data_from_file("D:\\Projects\\LeafRecognition\\data\\train_data.csv")
    train_features = train_data.get_features()
    train_labels = train_data.get_labels()

    # print("  Normalizing...")
    # train_features = Normalizer().fit_transform(train_features)

    print("Training...")
    clf = RandomForestClassifier(n_estimators=100, criterion="gini", n_jobs=-1)
    clf.fit(train_features, train_labels)

    print("Reading test data...")
    test_data = DataSet("test_data")
    test_data.read_data_from_file("D:\\Projects\\LeafRecognition\\data\\test_data.csv")
    test_features = test_data.get_features()

    # print("  Normalizing...")
    # test_features = Normalizer().fit_transform(test_features)

    print("Classifying...")
    answers = clf.predict(test_features)

    print("Writing results...")
    export_to_file("D:\\Projects\\LeafRecognition\\data\\submission.csv", test_data.get_ids(), answers)
