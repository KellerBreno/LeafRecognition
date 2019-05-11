from src.DataSet import DataSet
from src.RandomForest import RandomForest


def export_to_file(path, answers):
    output_file = open(path, "w+")
    output_file.write("file,species\n")
    for (filename, classname) in answers:
        output_file.write(filename + "," + classname + "\n")
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

    # Extract data from images
    # print("Extract features...")
    # Extraction.process_train_images("D:\\Projects\\LeafRecognition\\data\\NewData\\train",
    #                                 "D:\\Projects\\LeafRecognition\\data\\train_data.csv")
    # Extraction.process_test_images("D:\\Projects\\LeafRecognition\\data\\NewData\\test",
    #                                "D:\\Projects\\LeafRecognition\\data\\test_data.csv")

    # Retrive data
    print("Reading train data...")
    train_data = DataSet("train_data")
    train_data.read_data_from_file("D:\\Projects\\LeafRecognition\\data\\train_data.csv")

    # print("  Normalizing...")
    # train_data.normalize_z_score()

    # new_maxs = [1 for i in range(len(train_data.get_data()[0].get_features()))]
    # new_mins = [0 for i in range(len(train_data.get_data()[0].get_features()))]
    # train_data.normalize_min_max(new_maxs, new_mins)

    # Training
    print("Training...")
    random_forest = RandomForest(train_data)
    random_forest.train()

    # Classification
    print("Reading test data...")
    test_data = DataSet("test_data")
    test_data.read_data_from_file("D:\\Projects\\LeafRecognition\\data\\test_data.csv")

    # print("  Normalizing...")
    # test_data.normalize_z_score()

    # test_data.normalize_min_max(new_maxs, new_mins)

    print("Classifying...")
    answers = random_forest.classify_all(test_data)

    print("Writing results...")
    export_to_file("D:\\Projects\\LeafRecognition\\data\\submission.csv", answers)
