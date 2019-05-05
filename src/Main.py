from src import Mapping, Extraction

if __name__ == "__main__":
    # Map namefiles from kaggle in dataset
    # Mapping.map("D:\\Projects\\LeafRecognition\\data\\Original\\test",
    #             "D:\\Projects\\LeafRecognition\\data\\Nonsegmented")

    # Rename segmentated images using the previous map
    # Mapping.rewrite_train_data("D:\\Projects\\LeafRecognition\\data\\gabarito\\ScentlessMayweed.csv",
    #                            "D:\\Projects\\LeafRecognition\\data\\Segmented\\Scentless Mayweed",
    #                            "D:\\Projects\\LeafRecognition\\data\\NewData\\train\\Scentless Mayweed")
    # Mapping.rewrite_test_data("D:\\Projects\\LeafRecognition\\data\\gabarito\\test.csv",
    #                           "D:\\Projects\\LeafRecognition\\data\\Segmented",
    #                           "D:\\Projects\\LeafRecognition\\data\\NewData\\test")

    # Extract data from images
    Extraction.process_all_images("D:\\Projects\\LeafRecognition\\data\\NewData\\train",
                                  "D:\\Projects\\LeafRecognition\\data\\train_data.csv")
