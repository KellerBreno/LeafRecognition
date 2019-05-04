from src import Mapping

if __name__ == "__main__":
    # Mapping.map("D:\\Projects\\LeafRecognition\\data\\Original\\test",
    #             "D:\\Projects\\LeafRecognition\\data\\Nonsegmented")
    Mapping.rewrite_train_data("D:\\Projects\\LeafRecognition\\data\\gabarito\\ScentlessMayweed.csv",
                               "D:\\Projects\\LeafRecognition\\data\\Segmented\\Scentless Mayweed",
                               "D:\\Projects\\LeafRecognition\\data\\NewData\\train\\Scentless Mayweed")
