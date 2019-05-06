from src.Data import AREA_COLUMN, ECCENTRICITY_COLUMN, FILLED_AREA_COLUMN, CONVEX_AREA_COLUMN, SOLIDITY_COLUMN, \
    PERIMETER_COLUMN
from src.Processing import get_all_data, min_max_normalize

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
    # Extraction.process_all_images("D:\\Projects\\LeafRecognition\\data\\NewData\\train",
    #                               "D:\\Projects\\LeafRecognition\\data\\train_data.csv")

    # Retrive data
    data_list = get_all_data("D:\\Projects\\LeafRecognition\\data\\train_data.csv")

    print(data_list[0].filename)
    print(data_list[0].classname)
    print(data_list[0].area)
    print(data_list[0].convex_area)
    print(data_list[0].eccentricity)
    print(data_list[0].filled_area)
    print(data_list[0].perimeter)
    print(data_list[0].solidity)
    print()

    # Normalize data
    min_max_normalize(data_list, column=AREA_COLUMN)
    min_max_normalize(data_list, column=CONVEX_AREA_COLUMN)
    min_max_normalize(data_list, column=ECCENTRICITY_COLUMN)
    min_max_normalize(data_list, column=FILLED_AREA_COLUMN)
    min_max_normalize(data_list, column=PERIMETER_COLUMN)
    min_max_normalize(data_list, column=SOLIDITY_COLUMN)

    print(data_list[0].filename)
    print(data_list[0].classname)
    print(data_list[0].area)
    print(data_list[0].convex_area)
    print(data_list[0].eccentricity)
    print(data_list[0].filled_area)
    print(data_list[0].perimeter)
    print(data_list[0].solidity)
    print()
