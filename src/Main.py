import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier

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
    #                                 "D:\\Projects\\LeafRecognition\\data\\train_data_kurtosis_new_version.csv")
    # Extraction.process_test_images("D:\\Projects\\LeafRecognition\\data\\NewData\\test",
    #                                "D:\\Projects\\LeafRecognition\\data\\test_data_kurtosis_new_version.csv")

    print("Reading train data...")
    train_data = DataSet("train_data")
    # train_data.read_data_from_file("D:\\Projects\\LeafRecognition\\data\\train_data.csv")
    # train_data.read_data_from_file("D:\\Projects\\LeafRecognition\\data\\train_data_new_version.csv")
    # train_data.read_data_from_file("D:\\Projects\\LeafRecognition\\data\\train_data_kurtosis.csv")
    train_data.read_data_from_file("D:\\Projects\\LeafRecognition\\data\\train_data_kurtosis_new_version.csv")
    # train_data = train_data.get_balanced_dataset()
    train_features = train_data.get_features()
    train_labels = train_data.get_labels()

    print("  Normalizing...")
    train_features = Normalizer().fit_transform(train_features)

    print("Training...")
    random_forest_entropy = RandomForestClassifier(n_estimators=100, criterion="entropy", n_jobs=-1)
    random_forest_gini = RandomForestClassifier(n_estimators=100, criterion="gini", n_jobs=-1)
    knn_uniform_euclidian = KNeighborsClassifier(n_neighbors=15, weights="uniform", p=2)
    knn_uniform_manhatan = KNeighborsClassifier(n_neighbors=15, weights="uniform", p=1)
    knn_distance_euclidian = KNeighborsClassifier(n_neighbors=15, weights="distance", p=2)
    knn_distance_manhatan = KNeighborsClassifier(n_neighbors=15, weights="distance", p=1)
    gaussian_bayes = GaussianNB()
    bernoulli_bayes = BernoulliNB()
    tree_gini = DecisionTreeClassifier(criterion="gini")
    tree_entropy = DecisionTreeClassifier(criterion="entropy")
    adaboost_bernoulli = AdaBoostClassifier(bernoulli_bayes, n_estimators=100)
    adaboost_gaussian = AdaBoostClassifier(gaussian_bayes, n_estimators=100)
    adaboost_tree_gini = AdaBoostClassifier(tree_gini, n_estimators=100)
    adaboost_tree_entropy = AdaBoostClassifier(tree_entropy, n_estimators=100)
    bagging_knn_uniform_euclidian = BaggingClassifier(knn_uniform_euclidian, n_estimators=100, n_jobs=-1)
    bagging_knn_uniform_manhatan = BaggingClassifier(knn_uniform_manhatan, n_estimators=100, n_jobs=-1)
    bagging_knn_distance_euclidian = BaggingClassifier(knn_distance_euclidian, n_estimators=100, n_jobs=-1)
    bagging_knn_distance_manhatan = BaggingClassifier(knn_distance_manhatan, n_estimators=100, n_jobs=-1)
    bagging_gaussian = BaggingClassifier(gaussian_bayes, n_estimators=100, n_jobs=-1)
    bagging_bernoulli = BaggingClassifier(bernoulli_bayes, n_estimators=100, n_jobs=-1)
    bagging_decision_tree_gini = BaggingClassifier(tree_gini, n_estimators=100, n_jobs=-1)
    bagging_decision_tree_entropy = BaggingClassifier(tree_entropy, n_estimators=100, n_jobs=-1)

    # clf = RandomForestClassifier(n_estimators=100, criterion="entropy", n_jobs=-1)
    # clf.fit(train_features, train_labels)

    # print(" Normalize...")
    # transformer = Normalizer().fit(train_features)
    # train_features = transformer.transform(train_features)

    # print(" Discretize...")
    # transformer = KBinsDiscretizer(n_bins=20, strategy="uniform").fit(train_features)
    # # transformer = KBinsDiscretizer(n_bins=20, strategy="quantile").fit(train_features)
    # # transformer = KBinsDiscretizer(n_bins=20, strategy="kmeans").fit(train_features)
    # train_features = transformer.transform(train_features)

    print("Cross Validation...")
    print(" Split...")
    skf = StratifiedKFold(n_splits=10, shuffle=True)

    print(" Simple...")
    try:
        scores = cross_val_score(tree_gini, train_features, y=train_labels, cv=skf, n_jobs=-1, error_score=np.nan)
        print("  Accuracy - Tree - Gini: %0.5f" % scores.mean())
    except:
        print("  ERROR: Tree - Gini")

    try:
        scores = cross_val_score(tree_entropy, train_features, y=train_labels, cv=skf, n_jobs=-1, error_score=np.nan)
        print("  Accuracy - Tree - Entropy: %0.5f" % scores.mean())
    except:
        print("  ERROR: Tree - Entropy")

    try:
        scores = cross_val_score(knn_uniform_euclidian, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Knn - Uniform - Euclidian: %0.5f" % scores.mean())
    except:
        print("  ERROR: KNN - Uniform - Euclidian")

    try:
        scores = cross_val_score(knn_uniform_manhatan, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Knn - Uniform - Manhatan: %0.5f" % scores.mean())
    except:
        print("  ERROR: KNN - Uniform - Manhatan")

    try:
        scores = cross_val_score(knn_distance_euclidian, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Knn - Distance - Euclidian: %0.5f" % scores.mean())
    except:
        print("  ERROR: KNN - Distance - Euclidian")

    try:
        scores = cross_val_score(knn_distance_manhatan, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Knn - Distance - Manhatan: %0.5f" % scores.mean())
    except:
        print("  ERROR: KNN - Distance - Manhatan")

    try:
        scores = cross_val_score(gaussian_bayes, train_features, y=train_labels, cv=skf, n_jobs=-1, error_score=np.nan)
        print("  Accuracy - Gaussian: %0.5f" % scores.mean())
    except:
        print("  ERROR: Gaussian")

    try:
        scores = cross_val_score(bernoulli_bayes, train_features, y=train_labels, cv=skf, n_jobs=-1, error_score=np.nan)
        print("  Accuracy - Bernoulli: %0.5f" % scores.mean())
    except:
        print("  ERROR: Bernoulli")

    print(" Ensembles...")

    try:
        scores = cross_val_score(random_forest_gini, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - RandomForest - Gini: %0.5f" % scores.mean())
    except:
        print("  ERROR: RandomForest - Gini")

    try:
        scores = cross_val_score(random_forest_gini, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - RandomForest - Entropy: %0.5f" % scores.mean())
    except:
        print("  ERROR: RandomForest - Entropy")

    try:
        scores = cross_val_score(adaboost_bernoulli, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Adaboost - Bernoulli: %0.5f" % scores.mean())
    except:
        print("  ERROR: Adaboost - Bernoulli")

    try:
        scores = cross_val_score(adaboost_gaussian, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Adaboost - Gaussian: %0.5f" % scores.mean())
    except:
        print("  ERROR: Adaboost - Gaussian")

    try:
        scores = cross_val_score(adaboost_tree_gini, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Adaboost - Tree - Gini: %0.5f" % scores.mean())
    except:
        print("  ERROR: Adaboost - Tree - Gini")

    try:
        scores = cross_val_score(adaboost_tree_entropy, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Adaboost - Tree - Entropy: %0.5f" % scores.mean())
    except:
        print("  ERROR: Adaboost - Tree - Entropy")

    try:
        scores = cross_val_score(bagging_knn_uniform_euclidian, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Bagging - KNN - Uniform - Euclidian: %0.5f" % scores.mean())
    except:
        print("  ERROR: Bagging - KNN - Uniform - Euclidian")

    try:
        scores = cross_val_score(bagging_knn_uniform_manhatan, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Bagging - KNN - Uniform - Manhatan: %0.5f" % scores.mean())
    except:
        print("  ERROR: Bagging - KNN - Uniform - Manhatan")

    try:
        scores = cross_val_score(bagging_knn_distance_euclidian, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Bagging - KNN - Distance - Euclidian: %0.5f" % scores.mean())
    except:
        print("  ERROR: Bagging - KNN - Distance - Euclidian")

    try:
        scores = cross_val_score(bagging_knn_distance_manhatan, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Bagging - KNN - Distance - Manhatan: %0.5f" % scores.mean())
    except:
        print("  ERROR: Bagging - KNN - Distance - Manhatan")

    try:
        scores = cross_val_score(bagging_bernoulli, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Bagging - Bernoulli: %0.5f" % scores.mean())
    except:
        print("  ERROR: Bagging - Bernoulli")

    try:
        scores = cross_val_score(bagging_gaussian, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Bagging - Gaussian: %0.5f" % scores.mean())
    except:
        print("  ERROR: Bagging - Gaussian")

    try:
        scores = cross_val_score(bagging_decision_tree_entropy, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Bagging - Tree - Entropy: %0.5f" % scores.mean())
    except:
        print("  ERROR: Bagging - Tree - Entropy")

    try:
        scores = cross_val_score(bagging_decision_tree_gini, train_features, y=train_labels, cv=skf, n_jobs=-1,
                                 error_score=np.nan)
        print("  Accuracy - Bagging - Tree - Gini: %0.5f" % scores.mean())
    except:
        print("  ERROR: Bagging - Tree - Gini")

    # print("Reading test data...")
    # test_data = DataSet("test_data")
    # test_data.read_data_from_file("D:\\Projects\\LeafRecognition\\data\\test_data.csv")
    # test_features = test_data.get_features()
    #
    # # print("  Normalizing...")
    # # test_features = Normalizer().fit_transform(test_features)
    #
    # print("Classifying...")
    # answers = clf.predict(test_features)
    #
    # print("Writing results...")
    # export_to_file("D:\\Projects\\LeafRecognition\\data\\submission.csv", test_data.get_filenames(), answers)
