import os

from numpy import uint8
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import regionprops


class Extraction:
    @staticmethod
    def process_train_images(path, path_result):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if ".png" in file:
                    files.append(os.path.join(r, file))

        output_file = open(path_result, "w+")
        output_file.write("filename;class;area;convex_area;eccentricity;filled_area;perimeter;solidity\n")
        for f in files:
            print(f)
            img = io.imread(f)
            thresh = threshold_otsu(rgb2gray(img))
            binary = rgb2gray(img) > thresh
            regions = regionprops(uint8(binary))
            for region in regions:
                segments = f.split("\\")
                filename = segments[len(segments) - 1]
                classname = segments[len(segments) - 2]
                output_file.write(str(
                    filename + ";" + classname + ";" + str(region.area) + ";" + str(region.convex_area) + ";" + str(
                        region.eccentricity) + ";" + str(region.filled_area) + ";" + str(region.perimeter) + ";" + str(
                        region.solidity) + "\n"))
                output_file.flush()
        output_file.close()

    @staticmethod
    def process_test_images(path, path_result):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if ".png" in file:
                    files.append(os.path.join(r, file))

        output_file = open(path_result, "w+")
        output_file.write("filename;class;area;convex_area;eccentricity;filled_area;perimeter;solidity\n")
        for f in files:
            print(f)
            img = io.imread(f)
            thresh = threshold_otsu(rgb2gray(img))
            binary = rgb2gray(img) > thresh
            regions = regionprops(uint8(binary))
            for region in regions:
                segments = f.split("\\")
                filename = segments[len(segments) - 1]
                classname = ""
                output_file.write(str(
                    filename + ";" + classname + ";" + str(region.area) + ";" + str(region.convex_area) + ";" + str(
                        region.eccentricity) + ";" + str(region.filled_area) + ";" + str(region.perimeter) + ";" + str(
                        region.solidity) + "\n"))
                output_file.flush()
        output_file.close()
