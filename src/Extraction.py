import os

from numpy import uint8
from scipy.stats import kurtosis
from skimage import io
from skimage.color import rgb2gray
from skimage.exposure import histogram
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
        # ========================================= V1 =================================================================
        # output_file.write(
        #     "filename;class;area;convex_area;eccentricity;filled_area;perimeter;solidity\n")

        # ========================================= V2 =================================================================
        # output_file.write(
        #     "filename;class;area;convex_area;eccentricity;filled_area;perimeter;solidity;extent;orientation\n")

        # ========================================= V3 =================================================================
        # output_file.write(
        #     "filename;class;area;convex_area;eccentricity;filled_area;perimeter;solidity;extent;orientation;kurtosis\n")

        # ========================================= V4 =================================================================
        # output_file.write(
        #     "filename;class;area;convex_area;eccentricity;filled_area;perimeter;solidity;extent;orientation;hu0;hu1;hu2;hu3;hu4;hu5;hu6\n")

        # ========================================= V5 =================================================================
        # output_file.write(
        #     "filename;class;area;convex_area;eccentricity;filled_area;perimeter;solidity;extent;orientation;kurtosis;hu0;hu1;hu2;hu3;hu4;hu5;hu6\n")

        # ========================================= V6 =================================================================
        output_file.write(
            "filename;class;area;convex_area;centroid_x;centroid_y;eccentricity;filled_area;perimeter;solidity;extent;orientation;kurtosis;hu0;hu1;hu2;hu3;hu4;hu5;hu6\n")

        for f in files:
            img = io.imread(f)
            thresh = threshold_otsu(rgb2gray(img))
            hist, _ = histogram(rgb2gray(img))
            binary = rgb2gray(img) > thresh
            regions = regionprops(uint8(binary))
            for region in regions:
                segments = f.split("\\")
                filename = segments[len(segments) - 1]
                classname = segments[len(segments) - 2]
                # ========================================= V1 =========================================================
                # output_file.write(str(
                #     filename + ";" + classname + ";" + str(region.area) + ";" + str(region.convex_area) + ";" + str(
                #         region.eccentricity) + ";" + str(region.filled_area) + ";" + str(region.perimeter) + ";" +
                #         str(region.solidity) + "\n"))

                # ========================================= V2 =========================================================
                # output_file.write(str(
                #     filename + ";" + classname + ";" + str(region.area) + ";" + str(region.convex_area) + ";" + str(
                #         region.eccentricity) + ";" + str(region.filled_area) + ";" + str(region.perimeter) + ";" + str(
                #         region.solidity) + ";" + str(region.extent) + ";" + str(region.orientation) + "\n"))

                # ========================================= V3 =========================================================
                # output_file.write(str(
                #     filename + ";" + classname + ";" + str(region.area) + ";" + str(region.convex_area) + ";" + str(
                #         region.eccentricity) + ";" + str(region.filled_area) + ";" + str(region.perimeter) + ";" + str(
                #         region.solidity) + ";" + str(region.extent) + ";" + str(region.orientation) + ";" + str(
                #         kurtosis(kurtosis(binary))) + "\n"))

                # ========================================= V4 =========================================================
                # output_file.write(str(
                #     filename + ";" + classname + ";" + str(region.area) + ";" + str(region.convex_area) + ";" + str(
                #         region.eccentricity) + ";" + str(region.filled_area) + ";" + str(region.perimeter) + ";" + str(
                #         region.solidity) + ";" + str(region.extent) + ";" + str(region.orientation) + ";" + str(
                #         region.moments_hu[0]) + ";" + str(region.moments_hu[1]) + ";" + str(
                #         region.moments_hu[2]) + ";" + str(region.moments_hu[3]) + ";" + str(
                #         region.moments_hu[4]) + ";" + str(region.moments_hu[5]) + ";" + str(
                #         region.moments_hu[6]) + "\n"))

                # ========================================= V5 =========================================================
                # output_file.write(str(
                #     filename + ";" + classname + ";" + str(region.area) + ";" + str(region.convex_area) + ";" + str(
                #         region.eccentricity) + ";" + str(region.filled_area) + ";" + str(region.perimeter) + ";" + str(
                #         region.solidity) + ";" + str(region.extent) + ";" + str(region.orientation) + ";" + str(
                #         kurtosis(kurtosis(binary))) + ";" + str(region.moments_hu[0]) + ";" + str(
                #         region.moments_hu[1]) + ";" + str(region.moments_hu[2]) + ";" + str(
                #         region.moments_hu[3]) + ";" + str(region.moments_hu[4]) + ";" + str(
                #         region.moments_hu[5]) + ";" + str(region.moments_hu[6]) + "\n"))

                # ========================================= V6 =========================================================
                output_file.write(
                    str(filename + ";" + classname + ";" + str(region.area) + ";" + str(region.convex_area) + ";" + str(
                        region.eccentricity) + ";" + str(region.filled_area) + ";" + str(
                        region.centroid[0]) + ";" + str(region.centroid[1]) + ";" + str(
                        region.perimeter) + ";" + str(region.solidity) + ";" + str(region.extent) + ";" + str(
                        region.orientation) + ";" + str(kurtosis(kurtosis(region.image))) + ";" + str(
                        region.moments_hu[0]) + ";" + str(region.moments_hu[1]) + ";" + str(
                        region.moments_hu[2]) + ";" + str(region.moments_hu[3]) + ";" + str(
                        region.moments_hu[4]) + ";" + str(region.moments_hu[5]) + ";" + str(
                        region.moments_hu[6]) + "\n"))

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
        # ========================================= V1 =================================================================
        # output_file.write(
        #     "filename;class;area;convex_area;eccentricity;filled_area;perimeter;solidity\n")

        # ========================================= V2 =================================================================
        # output_file.write(
        #     "filename;class;area;convex_area;eccentricity;filled_area;perimeter;solidity;extent;orientation\n")

        # ========================================= V3 =================================================================
        # output_file.write(
        #     "filename;class;area;convex_area;eccentricity;filled_area;perimeter;solidity;extent;orientation;kurtosis\n")

        # ========================================= V4 =================================================================
        # output_file.write(
        #     "filename;class;area;convex_area;eccentricity;filled_area;perimeter;solidity;extent;orientation;hu0;hu1;hu2;hu3;hu4;hu5;hu6\n")

        # ========================================= V5 =================================================================
        # output_file.write(
        #     "filename;class;area;convex_area;eccentricity;filled_area;perimeter;solidity;extent;orientation;kurtosis;hu0;hu1;hu2;hu3;hu4;hu5;hu6\n")

        # ========================================= V6 =================================================================
        output_file.write(
            "filename;class;area;convex_area;centroid_x;centroid_y;eccentricity;filled_area;perimeter;solidity;extent;orientation;kurtosis;hu0;hu1;hu2;hu3;hu4;hu5;hu6\n")

        for f in files:
            img = io.imread(f)
            thresh = threshold_otsu(rgb2gray(img))
            hist, _ = histogram(rgb2gray(img))
            binary = rgb2gray(img) > thresh
            regions = regionprops(uint8(binary))
            for region in regions:
                segments = f.split("\\")
                filename = segments[len(segments) - 1]
                classname = ""
                # ========================================= V1 =========================================================
                # output_file.write(str(
                #     filename + ";" + classname + ";" + str(region.area) + ";" + str(region.convex_area) + ";" + str(
                #         region.eccentricity) + ";" + str(region.filled_area) + ";" + str(region.perimeter) + ";" +
                #         str(region.solidity) + "\n"))

                # ========================================= V2 =========================================================
                # output_file.write(str(
                #     filename + ";" + classname + ";" + str(region.area) + ";" + str(region.convex_area) + ";" + str(
                #         region.eccentricity) + ";" + str(region.filled_area) + ";" + str(region.perimeter) + ";" + str(
                #         region.solidity) + ";" + str(region.extent) + ";" + str(region.orientation) + "\n"))

                # ========================================= V3 =========================================================
                # output_file.write(str(
                #     filename + ";" + classname + ";" + str(region.area) + ";" + str(region.convex_area) + ";" + str(
                #         region.eccentricity) + ";" + str(region.filled_area) + ";" + str(region.perimeter) + ";" + str(
                #         region.solidity) + ";" + str(region.extent) + ";" + str(region.orientation) + ";" + str(
                #         kurtosis(kurtosis(binary))) + "\n"))

                # ========================================= V4 =========================================================
                # output_file.write(str(
                #     filename + ";" + classname + ";" + str(region.area) + ";" + str(region.convex_area) + ";" + str(
                #         region.eccentricity) + ";" + str(region.filled_area) + ";" + str(region.perimeter) + ";" + str(
                #         region.solidity) + ";" + str(region.extent) + ";" + str(region.orientation) + ";" + str(
                #         region.moments_hu[0]) + ";" + str(region.moments_hu[1]) + ";" + str(
                #         region.moments_hu[2]) + ";" + str(region.moments_hu[3]) + ";" + str(
                #         region.moments_hu[4]) + ";" + str(region.moments_hu[5]) + ";" + str(
                #         region.moments_hu[6]) + "\n"))

                # ========================================= V5 =========================================================
                # output_file.write(str(
                #     filename + ";" + classname + ";" + str(region.area) + ";" + str(region.convex_area) + ";" + str(
                #         region.eccentricity) + ";" + str(region.filled_area) + ";" + str(region.perimeter) + ";" + str(
                #         region.solidity) + ";" + str(region.extent) + ";" + str(region.orientation) + ";" + str(
                #         kurtosis(kurtosis(binary))) + ";" + str(region.moments_hu[0]) + ";" + str(
                #         region.moments_hu[1]) + ";" + str(region.moments_hu[2]) + ";" + str(
                #         region.moments_hu[3]) + ";" + str(region.moments_hu[4]) + ";" + str(
                #         region.moments_hu[5]) + ";" + str(region.moments_hu[6]) + "\n"))

                # ========================================= V6 =========================================================
                output_file.write(
                    str(filename + ";" + classname + ";" + str(region.area) + ";" + str(region.convex_area) + ";" + str(
                        region.eccentricity) + ";" + str(region.filled_area) + ";" + str(
                        region.centroid[0]) + ";" + str(region.centroid[1]) + ";" + str(
                        region.perimeter) + ";" + str(region.solidity) + ";" + str(region.extent) + ";" + str(
                        region.orientation) + ";" + str(kurtosis(kurtosis(region.image))) + ";" + str(
                        region.moments_hu[0]) + ";" + str(region.moments_hu[1]) + ";" + str(
                        region.moments_hu[2]) + ";" + str(region.moments_hu[3]) + ";" + str(
                        region.moments_hu[4]) + ";" + str(region.moments_hu[5]) + ";" + str(
                        region.moments_hu[6]) + "\n"))

                output_file.flush()
        output_file.close()
