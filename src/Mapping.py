import os

import numpy as np
from skimage import io


def main(path_base, path_comparative):
    files_base = []
    for r, d, f in os.walk(path_base):
        for file in f:
            if ".png" in file:
                files_base.append(os.path.join(r, file))

    files_comparative = []
    for r, d, f in os.walk(path_comparative):
        for file in f:
            if ".png" in file:
                files_comparative.append(os.path.join(r, file))

    output_file = open("..\\data\\gabarito\\test.csv", "w+")
    output_file.write("path1;path2\n")
    # print("path1,path2")
    for f_base in files_base:
        img_base = io.imread(f_base)
        for f_comp in files_comparative:
            img_comparative = io.imread(f_comp)
            if np.shape(img_base) == np.shape(img_comparative):
                img_difference = img_base - img_comparative
                if img_difference.sum() == 0:
                    output_file.write(f_base + ";" + f_comp + "\n")
                    output_file.flush()
                    # print(f_base + "," + f_comp)
                    break
    output_file.close()
