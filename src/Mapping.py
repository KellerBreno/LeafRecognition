import os

import numpy as np
from skimage import io


class Map:
    @staticmethod
    def map(path_base, path_comparative):
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

    @staticmethod
    def rewrite_train_data(path_gabarito, path_segments, path_destiny):
        gab_file = open(path_gabarito, "r")
        gab_file.readline()
        lines = gab_file.readlines()
        for line in lines:
            line.strip("\n")
            line_split = line.split(";")
            path_name = line_split[1].split("\\")
            path_new_name = line_split[0].split("\\")
            img_name = path_name[len(path_name) - 1]
            img_new_name = path_new_name[len(path_new_name) - 1]
            try:
                img = io.imread(os.path.join(path_segments.strip(), img_name.strip()))
                io.imsave(os.path.join(path_destiny, img_new_name), img)
            except FileNotFoundError:
                print(os.path.join(path_segments.strip(), img_name.strip()))

    @staticmethod
    def rewrite_test_data(path_gabarito, path_segments, path_destiny):
        gab_file = open(path_gabarito, "r")
        gab_file.readline()
        lines = gab_file.readlines()
        for line in lines:
            line.strip("\n")
            line_split = line.split(";")
            path_name = line_split[1].split("\\")
            path_new_name = line_split[0].split("\\")
            img_name = path_name[len(path_name) - 1]
            img_name_directory = path_name[len(path_name) - 2]
            img_new_name = path_new_name[len(path_new_name) - 1]
            try:
                img = io.imread(os.path.join(os.path.join(path_segments.strip(), img_name_directory), img_name.strip()))
                io.imsave(os.path.join(path_destiny, img_new_name), img)
            except FileNotFoundError:
                print(os.path.join(path_segments.strip(), img_name.strip()))
