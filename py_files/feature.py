# !/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops

import py_files.data_tool as dt


class Feature(object):

    def __init__(self, img_array):
        self.img_array = img_array
        self.img_array_uint8 = dt.convert_to_uint8(self.img_array)

    def get_mean(self):
        band_count = self.img_array.shape[2]
        mean_arr = np.zeros(band_count)
        for i in range(0, band_count):
            mean_arr[i] = np.mean(self.img_array[:, :, i])
        return mean_arr

    def get_standard_deviation(self):
        band_count = self.img_array.shape[2]
        std_arr = np.zeros(band_count)
        for i in range(0, band_count):
            std_arr[i] = np.std(self.img_array[:, :, i])
        return std_arr

    def get_spectral(self):
        mean_arr = self.get_mean()
        std_arr = self.get_standard_deviation()
        spec_arr = np.zeros(2*mean_arr.shape[0])
        for i in range(0, mean_arr.shape[0]):
            spec_arr[2*i] = mean_arr[i]
            spec_arr[2*i-1] = std_arr[i]
        return spec_arr

    def get_sift(self):
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(self.img_array_uint8, None)
        return kp, des

    def get_surf(self, threshold):
        surf = cv2.xfeatures2d.SURF_create(threshold)
        kp, des = surf.detectAndCompute(self.img_array_uint8, None)
        return kp, des

    def get_glcm(self, distance=(2, 8, 16), angle=(0, np.pi/4, np.pi/2, np.pi*3/4),
                 prop_name=('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM')):
        glcm_features = np.zeros(1)
        band_count = self.img_array_uint8.shape[2]
        for band_index in range(0, band_count):
            cur_glcm = greycomatrix(self.img_array_uint8[:, :, band_index],
                                    distance, angle, 256, symmetric=True, normed=True)
            for prop in prop_name:
                tmp_value = greycoprops(cur_glcm, prop)
                tmp_value = tmp_value.reshape(-1)
                glcm_features = np.concatenate((glcm_features, tmp_value), axis=0)
        glcm_features = np.delete(glcm_features, 0, axis=0)
        return glcm_features

    def get_gist(self):
        pass
