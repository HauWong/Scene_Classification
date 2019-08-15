# !/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy


def convert_to_uint8(img_arr):
    bands = numpy.split(img_arr, img_arr.shape[2], axis=2)
    img_min = img_arr.min()
    img_max = img_arr.max()
    img = band_process(bands[0], img_min, img_max)
    for i in range(1, len(bands)):
        bp = band_process(bands[i], img_min, img_max)
        img = numpy.concatenate((img, bp), axis=2)
    return img


def band_process(band_arr, min, max):
    calc_ls = (255 * (band_arr - min) / (max - min)).tolist()
    map_ls = list(map(float_to_int8, calc_ls))
    res_ls = []
    for i in range(0, len(map_ls)):  # 行
        cur_col = list(map_ls[i])
        col_ls = []
        for j in range(0, len(cur_col)):  # 列
            data = list(cur_col[j])
            col_ls.append(data)
        res_ls.append(col_ls)
    res_arr = numpy.array(res_ls)
    res_arr = res_arr.astype('uint8')
    return res_arr


def float_to_int8(l):
    if type(l) == list:
        return map(float_to_int8, l)
    return int(l)


def normalize(arr):
    amin = arr.min()
    amax = arr.max()
    res = (arr - amin)/(amax - amin)
    return res


def statistics(arr, num):
    res = numpy.zeros(num)
    for i in arr:
        res[i] += 1
    return res


def dms_to_ten(loc_dms):
    dms_deg = loc_dms.split(' ')
    loc_ten = float(dms_deg[0]) + float(dms_deg[1])/60 + float(dms_deg[2])/3600

    return loc_ten


def ten_to_dms(loc_ten):
    deg = int(loc_ten)
    mint = int((loc_ten - deg)*60)
    sec = ((loc_ten - deg)*60 - mint)*60
    dms_str = '%03d %02d %.2f'%(deg, mint, sec)

    return dms_str


def count_percent(arr, value):
    cond = arr == value
    value_arr = numpy.extract(cond, arr)
    percent = float(value_arr.size)/float(arr.size)
    return percent


def connected_area_mark(img_arr, type_dict):
    res = []
    for reg_type in type_dict.keys():
        min_v = type_dict[reg_type][0]
        max_v = type_dict[reg_type][1]
        for value in range(min_v, max_v+1):
            cur_area = {'name': value, 'class': reg_type}
            pos = numpy.argwhere(img_arr == value)
            min_row = int(pos[:, 0].min())
            max_row = int(pos[:, 0].max())
            min_col = int(pos[:, 1].min())
            max_col = int(pos[:, 1].max())
            rect = (min_row, min_col, max_row+1, max_col+1)
            cur_area['rect'] = rect
            res.append(cur_area)
    return res


def mask(img_arr, mask_arr, value):
    for i in range(mask_arr.shape[0]):
        for j in range(mask_arr.shape[1]):
            if mask_arr[i, j] != value:
                img_arr[i, j] = 0
    return img_arr
