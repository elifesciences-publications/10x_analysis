'''
Script to load the stitched images and save as stacks in correct order.
'''

import os
from os.path import isfile, join
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tifffile import imsave
from tkinter import filedialog
from tkinter import *
import cv2 as cv



# dirs containing the stitched images from the two different channels
#w470_dir = '/home/jacob/Documents/Chubb_lab/notebook/why_beome_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/analysis/100419_JR3_testing/w470_stitched'
#w572_dir = '/home/jacob/Documents/Chubb_lab/notebook/why_beome_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/analysis/100419_JR3_testing/w572_stitched'

# or let user select dir from dialog
root = Tk()
data_dir = filedialog.askdirectory(parent=root, initialdir="/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/", title='Please select directory containing directories containing stitched images')
w470_dir = data_dir + '/w470_stitched'
w572_dir = data_dir + '/w572_stitched'


## add seconds since first image to file name ###

# for each wavelength folder contain all stitched images
#for dir in [w470_dir, w572_dir, wExt_dir]:
#for dir in [wExt_dir]:


for dir in [w470_dir, w572_dir]:

    os.chdir(dir)  # move into that dir
    flist = [f for f in os.listdir(dir) if isfile(join(dir, f))]
    print(flist)

    # to contain datatime objects
    date_list = []

    for fname in flist:
        date_time_fname_string = re.findall('(?<=\s)(.*)(?=_stitched.tif)', fname)[0]
        print(date_time_fname_string)

        split = date_time_fname_string.split(' ')
        month = split[0]
        if len(month) == 1:
            month = '0' + month
        day = split[1]
        time_str = split[2]
        year = split[3]

        # now convert time string to hrs, min, s
        hour = time_str.split('h')[0][0:2]
        mins = time_str.split('h')[0][2:]
        if len(mins) == 1:
            mins = '0' + mins
        s = re.findall('(?<=h)(.*)(?=s)',time_str)[0]
        if len(s) == 1:
            s = '0' + s
        milli = time_str.split('s')[-1]

        new_datetime_str = year + ' ' + month + ' ' + day + ' ' + hour + ' ' + mins + ' ' + s
        print(new_datetime_str)

        # create datetime object from new_datetime_str
        datetime = datetime.strptime(new_datetime_str, '%Y %b %d %H %M %S')

        # add to date_list
        date_list = date_list + [datetime]


    # for each element in date_list, calulucate difference in seconds from fisrt
    s_diff_list = [None]*len(date_list)
    s_diff_list[0] = 0
    for i in range(1, len(date_list)):
        s_diff_list[i] = (date_list[i] - date_list[0]).total_seconds()


    # now subtradt min value from s_diff_list to get seconds since first image for each fname
    s_diff_list = np.asarray(s_diff_list)
    s_diff_list = s_diff_list - np.amin(s_diff_list)
    s_diff_list = np.ndarray.tolist(s_diff_list)
    s_diff_list = [round(x) for x in s_diff_list]
    print(s_diff_list)




    # rename files with time since first at start
    for i in range(0, len(flist)):
        os.rename(dir + '/' + flist[i], dir + '/' + 'deltaT' + str(s_diff_list[i]) + 'secs_' + flist[i])





