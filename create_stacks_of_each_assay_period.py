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
data_dir = filedialog.askdirectory(parent=root, initialdir="/media/jacob/data/Chubb_lab/notebook/feeding_front_decision/imaging" , title='Please select directory containing directories containing stitched images')
w470_dir = data_dir + '/w470_stitched'
w572_dir = data_dir + '/w572_stitched'
wExt_dir = data_dir + '/wExt_stitched'

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







#######################################################################################################################
#### now get file names, and reorder based on deltaT, open images in order, and add to stack ####


#for dir in [w470_dir, w572_dir, wExt_dir]:
#for dir in [wExt_dir]:
for dir in [w470_dir, w572_dir]:

    os.chdir(dir)  # move into that dir
    flist = [f for f in os.listdir(dir) if isfile(join(dir, f))]
    print(flist)

    # to contain datatime objects
    deltaT_list = [None]*len(flist)

    for i in range(0, len(flist)):
        deltaT_list[i] = int(re.findall('(?<=deltaT)(.*)(?=secs_)', flist[i])[0])
    print(deltaT_list)

    # sort flist based on deltaT list
    deltaT_list, flist = zip(*sorted(zip(deltaT_list, flist)))  # OMG, it works!
    print(flist)


    # open files in flist (now in correct order) and add to stack
    ## open each image in file as array and put in list ##
    image_list = [None]*len(flist)
    i = 0
    for fname in flist:
        #print(dir+'/'+fname)
        #image = plt.imread(dir+'/'+fname)  # for some reason image ends up duplicated on 4 planes...
        image = cv.imread(dir + '/' + fname, -1)
        #image = image[:, :, 1]
        print(image.shape)
        image_list[i] = image
        i += 1

    #image_list.reverse()  # reverse order of images

    # one problem is that because the x and y stage is not perfect, the different stitched images can
    # be of slightly different size, so need to trim to same size to be able to make a stack

    im_width_list = []
    im_height_list = []
    for image in image_list:
        im_width_list = im_width_list + [image.shape[0]]
        im_height_list = im_height_list + [image.shape[1]]

    min_width = min(im_width_list)
    min_height = min(im_height_list)

    # now trim
    for i in range(0, len(image_list)):
        image_list[i] = image_list[i][0:min_width, 0:min_height]

    for image in image_list:
        print(image.shape)



    # create the stacks based on times of GFP images

    os.chdir(dir)
    stack = np.dstack(image_list)
    stack = np.swapaxes(stack,0,2)
    print(stack.shape)
    imsave(dir+'/'+ 'stack.tif', stack) # assumes stack shape is (z,x,y)


