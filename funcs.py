import trackpy as tp
import pims
import image_analysis_module2 as ia
import matplotlib.pyplot as plt
import re
import numpy as np
import os
from os.path import isfile, join
import pandas as pd
import PIL
from PIL import Image
import cv2 as cv


def locate_feature(image_path, particle_size, minmass):

    #image = plt.imread(image_path)
    image = cv.imread(image_path,
                      -1)  # using cv2 https://stackoverflow.com/questions/18446804/python-read-and-write-tiff-16-bit-three-channel-colour-images

    image = image[8:-1, :]  # crop out noise

    # rotate by 180 (fudge)
    #image = np.rot90(image, 2)
    image = np.fliplr(image)

    frame = pims.Frame(image)

    # plt.figure(0)
    # plt.imshow(image)

    f = tp.locate(frame, particle_size, minmass=minmass)

    # plt.figure(1)
    # tp.annotate(f, frame)
    # plt.show(block=False)
    #
    # fig, ax = plt.subplots()
    # ax.hist(f['mass'], bins=20)
    # plt.show(block=False)

    return f


def locate_feature_calibrate(image_path, particle_size, minmass):

    #image = plt.imread(image_path, format='TIFF')
    image = cv.imread(image_path, -1)  # using cv2 https://stackoverflow.com/questions/18446804/python-read-and-write-tiff-16-bit-three-channel-colour-images

    #image = image[8:-1, :]  # crop out noise
    frame = pims.Frame(image)

    plt.figure(0)
    plt.imshow(image)

    f = tp.locate(frame, particle_size, minmass=minmass)

    plt.figure(1)
    tp.annotate(f, frame, plot_style={'markersize':10, 'markeredgewidth':1})
    plt.show(block=True)

    fig, ax = plt.subplots()
    ax.hist(f['mass'], bins=20)
    plt.show(block=True)

    return f


def is_point_in_box(area, point):

    xmin = area[0]
    xmax = area[1]
    ymin = area[2]
    ymax = area[3]

    x = point[0]
    y = point[1]

    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
        return True

    else:
        return False


def is_point_in_boxes(box_list, point):
    x = point[0]
    y = point[1]

    for area in box_list:

        xmin = area[0]
        xmax = area[1]
        ymin = area[2]
        ymax = area[3]

        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
            return True

    return False


def locate_across_image_collection(dir, pixel_size, particle_size, minmass):

    flist = [f for f in os.listdir(dir) if isfile(join(dir, f))]
    #print(flist)

    # remove file names that contain 'stitched'
    flist = [f for f in flist if 'stitched' not in f]
    #print(flist)

    # make list of x and y coordinates for each image
    x_coord_list = [''] * len(flist)
    y_coord_list = [''] * len(flist)

    for i in range(0, len(flist)):
        f = flist[i]
        x_string = re.findall('(?<=X)(.*)(?=_Y)', f)[0]
        x_string = x_string.replace(',', '.')
        if x_string[0] == '.':
            x_string = x_string[1:]
        x = float(x_string)

        y_string = re.findall('(?<=Y)(.*)(?=_)', f)[0]
        y_string = y_string.replace(',', '.')
        if y_string[0] == '.':
            y_string = y_string[1:]
        y = float(y_string)

        x_coord_list[i] = x
        y_coord_list[i] = y

    x_coords = np.asarray(x_coord_list)
    y_coords = np.asarray(y_coord_list)

    # normalise coordinates, and convert to pixel units

    x_coords = x_coords - (np.amin(x_coords))
    x_coords = x_coords / pixel_size

    y_coords = y_coords - (np.amin(y_coords))
    y_coords = y_coords / pixel_size


    # round coordinates

    x_coords = np.rint(x_coords)
    y_coords = np.rint(y_coords)


    ## now load each image, and get its size, locate particles, add the x and y coord offset, add the area to areas_covered

    areas_covered = []

    locations = locate_feature(dir + '/' + flist[0], particle_size, minmass)

    locations['x'] = locations['x'].add(x_coords[0])
    locations['y'] = locations['y'].add(y_coords[0])


    ## create area
    image = plt.imread(dir + '/' + flist[0])

    xmin = x_coords[0]
    xmax = xmin + image.shape[1]

    ymin = y_coords[0]
    ymax = ymin + image.shape[0]

    area = [xmin, xmax, ymin, ymax]
    areas_covered = areas_covered + [area]


    ## now do remaining images

    for i in range(1, len(flist)):

        #print(i)

        f = locate_feature(dir + '/' + flist[i], particle_size, minmass)

        #print(len(f.index))
        f['x'] = f['x'].add(x_coords[i])
        f['y'] = f['y'].add(y_coords[i])


        ## check if any point is in areas_covered

        # iterate through points and check whether in areas_covered
        for k in range(0, len(f.index)):
            x = f['x'].iloc[k]
            y = f['y'].iloc[k]
            if is_point_in_boxes(areas_covered, [x, y]) == False:
                locations = locations.append(f.iloc[k])


        ## add area to areas_covered

        ## create area
        image = plt.imread(dir + '/' + flist[i])

        xmin = x_coords[i]
        xmax = xmin + image.shape[1]

        ymin = y_coords[i]
        ymax = ymin + image.shape[0]

        area = [xmin, xmax, ymin, ymax]
        areas_covered = areas_covered + [area]


    ## convert locations back to um

    # locations['x'] = locations['x'].multiply(pixel_size)
    # locations['y'] = locations['y'].multiply(pixel_size)

    return locations



def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    #buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    #buf = np.roll(buf, 3, axis=2)


    return buf



def fig2img (fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    im=PIL.Image.frombytes( "RGB", ( w ,h ), buf.tostring())
    return im  #.convert(mode="RGB")

