'''
plot tracks from track_list.pickle onto previously stitched images, and create gif
'''

import pickle
import re
import os
from os.path import isfile, join
import cv2 as cv
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np

class Track:
    def __init__(self, ID, times, positions):
        self.ID = ID
        self.times = times  # tuple of times
        self.positions = positions  # tuple of positions (positions are also tuple (x, y) )

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





#211119
# path to folder of stitched ims
# w572_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/w572_stitched'
# # path to tracks.pickle
# tracks_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/track_list.pickle'


# 291119

w572_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/w572_stitched'
tracks_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/pos_/track_list.pickle'



# pickle in tracks
pickle_in = open(tracks_path, 'rb')
track_list = pickle.load(pickle_in)
pickle_in.close()

# factor by which imaged were downsampled during stitching
downsample_factor = 1


## create ordered list of stitched images

flist = [f for f in os.listdir(w572_dir) if isfile(join(w572_dir, f))]

# include only files which include 'w572_top'
flist = [f for f in flist if 'w572_' in f]

print(flist)

# to contain datatime objects
deltaT_list = [None]*len(flist)

for i in range(0, len(flist)):
    deltaT_list[i] = int(re.findall('(?<=deltaT)(.*)(?=secs_)', flist[i])[0])
print(deltaT_list)

# sort flist based on deltaT list
deltaT_list, flist = zip(*sorted(zip(deltaT_list, flist)))  # OMG, it works!
print(flist)


## iterate through ims and draw tracks

images = []

for i in range(0, 1):
    print('working on image ', i+1, ' of ', len(flist))

    image = cv.imread(w572_dir + '/' + flist[i], -1)  # using cv2 https://stackoverflow.com/questions/18446804/python-read-and-write-tiff-16-bit-three-channel-colour-images

    #image = np.flip(image)
    image = np.rot90(image, 2)

    figure, ax = plt.subplots()
    figure.set_size_inches(10, 10)
    ax.imshow(image, vmin=104, vmax=118)

    # draw tracks
    for track in track_list:

        # in this case, times is expressed as frame number, starting from zero
        if (i in track.times) and (0 in track.times):

            # get index of i in track.times
            curr_frame_index = track.times.index(i)

            # create lists of x and y
            x_list = [item[0] for item in track.positions[0:curr_frame_index]]
            y_list = [item[1] for item in track.positions[0:curr_frame_index]]

            # divide x and y by downsample_factor
            x_list = [x/downsample_factor for x in x_list]
            y_list = [x/downsample_factor for x in y_list]

            plt.plot(x_list, y_list, color='white', alpha=0.2)

    ax.set_aspect('equal')
    im = fig2img(figure)
    # im.show()
    images.append(im)
    plt.close()

images[0].save(w572_dir + 'tracks.gif', format='GIF', append_images=images[1:], save_all=True, duration=100, loop=0)