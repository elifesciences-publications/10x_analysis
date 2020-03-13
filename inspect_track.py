'''
plot single track for inspection
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



# 211119
# path to folder of stitched ims
#w572_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/w572_stitched'
# path to tracks.pickle
#tracks_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/track_list.pickle'

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

## get tracks that start from frame 0
tracks_from_start_list = []
for track in track_list:
    if (0 in track.times) and (len(track.times) >= 500):
        tracks_from_start_list = tracks_from_start_list + [track]

print(len(tracks_from_start_list), 'tracks')



## go through tracks that don't start from frame zero,
## and see if they start near a track that starts from zero

def get_displacement(pos0, pos1):
    '''
    Return the displacement between to coordinated in the form (x, y)
    :param pos0:
    :param pos1:
    :return:
    '''

    displacement = ((pos0[0] - pos1[0])**2 + (pos0[1] - pos1[1])**2)**0.5

    return displacement


new_track_list = []
for track in track_list:
    if (track.times[0] != 0) and (len(track.times) >= 20):
        new_track_list = new_track_list + [track]

possible_dividers = []
possible_daughters = []
for track1 in new_track_list:
    start_time = track1.times[0]
    start_pos = track1.positions[0]

    for track2 in tracks_from_start_list:
        if start_time in track2.times:
            ind = track2.times.index(start_time)
            pos = track2.positions[ind]

            distance = get_displacement(start_pos, pos)
            if distance <= 20:
                possible_dividers = possible_dividers + [track2]
                possible_daughters = possible_daughters + [track1]


print(len(possible_dividers), 'possible dividers')

track_index = 0

track = possible_dividers[track_index]


## create max Z projection of track

# create axis limits
x_list = [item[0] for item in track.positions]
y_list = [item[1] for item in track.positions]

border = 10
x_min = int(min(x_list) - border)
x_max = int(max(x_list) + border)
y_min = int(min(y_list) - border)
y_max = int(max(y_list) + border)

# create empty 3d np array
x_dim = x_max - x_min
y_dim = y_max - y_min
z_dim = len(track.times)

stack = np.zeros((x_dim, y_dim, z_dim))

# for frame in track.times:
#     print(frame)
#     image = cv.imread(w572_dir + '/' + flist[frame], -1)  # using cv2 https://stackoverflow.com/questions/18446804/python-read-and-write-tiff-16-bit-three-channel-colour-images
#
#     image = np.flip(image, axis=0)
#     image = np.rot90(image, 2)
#
#     stack[:, :, int(frame)] = image[x_min:x_max, y_min:y_max]
#
# max_proj = np.amax(stack, axis=2)
# plt.imshow(max_proj)
# plt.show()

# plot possible dividers
for track in possible_dividers:
    x_list = [item[0] for item in track.positions]
    y_list = [item[1] for item in track.positions]
    plt.plot(x_list, y_list, c='k')

for track in possible_daughters:
    x_list = [item[0] for item in track.positions]
    y_list = [item[1] for item in track.positions]
    plt.plot(x_list, y_list, c='r')

plt.show()



# track_ID = 55 #112
# track = 0
# for item in new_track_list:
#     if item.ID == track_ID:
#         track = item

## iterate through ims and draw tracks for gif
counter = 1
for track in tracks_from_start_list:
    print('track', counter, 'of', len(tracks_from_start_list))
    counter += 1
    images = []

    for i in range(0, len(flist)):
        print('working on image ', i+1, ' of ', len(flist))

        image = cv.imread(w572_dir + '/' + flist[i], -1)  # using cv2 https://stackoverflow.com/questions/18446804/python-read-and-write-tiff-16-bit-three-channel-colour-images

        image = np.flip(image, axis=0)
        image = np.rot90(image, 2)

        figure, ax = plt.subplots()
        figure.set_size_inches(4, 4)
        ax.imshow(image, vmin=104, vmax=118)
        #ax.imshow(image)

        # create axis limits
        x_list = [item[0] for item in track.positions]
        y_list = [item[1] for item in track.positions]

        border = 10
        x_min = min(x_list) - border
        x_max = max(x_list) + border
        y_min = min(y_list) - border
        y_max = max(y_list) + border



        # draw track
        # in this case, times is expressed as frame number, starting from zero
        if i in track.times:

            # get index of i in track.times
            time = i
            while (time not in track.times) and (time >= 0):
                time = time - 1
            curr_frame_index = track.times.index(time)

            # create lists of x and y
            x_list = [item[0] for item in track.positions[0:curr_frame_index]]
            y_list = [item[1] for item in track.positions[0:curr_frame_index]]



            # divide x and y by downsample_factor
            x_list = [x/downsample_factor for x in x_list]
            y_list = [x/downsample_factor for x in y_list]

            plt.plot(x_list, y_list, color='white', alpha=0.8)
            # plt.xlim((x_min, x_max))
            # plt.ylim((y_min, y_max))


        if i > track.times[-1]:
            break

        ax.set_aspect('equal')
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_max, y_min))
        ax.set_title('ID ' + str(track.ID) + ', frame ' + str(i))
        im = fig2img(figure)
        # im.show()
        images.append(im)
        plt.close()


    stacks_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/291119_JR5_AX3_14hr_dediff_JN/stacks'

    images[0].save(stacks_dir + '/ID_' + str(track.ID) + '_divider_track.tiff', format='TIFF', append_images=images[1:], save_all=True, duration=100, loop=0)


