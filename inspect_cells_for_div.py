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


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()






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

def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

# path to folder of stitched ims
w572_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/w572_stitched'

# path to tracks.pickle
tracks_path = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/pos_/track_list.pickle'


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
    if (0 in track.times) and (len(track.times) >= 400):
        tracks_from_start_list = tracks_from_start_list + [track]

print(len(tracks_from_start_list), 'tracks')



from track_list_to_df import track_list_to_df
df = track_list_to_df(tracks_from_start_list, is_model=False)


# for each row
for i in range(0, 1):

    images = []
    row = df.iloc[i]

    frames = row['times']

    # create axis limits
    x_list = [item[0] for item in row['positions']]
    y_list = [item[1] for item in row['positions']]

    border = 10
    x_min = min(x_list) - border
    x_max = max(x_list) + border
    y_min = min(y_list) - border
    y_max = max(y_list) + border



    for j in range(0, len(frames)):

        print('preparing slice', j, 'of', len(frames))
        frame = int(frames[j])
        image = cv.imread(w572_dir + '/' + flist[frame], -1)  # using cv2 https://stackoverflow.com/questions/18446804/python-read-and-write-tiff-16-bit-three-channel-colour-images

        image = np.flip(image, axis=0)
        image = np.rot90(image, 2)


        figure, ax = plt.subplots()
        figure.set_size_inches(4, 4)
        ax.imshow(image, vmin=104, vmax=118)

        # plot track
        plt.plot(x_list[0:j], y_list[0:j], color='white', alpha=0.8)


        ax.set_aspect('equal')
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_max, y_min))
        im = fig2img(figure)
        im = PIL2array(im)
        # im.show()
        images.append(im)
        plt.close()

    stacks_dir = '/media/jacob/data/Chubb_lab/notebook/why_become_a_stalk_cell/imaging_dediff_cell_and_measuring_CryS_/cont_for_manual_tracking/10x/211119_14hr_dediff_JN_protocol_AX3_JR5_mix_2min_int_10x_bottom/stacks'

    images[0].save(stacks_dir + '/ID_' +  str(track.ID) + '_track.tiff', format='TIFF', append_images=images[1:],
                   save_all=True, duration=100, loop=0)



