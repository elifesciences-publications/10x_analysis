
import numpy as np
import matplotlib.pyplot as plt

def interpolate_between_points(p1, p2, n):
    # return just the new points
    return tuple(zip(np.linspace(p1[0], p2[0], n + 2), np.linspace(p1[1], p2[1], n + 2)))[1:-1]


def interpolate_points_between_points_tracks(tracks):
    # iterate through tracks
    for i in range(0, len(tracks)):
        #print(i)

        track = tracks[i]
        gap_list = []
        n_list = []

        # iterate through track, getting list of gaps
        for j in range(1, len(track.times)):

            if track.times[j] != track.times[j - 1] + 1:
                # must be missing frame(s)
                #print('missing frames between', j - 1, 'and', j)

                # number of points to create
                n = int(round((track.times[j] - track.times[j - 1]))) - 1

                gap_list = gap_list + [(track.times[j - 1], track.times[j])]
                n_list = n_list + [n]
                #print(gap_list)
                #print(n_list)

        if gap_list != []:

            # for each gap
            for k in range(0, len(gap_list)):
                end_of_gap_time = gap_list[k][1]
                j = track.times.index(end_of_gap_time)

                # interpolate to create new points
                pos0 = track.positions[j - 1]
                pos1 = track.positions[j]

                n = n_list[k]

                from interpolate_between_points import interpolate_between_points

                new_points = interpolate_between_points(pos0, pos1, n)

                # create new positions
                track.positions = track.positions[:j] + new_points + track.positions[j:]

                # create new times
                new_times = tuple(np.linspace(track.times[j - 1], track.times[j], n + 2))[1:-1]
                track.times = track.times[:j] + new_times + track.times[j:]

            tracks[i] = track

    return tracks


