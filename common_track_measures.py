'''
Calculate common features from

Svensson, C.-M., Medyukhina, A., Belyaev, I., Al-Zaben, N., & Figge, M. T. (2018). Untangling cell tracks:
Quantifying cell migration by time lapse image data analysis.
Cytometry Part A, 93(3), 357–370. https://doi.org/10.1002/cyto.a.23249

'''

import numpy as np
import statistics
import math
from scipy import spatial as sp

def get_displacement(pos0, pos1):
    '''
    Return the displacement between to coordinated in the form (x, y)
    :param pos0:
    :param pos1:
    :return:
    '''

    displacement = ((pos0[0] - pos1[0])**2 + (pos0[1] - pos1[1])**2)**0.5

    return displacement



def total_distance(positions):

    total_distance = 0

    for i in range(1, len(positions)):
        total_distance += get_displacement(positions[i - 1], positions[i])

    return total_distance


def net_distance(positions):

    return get_displacement(positions[0], positions[-1])


def max_distance(positions):
    max_dist = 0
    for i in range(1, len(positions)):
        dist = get_displacement(positions[0], positions[i])
        if dist > max_dist:
            max_dist = dist

    return max_dist


def speed(positions, times):

    speeds = [None] * (len(positions) - 1)
    for i in range(1, len(positions)):
        dist = get_displacement(positions[i], positions[i-1])
        delta_t = times[i] - times[i - 1]
        speeds[i-1] = dist/delta_t

    return speeds

def max_speed(speeds):

    if len(speeds) >= 1:
        return max(speeds)
    else:
        return np.nan

def y_pos(positions):

    y_pos = [item[1] for item in positions]

    return y_pos

def x_pos(positions):

    x_pos = [item[0] for item in positions]

    return x_pos

############################################################
#Piotr's additions:


def average_data(data):
    if len(data) >= 1:
        average_data = statistics.mean(data)
        return average_data
    else:
        return np.nan


def median_data(data):
    if len(data) >= 1:
        median_data = statistics.median(data)
        return median_data
    else:
        return np.nan


def stdev_data(data):
    if len(data) >= 1:
        stdev_data = statistics.stdev(data)
        return stdev_data
    else:
        return np.nan


def relative_turning_angles(x_pos, y_pos):
    rel_angles = []
    for n in range(len(x_pos)-2):
        vector_1 = np.array((x_pos[n+1]-x_pos[n], y_pos[n+1]-y_pos[n]))
        vector_2 = np.array((x_pos[n+2]-x_pos[n+1], y_pos[n+2]-y_pos[n+1]))

        vector_1_len = (vector_1[0]**2+vector_1[1]**2)**0.5
        vector_2_len = (vector_2[0]**2+vector_2[1]**2)**0.5

        theta = np.arccos(np.dot(vector_1, vector_2)/(vector_1_len*vector_2_len))
        # theta_deg = theta*180/math.pi
        rel_angles.append(theta.item())
    return rel_angles


def global_turning_angles(x_pos, y_pos):
    glob_angles = []
    for n in range(len(x_pos) - 2):
        opposite = y_pos[n+1] - y_pos[n]
        hypotenuse = ((x_pos[n] - x_pos[n+1]) ** 2 + (y_pos[n] - y_pos[n+1]) ** 2) ** 0.5
        opposite = opposite / hypotenuse
        theta = math.asin(opposite)
        if (x_pos[n+1] < x_pos[n]) and (y_pos[n+1] > y_pos[n]):
            theta = theta + 0.5*math.pi
        if (x_pos[n+1] < x_pos[n]) and (y_pos[n+1] < y_pos[n]):
            theta = theta - 0.5*math.pi
        glob_angles.append(theta)
    return glob_angles


def distances(positions):

    distances = []
    if len(positions)>0:
        for i in range(1, len(positions)):
            distances.append(get_displacement(positions[i - 1], positions[i]))

        return distances
    else:
        return np.nan


def asphericity(x_pos, y_pos):
    gyration_tensor = np.zeros((2,2))
    x_pos = np.asarray(x_pos)
    y_pos = np.asarray(y_pos)
    gyration_tensor[0, 1] = np.nanmean(x_pos * y_pos) - np.nanmean(x_pos)*np.nanmean(y_pos)
    gyration_tensor[0,0]  = np.nanmean(x_pos*x_pos) - np.nanmean(x_pos)*np.nanmean(x_pos)
    gyration_tensor[1,1]  = np.nanmean(y_pos*y_pos) - np.nanmean(y_pos)*np.nanmean(y_pos)
    gyration_tensor[1,0]  = gyration_tensor[0,1]
    gyration_tensor = np.nan_to_num(gyration_tensor)

    eigenvalues = np.linalg.eig(gyration_tensor)[0] # "0" becasue the function reuturns both eigenvalues and eigenvectors
    eigenvalues = abs(eigenvalues)

    volume_asphericity = np.var(eigenvalues, ddof=1) / (np.mean(eigenvalues)**2 * len(eigenvalues))
    # var() in R calculates sample variance (/n-1), np.var() calculates population /n). To change this, ddof=1.
    return volume_asphericity



def common_track_measures(df):

    # total length of track
    df['total distance'] = df['positions'].apply(total_distance)

    # distance from first to last position
    df['net distance'] = df['positions'].apply(net_distance)

    # max distance reached from starting position
    df['max distance'] = df['positions'].apply(max_distance)

    # meandering index
    df['meandering index'] = df['net distance'] / df['max distance']

    df['outreach ratio'] = df['max distance'] / df['total distance']

    df['speeds'] = df.apply(lambda row: speed(row['positions'], row['times']), axis=1)

    df['max speed'] = df['speeds'].apply(max_speed)

    df['x pos'] = df['positions'].apply(x_pos)
    df['y pos'] = df['positions'].apply(y_pos)

    ##################################################################################
# Piotr's additions:
    df['average speed'] = df['speeds'].apply(average_data)
    df['median speed'] = df['speeds'].apply(median_data)
    df['speed StDev'] = df['speeds'].apply(stdev_data)
    df['relative turning angles'] = df.apply(lambda row: relative_turning_angles(row['x pos'], row['y pos']), axis=1)
    df['global turning angles'] = df.apply(lambda row: global_turning_angles(row['x pos'], row['y pos']), axis=1)
    df['average relative turning angles'] = df['relative turning angles'].apply(average_data)
    df['median relative turning angles'] = df['relative turning angles'].apply(median_data)
    df['distances'] = df['positions'].apply(distances)
    df['confinement ratio'] = df['net distance'] / df['total distance']
    df['asphericity'] = df.apply(lambda row: asphericity(row['x pos'], row['y pos']), axis=1)
    df['average global turning angle'] = df['global turning angles'].apply(average_data)
    df['median global turning angle'] = df['global turning angles'].apply(median_data)
    df['sd of relative turning angle'] = df['relative turning angles'].apply(stdev_data)
    df['sd of global turning angle'] = df['global turning angles'].apply(stdev_data)

    return df
