import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from dtw import *
from fastdtw import fastdtw
from itertools import combinations
from scipy.signal import find_peaks
import pandas as pd

length_APC_213 = 151
length_KRAS_12 = 148
length_PIK3CA_542 = 144
length_PIK3CA_1047 = 142
length_TP53_273 = 140
length_BRAF_600 = 139
length_ERBB2_842 = 135
length_TP53_196 = 133
length_ERBB2_777 = 132
length_APC_1114 = 122
length_KRAS_61 = 120
length_APC_1450 = 118
length_KRAS_146 = 114
length_ERBB2_310 = 113
length_APC_876 = 106

# Set font size globally
fontsize = 14

def dtw_distance(x, y):
    # Calculate dynamic time warping distance between sequences x and y
    cost_matrix = dtw_cost_matrix(x, y)
    n, m = len(x), len(y)
    accumulated_cost = np.zeros((n, m))
    accumulated_cost[0, 0] = cost_matrix[0, 0]
    for i in range(1, n):
        accumulated_cost[i, 0] = accumulated_cost[i-1, 0] + cost_matrix[i, 0]
    for j in range(1, m):
        accumulated_cost[0, j] = accumulated_cost[0, j-1] + cost_matrix[0, j]
    for i in range(1, n):
        for j in range(1, m):
            accumulated_cost[i, j] = cost_matrix[i, j] + min(accumulated_cost[i-1, j], 
                                                             accumulated_cost[i, j-1], 
                                                             accumulated_cost[i-1, j-1])
    return accumulated_cost[-1, -1]

def dtw_distance_with_indices(x, y, start_index_x=0, end_index_x=None, start_index_y=0, end_index_y=None):
    # Calculate the DTW distance between sequences x and y, considering start and end indices
    if end_index_x is None:
        end_index_x = len(x) - 1
    if end_index_y is None:
        end_index_y = len(y) - 1
    x_subseq = x[start_index_x:end_index_x + 1]
    y_subseq = y[start_index_y:end_index_y + 1]
    return dtw_distance(x_subseq, y_subseq)


def dtw_cost_matrix(x, y, start_index_x=0, end_index_x=None, start_index_y=0, end_index_y=None):
    # Calculate the cost matrix for sequences x and y, considering start and end indices
    if end_index_x is None:
        end_index_x = len(x) - 1
    if end_index_y is None:
        end_index_y = len(y) - 1
    n, m = end_index_x - start_index_x + 1, end_index_y - start_index_y + 1
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = abs(x[start_index_x + i] - y[start_index_y + j])  # Calculate cost based on absolute difference
            if cost_matrix[i, j] <= 10:
                cost_matrix[i, j] = abs(x[start_index_x + i] - y[start_index_y + j])
            else:
                cost_matrix[i, j] = 100    
    return cost_matrix

# def dtw_cost_matrix(x, y, start_index_x=0, end_index_x=None, start_index_y=0, end_index_y=None):
#     # Calculate the cost matrix for sequences x and y, considering start and end indices
#     if end_index_x is None:
#         end_index_x = len(x) - 1
#     if end_index_y is None:
#         end_index_y = len(y) - 1

#     n, m = end_index_x - start_index_x + 1, end_index_y - start_index_y + 1
#     cost_matrix = np.zeros((n, m))
    
#     for i in range(n):
#         for j in range(m):
#             base_cost = abs(x[start_index_x + i] - y[start_index_y + j])
#             if base_cost <= 10:
#                 cost = base_cost
#             else:
#                 cost = 100
#             weight = 1 / (1 + 1 * (i + j))  # Example: inverse decay
#             cost_matrix[i, j] = cost * weight  # Apply weight to the cost
#     return cost_matrix

def dtw_path(cost_matrix):
    # Calculate the optimal path for the given cost matrix
    n, m = cost_matrix.shape
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if cost_matrix[i-1, j] == min(cost_matrix[i-1, j-1], cost_matrix[i-1, j], cost_matrix[i, j-1]):
                i -= 1
            elif cost_matrix[i, j-1] == min(cost_matrix[i-1, j-1], cost_matrix[i-1, j], cost_matrix[i, j-1]):
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
    return np.array(path[::-1])

def plot_dtw_path(cost_matrix, path, xlabel='Sequence Y', ylabel='Sequence X', title='DTW Cost Matrix'):
    # Plot the cost matrix and the optimal path
    plt.imshow(cost_matrix, origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Cost')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i, j in path:
        plt.plot(j, i, marker='o', color='red')  # Plotting the path
    for i in range(len(path) - 1):
        plt.plot([path[i][1], path[i+1][1]], [path[i][0], path[i+1][0]], color='red')  # Plotting the path segments
    # plt.show()

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/20250618_补充组织解谱/20250618_tube1e-15P-WB-0.5XHB-250115-S9.xlsx", header=None)

# sequence30A = df.iloc[1, 37:1136].tolist()
# sequence30G = df.iloc[2, 37:1136].tolist()
# sequence30C = df.iloc[3, 37:1136].tolist()
# sequence30T = df.iloc[4, 37:1136].tolist()

sequence30A = df.iloc[1, 0:1099].tolist()
sequence30G = df.iloc[2, 0:1099].tolist()
sequence30C = df.iloc[3, 0:1099].tolist()
sequence30T = df.iloc[4, 0:1099].tolist()
# Convert sequence30A to a numpy array
sequence30A = np.array(sequence30A)
sequence30G = np.array(sequence30G)
sequence30C = np.array(sequence30C)
sequence30T = np.array(sequence30T)

sequence30A = np.concatenate((np.zeros(2),sequence30A))
sequence30G = np.concatenate((np.zeros(2),sequence30G))
sequence30C = np.concatenate((np.zeros(2),sequence30C))
sequence30T = np.concatenate((np.zeros(2),sequence30T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/CRC-11plex-200nM-NB-TJG.APC-213-RP.xlsx", header=None)

sequence1A = df.iloc[1, 60:1159].tolist()
sequence1G = df.iloc[2, 60:1159].tolist()
sequence1C = df.iloc[3, 60:1159].tolist()
sequence1T = df.iloc[4, 60:1159].tolist()

sequence1A = np.array(sequence1A)
sequence1G = np.array(sequence1G)
sequence1C = np.array(sequence1C)
sequence1T = np.array(sequence1T)

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/Hieffblue_CRC15plex_KRAS-12.xlsx", header=None)

sequence2A = df.iloc[1, 0:1200].tolist()
sequence2G = df.iloc[2, 0:1099].tolist()
sequence2C = df.iloc[3, 0:1099].tolist()
sequence2T = df.iloc[4, 0:1099].tolist()

sequence2A = np.array(sequence2A)
sequence2G = np.array(sequence2G)
sequence2C = np.array(sequence2C)
sequence2T = np.array(sequence2T)

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/Hieffblue_CRC15plex_PIK3CA-542.xlsx", header=None)

sequence3A = df.iloc[1, 0:1099].tolist()
sequence3G = df.iloc[2, 0:1099].tolist()
sequence3C = df.iloc[3, 0:1099].tolist()
sequence3T = df.iloc[4, 0:1099].tolist()

sequence3A = np.array(sequence3A)
sequence3G = np.array(sequence3G)
sequence3C = np.array(sequence3C)
sequence3T = np.array(sequence3T)

sequence3A = np.concatenate((np.zeros(1),sequence3A))
sequence3G = np.concatenate((np.zeros(1),sequence3G))
sequence3C = np.concatenate((np.zeros(1),sequence3C))
sequence3T = np.concatenate((np.zeros(1),sequence3T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/Hieffblue_CRC15plex_PIK3CA-1047.xlsx", header=None)

sequence4A = df.iloc[1, 0:1099].tolist()
sequence4G = df.iloc[2, 0:1099].tolist()
sequence4C = df.iloc[3, 0:1099].tolist()
sequence4T = df.iloc[4, 0:1099].tolist()

sequence4A = np.array(sequence4A)
sequence4G = np.array(sequence4G)
sequence4C = np.array(sequence4C)
sequence4T = np.array(sequence4T)

sequence4A = np.concatenate((np.zeros(1),sequence4A))
sequence4G = np.concatenate((np.zeros(1),sequence4G))
sequence4C = np.concatenate((np.zeros(1),sequence4C))
sequence4T = np.concatenate((np.zeros(1),sequence4T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/Hieffblue_CRC15plex_TP53-273.xlsx", header=None)

sequence5A = df.iloc[1, 0:1099].tolist()
sequence5G = df.iloc[2, 0:1099].tolist()
sequence5C = df.iloc[3, 0:1099].tolist()
sequence5T = df.iloc[4, 0:1099].tolist()

sequence5A = np.array(sequence5A)
sequence5G = np.array(sequence5G)
sequence5C = np.array(sequence5C)
sequence5T = np.array(sequence5T)

sequence5A = np.concatenate((np.zeros(1),sequence5A))
sequence5G = np.concatenate((np.zeros(1),sequence5G))
sequence5C = np.concatenate((np.zeros(1),sequence5C))
sequence5T = np.concatenate((np.zeros(1),sequence5T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/Hieffblue_CRC15plex_BRAF-600.xlsx", header=None)

sequence6A = df.iloc[1, 0:1099].tolist()
sequence6G = df.iloc[2, 0:1099].tolist()
sequence6C = df.iloc[3, 0:1099].tolist()
sequence6T = df.iloc[4, 0:1099].tolist()

sequence6A = np.array(sequence6A)
sequence6G = np.array(sequence6G)
sequence6C = np.array(sequence6C)
sequence6T = np.array(sequence6T)

sequence6A = np.concatenate((np.zeros(1),sequence6A))
sequence6G = np.concatenate((np.zeros(1),sequence6G))
sequence6C = np.concatenate((np.zeros(1),sequence6C))
sequence6T = np.concatenate((np.zeros(1),sequence6T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/Hieffblue_CRC15plex_ERBB2-842.xlsx", header=None)

sequence7A = df.iloc[1, 0:1099].tolist()
sequence7G = df.iloc[2, 0:1099].tolist()
sequence7C = df.iloc[3, 0:1099].tolist()
sequence7T = df.iloc[4, 0:1099].tolist()

sequence7A = np.array(sequence7A)
sequence7G = np.array(sequence7G)
sequence7C = np.array(sequence7C)
sequence7T = np.array(sequence7T)

sequence7A = np.concatenate((np.zeros(1),sequence7A))
sequence7G = np.concatenate((np.zeros(1),sequence7G))
sequence7C = np.concatenate((np.zeros(1),sequence7C))
sequence7T = np.concatenate((np.zeros(1),sequence7T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/Hieffblue_CRC15plex_TP53-196.xlsx", header=None)

sequence8A = df.iloc[1, 0:1099].tolist()
sequence8G = df.iloc[2, 0:1099].tolist()
sequence8C = df.iloc[3, 0:1099].tolist()
sequence8T = df.iloc[4, 0:1099].tolist()

sequence8A = np.array(sequence8A)
sequence8G = np.array(sequence8G)
sequence8C = np.array(sequence8C)
sequence8T = np.array(sequence8T)

sequence8A = np.concatenate((np.zeros(1),sequence8A))
sequence8G = np.concatenate((np.zeros(1),sequence8G))
sequence8C = np.concatenate((np.zeros(1),sequence8C))
sequence8T = np.concatenate((np.zeros(1),sequence8T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/Hieffblue_CRC15plex_ERBB2-777.xlsx", header=None)

sequence9A = df.iloc[1, 0:1099].tolist()
sequence9G = df.iloc[2, 0:1099].tolist()
sequence9C = df.iloc[3, 0:1099].tolist()
sequence9T = df.iloc[4, 0:1099].tolist()

sequence9A = np.array(sequence9A)
sequence9G = np.array(sequence9G)
sequence9C = np.array(sequence9C)
sequence9T = np.array(sequence9T)

sequence9A = np.concatenate((np.zeros(13),sequence9A))
sequence9G = np.concatenate((np.zeros(13),sequence9G))
sequence9C = np.concatenate((np.zeros(13),sequence9C))
sequence9T = np.concatenate((np.zeros(13),sequence9T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/Hieffblue_CRC15plex_APC-1114.xlsx", header=None)

sequence10A = df.iloc[1, 0:1099].tolist()
sequence10G = df.iloc[2, 0:1099].tolist()
sequence10C = df.iloc[3, 0:1099].tolist()
sequence10T = df.iloc[4, 0:1099].tolist()

sequence10A = np.array(sequence10A)
sequence10G = np.array(sequence10G)
sequence10C = np.array(sequence10C)
sequence10T = np.array(sequence10T)

sequence10A = np.concatenate((np.zeros(1),sequence10A))
sequence10G = np.concatenate((np.zeros(1),sequence10G))
sequence10C = np.concatenate((np.zeros(1),sequence10C))
sequence10T = np.concatenate((np.zeros(1),sequence10T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/Hieffblue_CRC15plex_KRAS-61.xlsx", header=None)

# Read the first row (index 0) and store it in an array
sequence11A = df.iloc[1, 0:1099].tolist()
sequence11G = df.iloc[2, 0:1099].tolist()
sequence11C = df.iloc[3, 0:1099].tolist()
sequence11T = df.iloc[4, 0:1099].tolist()

sequence11A = np.array(sequence11A)
sequence11G = np.array(sequence11G)
sequence11C = np.array(sequence11C)
sequence11T = np.array(sequence11T)

sequence11A = np.concatenate((np.zeros(1),sequence11A))
sequence11G = np.concatenate((np.zeros(1),sequence11G))
sequence11C = np.concatenate((np.zeros(1),sequence11C))
sequence11T = np.concatenate((np.zeros(1),sequence11T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/Hieffblue_CRC15plex_APC-1450.xlsx", header=None)

sequence12A = df.iloc[1, 0:1099].tolist()
sequence12G = df.iloc[2, 0:1099].tolist()
sequence12C = df.iloc[3, 0:1099].tolist()
sequence12T = df.iloc[4, 0:1099].tolist()

sequence12A = np.array(sequence12A)
sequence12G = np.array(sequence12G)
sequence12C = np.array(sequence12C)
sequence12T = np.array(sequence12T)

sequence12A = np.concatenate((np.zeros(1),sequence12A))
sequence12G = np.concatenate((np.zeros(1),sequence12G))
sequence12C = np.concatenate((np.zeros(1),sequence12C))
sequence12T = np.concatenate((np.zeros(1),sequence12T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/Hieffblue_CRC15plex_KRAS-146.xlsx", header=None)

sequence13A = df.iloc[1, 0:1099].tolist()
sequence13G = df.iloc[2, 0:1099].tolist()
sequence13C = df.iloc[3, 0:1099].tolist()
sequence13T = df.iloc[4, 0:1099].tolist()

sequence13A = np.array(sequence13A)
sequence13G = np.array(sequence13G)
sequence13C = np.array(sequence13C)
sequence13T = np.array(sequence13T)

sequence13A = np.concatenate((np.zeros(1),sequence13A))
sequence13G = np.concatenate((np.zeros(1),sequence13G))
sequence13C = np.concatenate((np.zeros(1),sequence13C))
sequence13T = np.concatenate((np.zeros(1),sequence13T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/Hieffblue_CRC15plex_ERBB2-310.xlsx", header=None)

sequence14A = df.iloc[1, 0:1099].tolist()
sequence14G = df.iloc[2, 0:1099].tolist()
sequence14C = df.iloc[3, 0:1099].tolist()
sequence14T = df.iloc[4, 0:1099].tolist()

sequence14A = np.array(sequence14A)
sequence14G = np.array(sequence14G)
sequence14C = np.array(sequence14C)
sequence14T = np.array(sequence14T)

sequence14A = np.concatenate((np.zeros(2),sequence14A))
sequence14G = np.concatenate((np.zeros(2),sequence14G))
sequence14C = np.concatenate((np.zeros(2),sequence14C))
sequence14T = np.concatenate((np.zeros(2),sequence14T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/DTW python/Hieffblue_CRC15plex_APC-876.xlsx", header=None)

sequence15A = df.iloc[1, 0:1099].tolist()
sequence15G = df.iloc[2, 0:1099].tolist()
sequence15C = df.iloc[3, 0:1099].tolist()
sequence15T = df.iloc[4, 0:1099].tolist()

sequence15A = np.array(sequence15A)
sequence15G = np.array(sequence15G)
sequence15C = np.array(sequence15C)
sequence15T = np.array(sequence15T)

sequence15A = np.concatenate((np.zeros(1),sequence15A))
sequence15G = np.concatenate((np.zeros(1),sequence15G))
sequence15C = np.concatenate((np.zeros(1),sequence15C))
sequence15T = np.concatenate((np.zeros(1),sequence15T))

sequence0A = sequence1A*0
sequence0G = sequence1G*0
sequence0C = sequence1C*0
sequence0T = sequence1T*0

max_sequence1 = max(max(sequence1A, default=0),max(sequence1G, default=0), max(sequence1C, default=0),max(sequence1T, default=0))
max_sequence2 = max(max(sequence2A, default=0),max(sequence2G, default=0), max(sequence2C, default=0),max(sequence2T, default=0))
max_sequence3 = max(max(sequence3A, default=0),max(sequence3G, default=0), max(sequence3C, default=0),max(sequence3T, default=0))
max_sequence4 = max(max(sequence4A, default=0),max(sequence4G, default=0), max(sequence4C, default=0),max(sequence4T, default=0))
max_sequence5 = max(max(sequence5A, default=0),max(sequence5G, default=0), max(sequence5C, default=0),max(sequence5T, default=0))
max_sequence6 = max(max(sequence6A, default=0),max(sequence6G, default=0), max(sequence6C, default=0),max(sequence6T, default=0))
max_sequence7 = max(max(sequence7A, default=0),max(sequence7G, default=0), max(sequence7C, default=0),max(sequence7T, default=0))
max_sequence8 = max(max(sequence8A, default=0),max(sequence8G, default=0), max(sequence8C, default=0),max(sequence8T, default=0))
max_sequence9 = max(max(sequence9A, default=0),max(sequence9G, default=0), max(sequence9C, default=0),max(sequence9T, default=0))
max_sequence10 = max(max(sequence10A, default=0),max(sequence10G, default=0), max(sequence10C, default=0),max(sequence10T, default=0))
max_sequence11 = max(max(sequence11A, default=0),max(sequence11G, default=0), max(sequence11C, default=0),max(sequence11T, default=0))
max_sequence12= max(max(sequence12A, default=0),max(sequence12G, default=0), max(sequence12C, default=0),max(sequence12T, default=0))
max_sequence13 = max(max(sequence13A, default=0),max(sequence13G, default=0), max(sequence13C, default=0),max(sequence13T, default=0))
max_sequence14= max(max(sequence14A, default=0),max(sequence14G, default=0), max(sequence14C, default=0),max(sequence14T, default=0))
max_sequence15= max(max(sequence15A, default=0),max(sequence15G, default=0), max(sequence15C, default=0),max(sequence15T, default=0))
max_sequence30= max(max(sequence30A, default=0),max(sequence30G, default=0), max(sequence30C, default=0),max(sequence30T, default=0))

sequence1A = sequence1A/max_sequence1*100
sequence2A = sequence2A/max_sequence2*100
sequence3A = sequence3A/max_sequence3*100
sequence4A = sequence4A/max_sequence4*100
sequence5A = sequence5A/max_sequence5*100
sequence6A = sequence6A/max_sequence6*100
sequence1G = sequence1G/max_sequence1*100
sequence2G = sequence2G/max_sequence2*100
sequence3G = sequence3G/max_sequence3*100
sequence4G = sequence4G/max_sequence4*100
sequence5G = sequence5G/max_sequence5*100
sequence6G = sequence6G/max_sequence6*100
sequence1C = sequence1C/max_sequence1*100
sequence2C = sequence2C/max_sequence2*100
sequence3C = sequence3C/max_sequence3*100
sequence4C = sequence4C/max_sequence4*100
sequence5C = sequence5C/max_sequence5*100
sequence6C = sequence6C/max_sequence6*100
sequence1T = sequence1T/max_sequence1*100
sequence2T = sequence2T/max_sequence2*100
sequence3T = sequence3T/max_sequence3*100
sequence4T = sequence4T/max_sequence4*100
sequence5T = sequence5T/max_sequence5*100
sequence6T = sequence6T/max_sequence6*100
sequence7A = sequence7A/max_sequence7*100
sequence7G = sequence7G/max_sequence7*100
sequence7C = sequence7C/max_sequence7*100
sequence7T = sequence7T/max_sequence7*100
sequence8A = sequence8A/max_sequence8*100
sequence8G = sequence8G/max_sequence8*100
sequence8C = sequence8C/max_sequence8*100
sequence8T = sequence8T/max_sequence8*100
sequence9A = sequence9A/max_sequence9*100
sequence9G = sequence9G/max_sequence9*100
sequence9C = sequence9C/max_sequence9*100
sequence9T = sequence9T/max_sequence9*100
sequence10A = sequence10A/max_sequence10*100
sequence10G = sequence10G/max_sequence10*100
sequence10C = sequence10C/max_sequence10*100
sequence10T = sequence10T/max_sequence10*100
sequence11A = sequence11A/max_sequence11*100
sequence11G = sequence11G/max_sequence11*100
sequence11C = sequence11C/max_sequence11*100
sequence11T = sequence11T/max_sequence11*100
sequence12A = sequence12A/max_sequence12*100
sequence12G = sequence12G/max_sequence12*100
sequence12C = sequence12C/max_sequence12*100
sequence12T = sequence12T/max_sequence12*100
sequence13A = sequence13A/max_sequence13*100
sequence13G = sequence13G/max_sequence13*100
sequence13C = sequence13C/max_sequence13*100
sequence13T = sequence13T/max_sequence13*100
sequence14A = sequence14A/max_sequence14*100
sequence14G = sequence14G/max_sequence14*100
sequence14C = sequence14C/max_sequence14*100
sequence14T = sequence14T/max_sequence14*100
sequence15A = sequence15A/max_sequence15*100
sequence15G = sequence15G/max_sequence15*100
sequence15C = sequence15C/max_sequence15*100
sequence15T = sequence15T/max_sequence15*100
sequence30A = sequence30A/max_sequence30*100*1.5
sequence30G = sequence30G/max_sequence30*100*1.5
sequence30C = sequence30C/max_sequence30*100*1.5
sequence30T = sequence30T/max_sequence30*100*1.5

# Sort peaks above threshold value based on their peak position
def find_and_sort_peaks(sequence_data, segment_start, segment_end, height_threshold):
    peaks_all = []
    for sequence, name in sequence_data:
        peaks, _ = find_peaks(sequence[segment_start:segment_end], height=height_threshold, distance = 8)
        peaks_all += [(x, name) for x in peaks]
    sorted_peaks = sorted(peaks_all)
    return sorted_peaks

def print_peaks(sorted_peaks):
    for i, (peak, seq_name) in enumerate(sorted_peaks):
        print(f"{seq_name} has the {i+1}st peak at index {peak}")

def get_sorted_peak_positions(sorted_peaks):
    # Extract peak and sort
    peak_positions = sorted([peak for peak, _ in sorted_peaks])
    return [int(x) for x in peak_positions]

def get_sorted_bases(sorted_peaks):
    # Create a list to hold the sorted bases
    sorted_bases = []
    
    # Iterate over sorted peaks and extract base names (seq_name)
    for peak, seq_name in sorted_peaks:
        sorted_bases.append(seq_name)  # Append base name to list
    
    return sorted_bases

# Define the sequences for each channel
sequencesA = [sequence1A, sequence2A, sequence3A, sequence4A, sequence5A, sequence6A, sequence7A, sequence8A, sequence9A, sequence10A, sequence11A, sequence12A, sequence13A, sequence14A,sequence15A, sequence0A]  # Define all sequences for channel A
sequencesG = [sequence1G, sequence2G, sequence3G, sequence4G, sequence5G, sequence6G, sequence7G, sequence8G, sequence9G, sequence10G, sequence11G, sequence12G, sequence13G, sequence14G,sequence15G, sequence0G]  # Define all sequences for channel G
sequencesC = [sequence1C, sequence2C, sequence3C, sequence4C, sequence5C, sequence6C, sequence7C, sequence8C, sequence9C, sequence10C, sequence11C, sequence12C, sequence13C, sequence14C,sequence15C, sequence0C]  # Define all sequences for channel C
sequencesT = [sequence1T, sequence2T, sequence3T, sequence4T, sequence5T, sequence6T, sequence7T, sequence8A, sequence9T, sequence10T, sequence11T, sequence12T, sequence13T, sequence14T,sequence15T, sequence0T]  # Define all sequences for channel T

# Set parameters
segment_start = 20
segment_end = 100
height_threshold = 10

sequences = [
    [(sequence1A, 'A'), (sequence1G, 'G'), (sequence1C, 'C'), (sequence1T, 'T')],
    [(sequence2A, 'A'), (sequence2G, 'G'), (sequence2C, 'C'), (sequence2T, 'T')],
    [(sequence3A, 'A'), (sequence3G, 'G'), (sequence3C, 'C'), (sequence3T, 'T')],
    [(sequence4A, 'A'), (sequence4G, 'G'), (sequence4C, 'C'), (sequence4T, 'T')],
    [(sequence5A, 'A'), (sequence5G, 'G'), (sequence5C, 'C'), (sequence5T, 'T')],
    [(sequence6A, 'A'), (sequence6G, 'G'), (sequence6C, 'C'), (sequence6T, 'T')],
    [(sequence7A, 'A'), (sequence7G, 'G'), (sequence7C, 'C'), (sequence7T, 'T')],
    [(sequence8A, 'A'), (sequence8G, 'G'), (sequence8C, 'C'), (sequence8T, 'T')],
    [(sequence9A, 'A'), (sequence9G, 'G'), (sequence9C, 'C'), (sequence9T, 'T')],
    [(sequence10A, 'A'), (sequence10G, 'G'), (sequence10C, 'C'), (sequence10T, 'T')],
    [(sequence11A, 'A'), (sequence11G, 'G'), (sequence11C, 'C'), (sequence11T, 'T')],
    [(sequence12A, 'A'), (sequence12G, 'G'), (sequence12C, 'C'), (sequence12T, 'T')],
    [(sequence13A, 'A'), (sequence13G, 'G'), (sequence13C, 'C'), (sequence13T, 'T')],
    [(sequence14A, 'A'), (sequence14G, 'G'), (sequence14C, 'C'), (sequence14T, 'T')],
    [(sequence15A, 'A'), (sequence15G, 'G'), (sequence15C, 'C'), (sequence15T, 'T')],    
    [(sequence30A, 'A'), (sequence30G, 'G'), (sequence30C, 'C'), (sequence30T, 'T')]
]

sorted_peak_positions_all = []
sorted_bases_all = []
for i, seq_set in enumerate(sequences):
    sorted_peaks = find_and_sort_peaks(seq_set, segment_start, segment_end, height_threshold)
    print(f"\nPeaks for Sequence Set {i+1}:")
    print_peaks(sorted_peaks)
    sorted_peak_positions_all.append(get_sorted_peak_positions(sorted_peaks))
    # Get sorted bases for this sequence set
    sorted_bases_set = get_sorted_bases(sorted_peaks)
    sorted_bases_all.append(sorted_bases_set)
print(sorted_peak_positions_all[0])
print(sorted_bases_all[0])

def compare_peak_bases_with_tolerance(seq_set_4, seq_set, tolerance=2):
    matched_peaks = 0
    seq_set_4_peaks = list(zip(seq_set_4['peaks'], seq_set_4['bases']))  # base and peak matching
    for peak, base in zip(seq_set['peaks'], seq_set['bases']):
        for peak_4, base_4 in seq_set_4_peaks:
            if abs(peak - peak_4) <= tolerance and base == base_4:
                matched_peaks += 1
                break  # Once the match is successful, jump out of the inner loop

    return matched_peaks

sequence_set_16 = {'bases':sorted_bases_all[15], 'peaks': sorted_peak_positions_all[15]}

# Sequence sets to compare
sequence_sets = [
    {'bases': sorted_bases_all[0], 'peaks': sorted_peak_positions_all[0]},  # Sequence Set 1
    {'bases': sorted_bases_all[1], 'peaks': sorted_peak_positions_all[1]},   # Sequence Set 2
    {'bases': sorted_bases_all[2], 'peaks': sorted_peak_positions_all[2]},  # Sequence Set 3
    {'bases': sorted_bases_all[3], 'peaks': sorted_peak_positions_all[3]},  # Sequence Set 4
    {'bases': sorted_bases_all[4], 'peaks': sorted_peak_positions_all[4]},  # Sequence Set 5
    {'bases': sorted_bases_all[5], 'peaks': sorted_peak_positions_all[5]},  # Sequence Set 6
    {'bases': sorted_bases_all[6], 'peaks': sorted_peak_positions_all[6]},  # Sequence Set 7
    {'bases': sorted_bases_all[7], 'peaks': sorted_peak_positions_all[7]},  # Sequence Set 8
    {'bases': sorted_bases_all[8], 'peaks': sorted_peak_positions_all[8]},  # Sequence Set 9
    {'bases': sorted_bases_all[9], 'peaks': sorted_peak_positions_all[9]},  # Sequence Set 10
    {'bases': sorted_bases_all[10], 'peaks': sorted_peak_positions_all[10]},  # Sequence Set 11
    {'bases': sorted_bases_all[11], 'peaks': sorted_peak_positions_all[11]},  # Sequence Set 12
    {'bases': sorted_bases_all[12], 'peaks': sorted_peak_positions_all[12]},  # Sequence Set 13
    {'bases': sorted_bases_all[13], 'peaks': sorted_peak_positions_all[13]},  # Sequence Set 14
    {'bases': sorted_bases_all[14], 'peaks': sorted_peak_positions_all[14]},  # Sequence Set 15

]

# Calculate similarities to sequence sets
similarities = []
for i, seq_set in enumerate(sequence_sets[:15]):  # Only compare with Sequence Set 1 to 15
    matched_peaks = compare_peak_bases_with_tolerance(sequence_set_16, seq_set)
    similarities.append({'set': i+1, 'matched_peaks': matched_peaks})

# Sort by matched peaks
sorted_similarities = sorted(similarities, key=lambda x: x['matched_peaks'], reverse=True)

# Print results
print("Similarity Scores based on Peak Positions:")
for similarity in sorted_similarities:
    print(f"Sequence Set {similarity['set']} - Matched Peaks: {similarity['matched_peaks']}")

# Find the most similar sequence
most_similar_sequence = sorted_similarities[0]

# Print the most similar sequence set
print(f"\nMost similar sequence set: Sequence Set {most_similar_sequence['set']}")

# Assign the most similar sequence set as the reference sequences
if most_similar_sequence['set'] == 1:
    length_ref= length_APC_213
    ref_sequenceA = sequence1A
    ref_sequenceG = sequence1G
    ref_sequenceC = sequence1C
    ref_sequenceT = sequence1T

elif most_similar_sequence['set'] == 2:
    length_ref= length_KRAS_12
    ref_sequenceA = sequence2A
    ref_sequenceG = sequence2G
    ref_sequenceC = sequence2C
    ref_sequenceT = sequence2T

elif most_similar_sequence['set'] == 3:
    length_ref= length_PIK3CA_542
    ref_sequenceA = sequence3A
    ref_sequenceG = sequence3G
    ref_sequenceC = sequence3C
    ref_sequenceT = sequence3T

elif most_similar_sequence['set'] == 4:
    length_ref= length_PIK3CA_1047
    ref_sequenceA = sequence4A
    ref_sequenceG = sequence4G
    ref_sequenceC = sequence4C
    ref_sequenceT = sequence4T
elif most_similar_sequence['set'] == 5:
    length_ref= length_TP53_273
    ref_sequenceA = sequence5A
    ref_sequenceG = sequence5G
    ref_sequenceC = sequence5C
    ref_sequenceT = sequence5T
elif most_similar_sequence['set'] == 6:
    length_ref= length_BRAF_600
    ref_sequenceA = sequence6A
    ref_sequenceG = sequence6G
    ref_sequenceC = sequence6C
    ref_sequenceT = sequence6T
elif most_similar_sequence['set'] == 7:
    length_ref= length_ERBB2_842
    ref_sequenceA = sequence7A
    ref_sequenceG = sequence7G
    ref_sequenceC = sequence7C
    ref_sequenceT = sequence7T
elif most_similar_sequence['set'] == 8:
    length_ref= length_TP53_196
    ref_sequenceA = sequence8A
    ref_sequenceG = sequence8G
    ref_sequenceC = sequence8C
    ref_sequenceT = sequence8T
elif most_similar_sequence['set'] == 9:
    length_ref= length_ERBB2_777
    ref_sequenceA = sequence9A
    ref_sequenceG = sequence9G
    ref_sequenceC = sequence9C
    ref_sequenceT = sequence9T
elif most_similar_sequence['set'] == 10:
    length_ref= length_APC_1114
    ref_sequenceA = sequence10A
    ref_sequenceG = sequence10G
    ref_sequenceC = sequence10C
    ref_sequenceT = sequence10T
elif most_similar_sequence['set'] == 11:
    length_ref= length_KRAS_61
    ref_sequenceA = sequence11A
    ref_sequenceG = sequence11G
    ref_sequenceC = sequence11C
    ref_sequenceT = sequence11T
elif most_similar_sequence['set'] == 12:
    length_ref= length_APC_1450
    ref_sequenceA = sequence12A
    ref_sequenceG = sequence12G
    ref_sequenceC = sequence12C
    ref_sequenceT = sequence12T
elif most_similar_sequence['set'] == 13:
    length_ref= length_KRAS_146
    ref_sequenceA = sequence13A
    ref_sequenceG = sequence13G
    ref_sequenceC = sequence13C
    ref_sequenceT = sequence13T
elif most_similar_sequence['set'] == 14:
    length_ref= length_ERBB2_310
    ref_sequenceA = sequence14A
    ref_sequenceG = sequence14G
    ref_sequenceC = sequence14C
    ref_sequenceT = sequence14T
elif most_similar_sequence['set'] == 15:
    length_ref= length_APC_876
    ref_sequenceA = sequence15A
    ref_sequenceG = sequence15G
    ref_sequenceC = sequence15C
    ref_sequenceT = sequence15T
else:
    length_ref = None  # Handle the case where no ref_sequence is chosen

print(ref_sequenceA)
print(length_ref)

if length_ref == None:
    print("This is a chaos spectrum, no mutation in this sample")

# length_ref= length_PIK3CA_542
# ref_sequenceA = sequence12A
# ref_sequenceG = sequence12G
# ref_sequenceC = sequence12C
# ref_sequenceT = sequence12T
# print(length_ref)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(ref_sequenceA, color='green', label='ref data A lines', marker='', linestyle='-')
plt.plot(ref_sequenceG, color='black', label='ref data G lines', marker='', linestyle='-')
plt.plot(ref_sequenceC, color='blue', label='ref data C lines', marker='', linestyle='-')
plt.plot(ref_sequenceT, color='red', label='ref data T lines', marker='', linestyle='-')
plt.ylim(0, 100)
plt.ylabel('Value')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(sequence2A, color='green', label='ref data A lines', marker='', linestyle='-')
plt.plot(sequence2G, color='black', label='ref data G lines', marker='', linestyle='-')
plt.plot(sequence2C, color='blue', label='ref data C lines', marker='', linestyle='-')
plt.plot(sequence2T, color='red', label='ref data T lines', marker='', linestyle='-')
plt.ylim(0, 100)
plt.ylabel('Value')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(sequence30A, color='green', label='UP data A lines', marker='', linestyle='-')
plt.plot(sequence30G, color='black', label='UP data G lines', marker='', linestyle='-')
plt.plot(sequence30C, color='blue', label='UP data C lines', marker='', linestyle='-')
plt.plot(sequence30T, color='red', label='UP data T lines', marker='', linestyle='-')
plt.ylim(0, 100)
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

# Set parameters
segment_start = 0
segment_end = 700
height_threshold = 10

sequences = [
    [(ref_sequenceA, 'ref_sequenceA'), (ref_sequenceG, 'ref_sequenceG'), (ref_sequenceC, 'ref_sequenceC'), (ref_sequenceT, 'ref_sequenceT')],
    [(sequence1A, 'sequence1A'), (sequence1G, 'sequence1G'), (sequence1C, 'sequence1C'), (sequence1T, 'sequence1T')],
    [(sequence2A, 'sequence2A'), (sequence2G, 'sequence2G'), (sequence2C, 'sequence2C'), (sequence2T, 'sequence2T')],
    [(sequence3A, 'sequence3A'), (sequence3G, 'sequence3G'), (sequence3C, 'sequence3C'), (sequence3T, 'sequence3T')],
    [(sequence4A, 'sequence4A'), (sequence4G, 'sequence4G'), (sequence4C, 'sequence4C'), (sequence4T, 'sequence4T')],
    [(sequence5A, 'sequence5A'), (sequence5G, 'sequence5G'), (sequence5C, 'sequence5C'), (sequence5T, 'sequence5T')],
    [(sequence6A, 'sequence6A'), (sequence6G, 'sequence6G'), (sequence6C, 'sequence6C'), (sequence6T, 'sequence6T')],
    [(sequence7A, 'sequence7A'), (sequence7G, 'sequence7G'), (sequence7C, 'sequence7C'), (sequence7T, 'sequence7T')],
    [(sequence8A, 'sequence8A'), (sequence8G, 'sequence8G'), (sequence8C, 'sequence8C'), (sequence8T, 'sequence8T')],
    [(sequence9A, 'sequence9A'), (sequence9G, 'sequence9G'), (sequence9C, 'sequence9C'), (sequence9T, 'sequence9T')],
    [(sequence10A, 'sequence10A'), (sequence10G, 'sequence10G'), (sequence10C, 'sequence10C'), (sequence10T, 'sequence10T')],
    [(sequence11A, 'sequence11A'), (sequence11G, 'sequence11G'), (sequence11C, 'sequence11C'), (sequence11T, 'sequence11T')],
    [(sequence12A, 'sequence12A'), (sequence12G, 'sequence12G'), (sequence12C, 'sequence12C'), (sequence12T, 'sequence12T')],
    [(sequence13A, 'sequence13A'), (sequence13G, 'sequence13G'), (sequence13C, 'sequence13C'), (sequence13T, 'sequence13T')],
    [(sequence14A, 'sequence14A'), (sequence14G, 'sequence14G'), (sequence14C, 'sequence14C'), (sequence14T, 'sequence14T')],
    [(sequence15A, 'sequence15A'), (sequence15G, 'sequence15G'), (sequence15C, 'sequence15C'), (sequence15T, 'sequence15T')],
    [(sequence30A, 'sequence30A'), (sequence30G, 'sequence30G'), (sequence30C, 'sequence30C'), (sequence30T, 'sequence30T')]
]

# Stores the sequencing peak positions for all sequences
sorted_peak_positions_all = []
for i, seq_set in enumerate(sequences):
    sorted_peaks = find_and_sort_peaks(seq_set, segment_start, segment_end, height_threshold)
    print(f"\nPeaks for Sequence Set {i+1}:")
    print_peaks(sorted_peaks)
    sorted_peak_positions_all.append(get_sorted_peak_positions(sorted_peaks))

length_APC_213 = 151
length_KRAS_12 = 148
length_PIK3CA_542 = 144
length_PIK3CA_1047 = 142
length_TP53_273 = 140
length_BRAF_600 = 139
length_ERBB2_842 = 135
length_TP53_196 = 133
length_ERBB2_777 = 132
length_APC_1114 = 122
length_KRAS_61 = 120
length_APC_1450 = 118
length_KRAS_146 = 114
length_ERBB2_310 = 113
length_APC_876 = 106

# Extract different zero padding values
zerospadding_APC_213 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_APC_213,0)+1] - sorted_peak_positions_all[1][1],0)
zerospadding_KRAS_12 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_KRAS_12,0)+1] - sorted_peak_positions_all[2][1],0)
zerospadding_PIK3CA_542 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_PIK3CA_542,0)+1] - sorted_peak_positions_all[3][1],0)
zerospadding_PIK3CA_1047 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_PIK3CA_1047,0)+1] - sorted_peak_positions_all[4][1],0)
zerospadding_TP53_273 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_TP53_273,0)+1] - sorted_peak_positions_all[5][1],0)
zerospadding_BRAF_600 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_BRAF_600,0)+1] - sorted_peak_positions_all[6][1],0)
zerospadding_ERBB2_842 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_ERBB2_842,0)+1] - sorted_peak_positions_all[7][1],0)
zerospadding_TP53_196 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_TP53_196,0)+1] - sorted_peak_positions_all[8][1],0)
zerospadding_ERBB2_777 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_ERBB2_777,0)+1] - sorted_peak_positions_all[9][1],0)
zerospadding_APC_1114 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_APC_1114,0)+1] - sorted_peak_positions_all[10][1],0)
zerospadding_KRAS_61 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_KRAS_61,0)+1] - sorted_peak_positions_all[11][1],0)
zerospadding_APC_1450 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_APC_1450,0)+1] - sorted_peak_positions_all[12][1],0)
zerospadding_KRAS_146 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_KRAS_146,0)+1] - sorted_peak_positions_all[13][1],0)
zerospadding_ERBB2_310 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_ERBB2_310,0)+1] - sorted_peak_positions_all[14][1],0)
zerospadding_APC_876 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_APC_876,0)+1] - sorted_peak_positions_all[15][1],0)

print(zerospadding_APC_213)
print(zerospadding_KRAS_12)
print(zerospadding_PIK3CA_542)
print(zerospadding_PIK3CA_1047)
print(zerospadding_TP53_273)
print(zerospadding_BRAF_600)
print(zerospadding_ERBB2_842)
print(zerospadding_TP53_196)
print(zerospadding_ERBB2_777)
print(zerospadding_APC_1114)
print(zerospadding_KRAS_61)
print(zerospadding_APC_1450)
print(zerospadding_KRAS_146)
print(zerospadding_ERBB2_310)
print(zerospadding_APC_876)

sequence1A = np.concatenate((np.zeros(np.abs(zerospadding_APC_213)), sequence1A))
sequence1G = np.concatenate((np.zeros(np.abs(zerospadding_APC_213)), sequence1G))
sequence1C = np.concatenate((np.zeros(zerospadding_APC_213), sequence1C))
sequence1T = np.concatenate((np.zeros(zerospadding_APC_213), sequence1T))
sequence2A = np.concatenate((np.zeros(zerospadding_KRAS_12), sequence2A))
sequence2G = np.concatenate((np.zeros(zerospadding_KRAS_12), sequence2G))
sequence2C = np.concatenate((np.zeros(zerospadding_KRAS_12), sequence2C))
sequence2T = np.concatenate((np.zeros(zerospadding_KRAS_12), sequence2T))
sequence3A = np.concatenate((np.zeros(zerospadding_PIK3CA_542), sequence3A))
sequence3G = np.concatenate((np.zeros(zerospadding_PIK3CA_542), sequence3G))
sequence3C = np.concatenate((np.zeros(zerospadding_PIK3CA_542), sequence3C))
sequence3T = np.concatenate((np.zeros(zerospadding_PIK3CA_542), sequence3T))
sequence4A = np.concatenate((np.zeros(zerospadding_PIK3CA_1047), sequence4A))
sequence4G = np.concatenate((np.zeros(zerospadding_PIK3CA_1047), sequence4G))
sequence4C = np.concatenate((np.zeros(zerospadding_PIK3CA_1047), sequence4C))
sequence4T = np.concatenate((np.zeros(zerospadding_PIK3CA_1047), sequence4T))
sequence5A = np.concatenate((np.zeros(np.abs(zerospadding_TP53_273)), sequence5A))
sequence5G = np.concatenate((np.zeros(np.abs(zerospadding_TP53_273)), sequence5G))
sequence5C = np.concatenate((np.zeros(np.abs(zerospadding_TP53_273)), sequence5C))
sequence5T = np.concatenate((np.zeros(np.abs(zerospadding_TP53_273)), sequence5T))
sequence6A = np.concatenate((np.zeros(zerospadding_BRAF_600), sequence6A))
sequence6G = np.concatenate((np.zeros(zerospadding_BRAF_600), sequence6G))
sequence6C = np.concatenate((np.zeros(zerospadding_BRAF_600), sequence6C))
sequence6T = np.concatenate((np.zeros(zerospadding_BRAF_600), sequence6T))
sequence7A = np.concatenate((np.zeros(zerospadding_ERBB2_842), sequence7A))
sequence7G = np.concatenate((np.zeros(zerospadding_ERBB2_842), sequence7G))
sequence7C = np.concatenate((np.zeros(zerospadding_ERBB2_842), sequence7C))
sequence7T = np.concatenate((np.zeros(zerospadding_ERBB2_842), sequence7T))
sequence8A = np.concatenate((np.zeros(zerospadding_TP53_196), sequence8A))
sequence8G = np.concatenate((np.zeros(zerospadding_TP53_196), sequence8G))
sequence8C = np.concatenate((np.zeros(zerospadding_TP53_196), sequence8C))
sequence8T = np.concatenate((np.zeros(zerospadding_TP53_196), sequence8T))
sequence9A = np.concatenate((np.zeros(zerospadding_ERBB2_777), sequence9A))
sequence9G = np.concatenate((np.zeros(zerospadding_ERBB2_777), sequence9G))
sequence9C = np.concatenate((np.zeros(zerospadding_ERBB2_777), sequence9C))
sequence9T = np.concatenate((np.zeros(zerospadding_ERBB2_777), sequence9T))
sequence10A = np.concatenate((np.zeros(zerospadding_APC_1114), sequence10A))
sequence10G = np.concatenate((np.zeros(zerospadding_APC_1114), sequence10G))
sequence10C = np.concatenate((np.zeros(zerospadding_APC_1114), sequence10C))
sequence10T = np.concatenate((np.zeros(zerospadding_APC_1114), sequence10T))
sequence11A = np.concatenate((np.zeros(zerospadding_KRAS_61), sequence11A))
sequence11G = np.concatenate((np.zeros(zerospadding_KRAS_61), sequence11G))
sequence11C = np.concatenate((np.zeros(zerospadding_KRAS_61), sequence11C))
sequence11T = np.concatenate((np.zeros(zerospadding_KRAS_61), sequence11T))
sequence12A = np.concatenate((np.zeros(zerospadding_APC_1450), sequence12A))
sequence12G = np.concatenate((np.zeros(zerospadding_APC_1450), sequence12G))
sequence12C = np.concatenate((np.zeros(zerospadding_APC_1450), sequence12C))
sequence12T = np.concatenate((np.zeros(zerospadding_APC_1450), sequence12T))
sequence13A = np.concatenate((np.zeros(zerospadding_KRAS_146), sequence13A))
sequence13G = np.concatenate((np.zeros(zerospadding_KRAS_146), sequence13G))
sequence13C = np.concatenate((np.zeros(zerospadding_KRAS_146), sequence13C))
sequence13T = np.concatenate((np.zeros(zerospadding_KRAS_146), sequence13T))
sequence14A = np.concatenate((np.zeros(zerospadding_ERBB2_310), sequence14A))
sequence14G = np.concatenate((np.zeros(zerospadding_ERBB2_310), sequence14G))
sequence14C = np.concatenate((np.zeros(zerospadding_ERBB2_310), sequence14C))
sequence14T = np.concatenate((np.zeros(zerospadding_ERBB2_310), sequence14T))
sequence15A = np.concatenate((np.zeros(zerospadding_APC_876), sequence15A))
sequence15G = np.concatenate((np.zeros(zerospadding_APC_876), sequence15G))
sequence15C = np.concatenate((np.zeros(zerospadding_APC_876), sequence15C))
sequence15T = np.concatenate((np.zeros(zerospadding_APC_876), sequence15T))

max_sequence1 = max(max(sequence1A, default=0),max(sequence1G, default=0), max(sequence1C, default=0),max(sequence1T, default=0))
max_sequence2 = max(max(sequence2A, default=0),max(sequence2G, default=0), max(sequence2C, default=0),max(sequence2T, default=0))
max_sequence3 = max(max(sequence3A, default=0),max(sequence3G, default=0), max(sequence3C, default=0),max(sequence3T, default=0))
max_sequence4 = max(max(sequence4A, default=0),max(sequence4G, default=0), max(sequence4C, default=0),max(sequence4T, default=0))
max_sequence5 = max(max(sequence5A, default=0),max(sequence5G, default=0), max(sequence5C, default=0),max(sequence5T, default=0))
max_sequence6 = max(max(sequence6A, default=0),max(sequence6G, default=0), max(sequence6C, default=0),max(sequence6T, default=0))
max_sequence7 = max(max(sequence7A, default=0),max(sequence7G, default=0), max(sequence7C, default=0),max(sequence7T, default=0))
max_sequence8 = max(max(sequence8A, default=0),max(sequence8G, default=0), max(sequence8C, default=0),max(sequence8T, default=0))
max_sequence9 = max(max(sequence9A, default=0),max(sequence9G, default=0), max(sequence9C, default=0),max(sequence9T, default=0))
max_sequence10 = max(max(sequence10A, default=0),max(sequence10G, default=0), max(sequence10C, default=0),max(sequence10T, default=0))
max_sequence11 = max(max(sequence11A, default=0),max(sequence11G, default=0), max(sequence11C, default=0),max(sequence11T, default=0))
max_sequence12= max(max(sequence12A, default=0),max(sequence12G, default=0), max(sequence12C, default=0),max(sequence12T, default=0))
max_sequence13 = max(max(sequence13A, default=0),max(sequence13G, default=0), max(sequence13C, default=0),max(sequence13T, default=0))
max_sequence14= max(max(sequence14A, default=0),max(sequence14G, default=0), max(sequence14C, default=0),max(sequence14T, default=0))
max_sequence15= max(max(sequence15A, default=0),max(sequence15G, default=0), max(sequence15C, default=0),max(sequence15T, default=0))
max_sequence30= max(max(sequence30A, default=0),max(sequence30G, default=0), max(sequence30C, default=0),max(sequence30T, default=0))
max_sequence_ref = max(max(ref_sequenceA, default=0),max(ref_sequenceG, default=0), max(ref_sequenceC, default=0),max(ref_sequenceT, default=0))

sequence1A = sequence1A/max_sequence1*100
sequence2A = sequence2A/max_sequence2*100
sequence3A = sequence3A/max_sequence3*100
sequence4A = sequence4A/max_sequence4*100
sequence5A = sequence5A/max_sequence5*100
sequence6A = sequence6A/max_sequence6*100
sequence1G = sequence1G/max_sequence1*100
sequence2G = sequence2G/max_sequence2*100
sequence3G = sequence3G/max_sequence3*100
sequence4G = sequence4G/max_sequence4*100
sequence5G = sequence5G/max_sequence5*100
sequence6G = sequence6G/max_sequence6*100
sequence1C = sequence1C/max_sequence1*100
sequence2C = sequence2C/max_sequence2*100
sequence3C = sequence3C/max_sequence3*100
sequence4C = sequence4C/max_sequence4*100
sequence5C = sequence5C/max_sequence5*100
sequence6C = sequence6C/max_sequence6*100
sequence1T = sequence1T/max_sequence1*100
sequence2T = sequence2T/max_sequence2*100
sequence3T = sequence3T/max_sequence3*100
sequence4T = sequence4T/max_sequence4*100
sequence5T = sequence5T/max_sequence5*100
sequence6T = sequence6T/max_sequence6*100
sequence7A = sequence7A/max_sequence7*100
sequence7G = sequence7G/max_sequence7*100
sequence7C = sequence7C/max_sequence7*100
sequence7T = sequence7T/max_sequence7*100
sequence8A = sequence8A/max_sequence8*100
sequence8G = sequence8G/max_sequence8*100
sequence8C = sequence8C/max_sequence8*100
sequence8T = sequence8T/max_sequence8*100
sequence9A = sequence9A/max_sequence9*100
sequence9G = sequence9G/max_sequence9*100
sequence9C = sequence9C/max_sequence9*100
sequence9T = sequence9T/max_sequence9*100
sequence10A = sequence10A/max_sequence10*100
sequence10G = sequence10G/max_sequence10*100
sequence10C = sequence10C/max_sequence10*100
sequence10T = sequence10T/max_sequence10*100
sequence11A = sequence11A/max_sequence11*100
sequence11G = sequence11G/max_sequence11*100
sequence11C = sequence11C/max_sequence11*100
sequence11T = sequence11T/max_sequence11*100
sequence12A = sequence12A/max_sequence12*100
sequence12G = sequence12G/max_sequence12*100
sequence12C = sequence12C/max_sequence12*100
sequence12T = sequence12T/max_sequence12*100
sequence13A = sequence13A/max_sequence13*100
sequence13G = sequence13G/max_sequence13*100
sequence13C = sequence13C/max_sequence13*100
sequence13T = sequence13T/max_sequence13*100
sequence14A = sequence14A/max_sequence14*100
sequence14G = sequence14G/max_sequence14*100
sequence14C = sequence14C/max_sequence14*100
sequence14T = sequence14T/max_sequence14*100
sequence15A = sequence15A/max_sequence15*100
sequence15G = sequence15G/max_sequence15*100
sequence15C = sequence15C/max_sequence15*100
sequence15T = sequence15T/max_sequence15*100
sequence30A = sequence30A/max_sequence30*100
sequence30G = sequence30G/max_sequence30*100
sequence30C = sequence30C/max_sequence30*100
sequence30T = sequence30T/max_sequence30*100

ref_sequenceA = ref_sequenceA/max_sequence_ref*100
ref_sequenceG = ref_sequenceG/max_sequence_ref*100
ref_sequenceC = ref_sequenceC/max_sequence_ref*100
ref_sequenceT = ref_sequenceT/max_sequence_ref*100

# Find the length of the shortest sequence
min_length = min(len(sequence1A), len(sequence2A),len(sequence3A),len(sequence4A),len(sequence5A),len(sequence6A),len(sequence7A),len(sequence8A),len(sequence9A),len(sequence10A),len(sequence11A),len(sequence12A),len(sequence13A),len(sequence14A),len(sequence15A))

# Shorten the longer sequence to match the length of the shortest sequence
sequence1A = sequence1A[:min_length]
sequence2A = sequence2A[:min_length]
sequence1G = sequence1G[:min_length]   
sequence2G = sequence2G[:min_length] 
sequence1C = sequence1C[:min_length] 
sequence2C = sequence2C[:min_length] 
sequence1T = sequence1T[:min_length] 
sequence2T = sequence2T[:min_length] 
sequence3A = sequence3A[:min_length] 
sequence3G = sequence3G[:min_length] 
sequence3C = sequence3C[:min_length] 
sequence3T = sequence3T[:min_length] 
sequence4A = sequence4A[:min_length] 
sequence4G = sequence4G[:min_length] 
sequence4C = sequence4C[:min_length] 
sequence4T = sequence4T[:min_length] 
sequence5A = sequence5A[:min_length] 
sequence5G = sequence5G[:min_length] 
sequence5C = sequence5C[:min_length] 
sequence5T = sequence5T[:min_length]
sequence6A = sequence6A[:min_length] 
sequence6G = sequence6G[:min_length] 
sequence6C = sequence6C[:min_length] 
sequence6T = sequence6T[:min_length]
sequence7A= sequence7A[:min_length]
sequence7G = sequence7G[:min_length] 
sequence7C = sequence7C[:min_length]
sequence7T = sequence7T[:min_length]
sequence8A = sequence8A[:min_length] 
sequence8G = sequence8G[:min_length] 
sequence8C = sequence8C[:min_length]
sequence8T = sequence8T[:min_length]
sequence9A = sequence9A[:min_length] 
sequence9G = sequence9G[:min_length] 
sequence9C = sequence9C[:min_length]
sequence9T = sequence9T[:min_length]
sequence10A = sequence10A[:min_length] 
sequence10G = sequence10G[:min_length] 
sequence10C = sequence10C[:min_length]
sequence10T = sequence10T[:min_length]
sequence11A = sequence11A[:min_length] 
sequence11G = sequence11G[:min_length] 
sequence11C = sequence11C[:min_length] 
sequence11T = sequence11T[:min_length]
sequence12A = sequence12A[:min_length] 
sequence12G = sequence12G[:min_length] 
sequence12C = sequence12C[:min_length] 
sequence12T = sequence12T[:min_length]
sequence13A = sequence13A[:min_length] 
sequence13G = sequence13G[:min_length] 
sequence13C = sequence13C[:min_length] 
sequence13T = sequence13T[:min_length]
sequence14A = sequence14A[:min_length] 
sequence14G = sequence14G[:min_length] 
sequence14C = sequence14C[:min_length] 
sequence14T = sequence14T[:min_length]
sequence15A = sequence15A[:min_length] 
sequence15G = sequence15G[:min_length] 
sequence15C = sequence15C[:min_length] 
sequence15T = sequence15T[:min_length]
sequence30A = sequence30A[:min_length] 
sequence30G = sequence30G[:min_length] 
sequence30C = sequence30C[:min_length] 
sequence30T = sequence30T[:min_length]
sequence0A = sequence0A[:min_length] 
sequence0G = sequence0G[:min_length] 
sequence0C = sequence0C[:min_length]
sequence0T = sequence0T[:min_length]
ref_sequenceA =ref_sequenceA[:min_length]
ref_sequenceG =ref_sequenceG[:min_length]
ref_sequenceC =ref_sequenceC[:min_length]
ref_sequenceT =ref_sequenceT[:min_length]

plt.figure(figsize=(12, 8))
# Plot the reference sequence spectrum
plt.subplot(2, 1, 1)
plt.plot(ref_sequenceA, color='green', label='ref data A lines', marker='', linestyle='-')
plt.plot(ref_sequenceG, color='black', label='ref data G lines', marker='', linestyle='-')
plt.plot(ref_sequenceC, color='blue', label='ref data C lines', marker='', linestyle='-')
plt.plot(ref_sequenceT, color='red', label='ref data T lines', marker='', linestyle='-')
plt.ylim(0, 100)
plt.ylabel('Value')
plt.legend()

# Plot the multiplex sequence spectrum
plt.subplot(2, 1, 2)
plt.plot(sequence30A, color='green', label='UP data A lines', marker='', linestyle='-')
plt.plot(sequence30G, color='black', label='UP data G lines', marker='', linestyle='-')
plt.plot(sequence30C, color='blue', label='UP data C lines', marker='', linestyle='-')
plt.plot(sequence30T, color='red', label='UP data T lines', marker='', linestyle='-')
plt.ylim(0, 100)
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

segment_start = 20
segment_end = 400
height_threshold = 5
peaks_1T, _ = find_peaks(ref_sequenceT[segment_start:segment_end],height = height_threshold)
peaks_30T, _ = find_peaks(sequence30T[segment_start:segment_end],height = height_threshold)
print(peaks_1T[0])
print(peaks_30T[0])

def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

# Calculate the shift required to align sequence1T to sequence3T
shift = peaks_1T[0] - peaks_30T[0]

# Apply this shift to all sequences in channels A, G, C and T
sequence30A = align_sequences_based_on_shift(sequence30A, shift)
sequence30G = align_sequences_based_on_shift(sequence30G, shift)
sequence30C = align_sequences_based_on_shift(sequence30C, shift)
sequence30T = align_sequences_based_on_shift(sequence30T, shift) 

sequence30A = sequence30A[:min_length] 
sequence30G = sequence30G[:min_length] 
sequence30C = sequence30C[:min_length]
sequence30T = sequence30T[:min_length]
# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(ref_sequenceA, color='green', label='ref data A lines', marker='', linestyle='-')
# plt.plot(ref_sequenceG, color='black', label='ref data G lines', marker='', linestyle='-')
# plt.plot(ref_sequenceC, color='blue', label='ref data C lines', marker='', linestyle='-')
# plt.plot(ref_sequenceT, color='red', label='ref data T lines', marker='', linestyle='-')
# # plt.title('mixed sequence')
# plt.ylim(0, 100)
# plt.ylabel('Value')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(sequence30A, color='green', label='UP data A lines', marker='', linestyle='-')
# plt.plot(sequence30G, color='black', label='UP data G lines', marker='', linestyle='-')
# plt.plot(sequence30C, color='blue', label='UP data C lines', marker='', linestyle='-')
# plt.plot(sequence30T, color='red', label='UP data T lines', marker='', linestyle='-')
# # plt.title('UP sequence')
# plt.legend()
# plt.tight_layout()
# plt.show()

segment_start = 20
segment_end = 400
height_threshold = 10
# Find the peak position of each sequence
peaks_1A, _ = find_peaks(ref_sequenceA[segment_start:segment_end],height = height_threshold)
peaks_1G, _ = find_peaks(ref_sequenceG[segment_start:segment_end],height = height_threshold)
peaks_1C, _ = find_peaks(ref_sequenceC[segment_start:segment_end],height = height_threshold)
peaks_1T, _ = find_peaks(ref_sequenceT[segment_start:segment_end],height = height_threshold)

# Record all peaks and their positions and corresponding sequence names
all_peaks_1 = [(x, 'ref_sequenceA') for x in peaks_1A] + \
            [(x, 'ref_sequenceG') for x in peaks_1G] + \
            [(x, 'ref_sequenceC') for x in peaks_1C] + \
            [(x, 'ref_sequenceT') for x in peaks_1T]

# Sort by x axis position
sorted_peaks = sorted(all_peaks_1)
for i, (peak, seq_name) in enumerate(sorted_peaks):
    print(f"{seq_name} has the {i+1}st peak at index {peak}")

# Extract peak position
peak_positions_1 = [peak for peak, _ in all_peaks_1]

# Sort peaks according to their x position
sorted_peak_positions_1 = sorted(peak_positions_1)

# Convert to a list of regular integers
sorted_peak_positions_1 = [int(x) for x in sorted_peak_positions_1]
print(sorted_peak_positions_1[1])

# Find the peak position of each sequence
peaks_2A, _ = find_peaks(sequence30A[segment_start:segment_end],height = height_threshold)
peaks_2G, _ = find_peaks(sequence30G[segment_start:segment_end],height = height_threshold)
peaks_2C, _ = find_peaks(sequence30C[segment_start:segment_end],height = height_threshold)
peaks_2T, _ = find_peaks(sequence30T[segment_start:segment_end],height = height_threshold)

# Record all peaks and their positions and corresponding sequence names
all_peaks_2 = [(x, 'sequence30A') for x in peaks_2A] + \
            [(x, 'sequence30G') for x in peaks_2G] + \
            [(x, 'sequence30C') for x in peaks_2C] + \
            [(x, 'sequence30T') for x in peaks_2T]

# Sort by x axis position
sorted_peaks = sorted(all_peaks_2)
for i, (peak, seq_name) in enumerate(sorted_peaks):
    print(f"{seq_name} has the {i+1}st peak at index {peak}")

# Extract peak position
peak_positions_2 = [peak for peak, _ in all_peaks_2]
print(peak_positions_2)
# Sort peaks according to their x position
sorted_peak_positions_2 = sorted(peak_positions_2)
# Convert to a list of regular integers
sorted_peak_positions_2 = [int(x) for x in sorted_peak_positions_2]
print(sorted_peak_positions_2)

# Adjust peak indices according to segment start
# Append segment_start to the list
sorted_peak_positions_1.append(segment_start)
sorted_peak_positions_2.append(segment_start)

#Filter out the peaks whose adjacent position difference is less than min_distance from the sorted peak list
def filter_min_distance(peaks, min_distance=8):
    if not peaks:
        return []   
    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_distance:
            filtered.append(p)
    return filtered

filtered_peak_positions_1 = filter_min_distance(sorted_peak_positions_1, min_distance=8)
filtered_peak_positions_2 = filter_min_distance(sorted_peak_positions_2, min_distance=8)

print("Filtered Peaks 1:", filtered_peak_positions_1)
print("Filtered Peaks 2:", filtered_peak_positions_2)

# Number of peaks used to calculate stretch factor
num_peaks_to_use = 3
if len(filtered_peak_positions_1) >= num_peaks_to_use and len(filtered_peak_positions_2) >= num_peaks_to_use:
    # Calculate stretch factors
    stretch_factors = np.diff(filtered_peak_positions_1[:num_peaks_to_use]) / np.diff(filtered_peak_positions_2[:num_peaks_to_use])
    average_stretch_factor = np.mean(stretch_factors)
if  0.8 < average_stretch_factor <=0.98 or 1.2 > average_stretch_factor >=1.02:
    average_stretch_factor == average_stretch_factor
elif average_stretch_factor <=0.8 or average_stretch_factor >=1.2:
    average_stretch_factor = 1
else:
    average_stretch_factor = 1

# average_stretch_factor = 1
print("Average_stretch_factor :", average_stretch_factor)
# Apply stretch factor to the x coordinates of signal2
x = np.arange(len(sequence30T))  # Original x coordinates
# x_stretchedT = x * average_stretch_factor
x_stretchedT = x * average_stretch_factor
sequence30A = np.interp(x, x_stretchedT, sequence30A) # Interpolate signal2 to new x coordinates
sequence30G = np.interp(x, x_stretchedT, sequence30G) 
sequence30C = np.interp(x, x_stretchedT, sequence30C)  
sequence30T = np.interp(x, x_stretchedT, sequence30T) 

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(ref_sequenceA, color='green', label='ref data A lines', marker='', linestyle='-')
# plt.plot(ref_sequenceG, color='black', label='ref data G lines', marker='', linestyle='-')
# plt.plot(ref_sequenceC, color='blue', label='ref data C lines', marker='', linestyle='-')
# plt.plot(ref_sequenceT, color='red', label='ref data T lines', marker='', linestyle='-')
# # plt.title('mixed sequence')
# plt.ylim(0, 100)
# plt.ylabel('Value')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(sequence30A, color='green', label='UP data A lines', marker='', linestyle='-')
# plt.plot(sequence30G, color='black', label='UP data G lines', marker='', linestyle='-')
# plt.plot(sequence30C, color='blue', label='UP data C lines', marker='', linestyle='-')
# plt.plot(sequence30T, color='red', label='UP data T lines', marker='', linestyle='-')
# # plt.title('UP sequence')
# plt.legend()
# plt.tight_layout()
# plt.show()

length_APC_213 = 151
length_KRAS_12 = 148
length_PIK3CA_542 = 144
length_PIK3CA_1047 = 142
length_TP53_273 = 140
length_BRAF_600 = 139
length_ERBB2_842 = 135
length_TP53_196 = 133
length_ERBB2_777 = 132
length_APC_1114 = 122
length_KRAS_61 = 120
length_APC_1450 = 118
length_KRAS_146 = 114
length_ERBB2_310 = 113
length_APC_876 = 106

print(sorted_peak_positions_all[0][length_ref-length_APC_213]-5)
print(sorted_peak_positions_all[0][length_ref-length_KRAS_12]-5)
print(sorted_peak_positions_all[0][length_ref-length_PIK3CA_542]-5)
print(sorted_peak_positions_all[0][length_ref-length_PIK3CA_1047]-5)
print(sorted_peak_positions_all[0][length_ref-length_TP53_273]-5)
print(sorted_peak_positions_all[0][length_ref-length_BRAF_600]-5)
print(sorted_peak_positions_all[0][length_ref-length_ERBB2_842]-5)
print(sorted_peak_positions_all[0][length_ref-length_TP53_196]-5)
print(sorted_peak_positions_all[0][length_ref-length_ERBB2_777]-5)
print(sorted_peak_positions_all[0][length_ref-length_APC_1114]-5)
print(sorted_peak_positions_all[0][length_ref-length_KRAS_61]-5)
print(sorted_peak_positions_all[0][length_ref-length_APC_1450]-5)
print(sorted_peak_positions_all[0][length_ref-length_KRAS_146]-5)
print(sorted_peak_positions_all[0][length_ref-length_ERBB2_310]-5)
print(sorted_peak_positions_all[0][length_ref-length_APC_876]-5)

# Define DTW distance functions for each combination of channels and plex name
def dtw_distance_ref(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=15, end_index_x=165, start_index_y=15, end_index_y=165)

def dtw_distance_1(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_APC_213]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_APC_213]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_APC_213]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_APC_213]+70)

def dtw_distance_2(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_KRAS_12]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_KRAS_12]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_KRAS_12]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_KRAS_12]+70)

def dtw_distance_3(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_PIK3CA_542]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_PIK3CA_542]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_PIK3CA_542]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_PIK3CA_542]+70)

def dtw_distance_4(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_PIK3CA_1047]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_PIK3CA_1047]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_PIK3CA_1047]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_PIK3CA_1047]+70)

def dtw_distance_5(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_TP53_273]-5, end_index_x=sorted_peak_positions_all[0][length_ref-length_TP53_273]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_TP53_273]-5, end_index_y=sorted_peak_positions_all[0][length_ref-length_TP53_273]+70)

def dtw_distance_6(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_BRAF_600]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_BRAF_600]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_BRAF_600]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_BRAF_600]+70)

def dtw_distance_7(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_ERBB2_842]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_ERBB2_842]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_ERBB2_842]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_ERBB2_842]+70)

def dtw_distance_8(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_TP53_196]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_TP53_196]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_TP53_196]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_TP53_196]+70)

def dtw_distance_9(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_ERBB2_777]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_ERBB2_777]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_ERBB2_777]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_ERBB2_777]+70)

def dtw_distance_10(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_APC_1114]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_APC_1114]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_APC_1114]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_APC_1114]+70)

def dtw_distance_11(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_KRAS_61]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_KRAS_61]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_KRAS_61]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_KRAS_61]+70)

def dtw_distance_12(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_APC_1450]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_APC_1450]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_APC_1450]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_APC_1450]+70)

def dtw_distance_13(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_KRAS_146]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_KRAS_146]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_KRAS_146]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_KRAS_146]+70)

def dtw_distance_14(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_ERBB2_310]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_ERBB2_310]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_ERBB2_310]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_ERBB2_310]+70)

def dtw_distance_15(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_APC_876]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_APC_876]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_APC_876]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_APC_876]+70)

# plt.figure(figsize=(12, 8))
# plt.subplot(3, 1, 1)
# plt.plot(sequence3A, color='green', label='mixed data A lines', marker='', linestyle='-')
# plt.plot(sequence3G, color='black', label='mixed data G lines', marker='', linestyle='-')
# plt.plot(sequence3C, color='blue', label='mixed data C lines', marker='', linestyle='-')
# plt.plot(sequence3T, color='red', label='mixed data T lines', marker='', linestyle='-')
# # plt.title('mixed sequence')
# plt.ylim(0, 100)
# plt.ylabel('Value')
# plt.legend()

# plt.subplot(3, 1, 2)
# plt.plot(sequence5A, color='green', label='mixed data A lines', marker='', linestyle='-')
# plt.plot(sequence5G, color='black', label='mixed data G lines', marker='', linestyle='-')
# plt.plot(sequence5C, color='blue', label='mixed data C lines', marker='', linestyle='-')
# plt.plot(sequence5T, color='red', label='mixed data T lines', marker='', linestyle='-')
# # plt.title('mixed sequence')
# plt.ylim(0, 100)
# plt.ylabel('Value')
# plt.legend()

# plt.subplot(3, 1, 3)
# plt.plot(sequence30A, color='green', label='UP data A lines', marker='', linestyle='-')
# plt.plot(sequence30G, color='black', label='UP data G lines', marker='', linestyle='-')
# plt.plot(sequence30C, color='blue', label='UP data C lines', marker='', linestyle='-')
# plt.plot(sequence30T, color='red', label='UP data T lines', marker='', linestyle='-')
# # plt.title('UP sequence')
# plt.legend()
# plt.tight_layout()
# plt.show()

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_APC_213]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] > sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 1:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak1T = find_peaks(sequence1T)
peak30T = find_peaks(sequence30T)

print(peak1T)
print(peak30T)
# Calculate the shift required to align sequence1T to sequence3T
shift = peak30T - peak1T

# Apply this shift to all sequences in channels A, G, C and T
sequence1A = align_sequences_based_on_shift(sequence1A, shift)
sequence1G = align_sequences_based_on_shift(sequence1G, shift)
sequence1C = align_sequences_based_on_shift(sequence1C, shift)
sequence1T = align_sequences_based_on_shift(sequence1T, shift)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(sequence10A, color='green', label='mixed data A lines', marker='', linestyle='-')
plt.plot(sequence10G, color='black', label='mixed data G lines', marker='', linestyle='-')
plt.plot(sequence10C, color='blue', label='mixed data C lines', marker='', linestyle='-')
plt.plot(sequence10T, color='red', label='mixed data T lines', marker='', linestyle='-')
plt.ylim(0, 100)
plt.ylabel('Value')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(sequence15A, color='green', label='mixed data A lines', marker='', linestyle='-')
plt.plot(sequence15G, color='black', label='mixed data G lines', marker='', linestyle='-')
plt.plot(sequence15C, color='blue', label='mixed data C lines', marker='', linestyle='-')
plt.plot(sequence15T, color='red', label='mixed data T lines', marker='', linestyle='-')
plt.ylim(0, 100)
plt.ylabel('Value')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(sequence30A, color='green', label='UP data A lines', marker='', linestyle='-')
plt.plot(sequence30G, color='black', label='UP data G lines', marker='', linestyle='-')
plt.plot(sequence30C, color='blue', label='UP data C lines', marker='', linestyle='-')
plt.plot(sequence30T, color='red', label='UP data T lines', marker='', linestyle='-')
plt.legend()
plt.tight_layout()
plt.show()

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_KRAS_12+1]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] >= sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 1:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak2C = find_peaks(sequence2C)
peak30C = find_peaks(sequence30C)

print(peak2C)
print(peak30C)

shift = peak30C - peak2C

# Apply this shift to all sequences in channels A, G, C and T
sequence2A = align_sequences_based_on_shift(sequence2A, shift)
sequence2G = align_sequences_based_on_shift(sequence2G, shift)
sequence2C = align_sequences_based_on_shift(sequence2C, shift)
sequence2T = align_sequences_based_on_shift(sequence2T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_PIK3CA_542+1]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] >= sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 2:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak3C = find_peaks(sequence3C)
peak30C = find_peaks(sequence30C)

print(peak3C)
print(peak30C)

shift = peak30C - peak3C

# Apply this shift to all sequences in channels A, G, C and T
sequence3A = align_sequences_based_on_shift(sequence3A, shift)
sequence3G = align_sequences_based_on_shift(sequence3G, shift)
sequence3C = align_sequences_based_on_shift(sequence3C, shift)
sequence3T = align_sequences_based_on_shift(sequence3T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_PIK3CA_1047+1]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] >= sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 2:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak4G = find_peaks(sequence4G)
peak30G = find_peaks(sequence30G)

print(peak4G)
print(peak30G)

shift = peak30G - peak4G

# Apply this shift to all sequences in channels A, G, C and T
sequence4A = align_sequences_based_on_shift(sequence4A, shift)
sequence4G = align_sequences_based_on_shift(sequence4G, shift)
sequence4C = align_sequences_based_on_shift(sequence4C, shift)
sequence4T = align_sequences_based_on_shift(sequence4T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_TP53_273+1]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] >= sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 2:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak5C = find_peaks(sequence5C)
peak30C = find_peaks(sequence30C)

print(peak5C)
print(peak30C)

shift = peak30C - peak5C

# Apply this shift to all sequences in channels A, G, C and T
sequence5A = align_sequences_based_on_shift(sequence5A, shift)
sequence5G = align_sequences_based_on_shift(sequence5G, shift)
sequence5C = align_sequences_based_on_shift(sequence5C, shift)
sequence5T = align_sequences_based_on_shift(sequence5T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_BRAF_600+1]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] >= sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 2:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak6G = find_peaks(sequence6G)
peak30G = find_peaks(sequence30G)

print(peak6G)
print(peak30G)

shift = peak30G - peak6G

# Apply this shift to all sequences in channels A, G, C and T
sequence6A = align_sequences_based_on_shift(sequence6A, shift)
sequence6G = align_sequences_based_on_shift(sequence6G, shift)
sequence6C = align_sequences_based_on_shift(sequence6C, shift)
sequence6T = align_sequences_based_on_shift(sequence6T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=20, start_point=sorted_peak_positions_all[0][length_ref-length_ERBB2_842+1]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] >= sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 2:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak7G = find_peaks(sequence7G)
peak30G = find_peaks(sequence30G)

shift = peak30G - peak7G

print(peak7G)
print(peak30G)
# Apply this shift to all sequences in channels A, G, C and T
sequence7A = align_sequences_based_on_shift(sequence7A, shift)
sequence7G = align_sequences_based_on_shift(sequence7G, shift)
sequence7C = align_sequences_based_on_shift(sequence7C, shift)
sequence7T = align_sequences_based_on_shift(sequence7T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_TP53_196+1]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] >= sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 2:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak8T = find_peaks(sequence8T)
peak30T = find_peaks(sequence30T)

shift = peak30T - peak8T

print(peak8T)
print(peak30T)
# Apply this shift to all sequences in channels A, G, C and T
sequence8A = align_sequences_based_on_shift(sequence8A, shift)
sequence8G = align_sequences_based_on_shift(sequence8G, shift)
sequence8C = align_sequences_based_on_shift(sequence8C, shift)
sequence8T = align_sequences_based_on_shift(sequence8T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_ERBB2_777]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] >= sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 2:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak9T = find_peaks(sequence9T)
peak30T = find_peaks(sequence30T)

shift = peak30T - peak9T

print(peak9T)
print(peak30T)
# Apply this shift to all sequences in channels A, G, C and T
sequence9A = align_sequences_based_on_shift(sequence9A, shift)
sequence9G = align_sequences_based_on_shift(sequence9G, shift)
sequence9C = align_sequences_based_on_shift(sequence9C, shift)
sequence9T = align_sequences_based_on_shift(sequence9T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_APC_1114]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] >= sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 2:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak10T = find_peaks(sequence10T)
peak30T = find_peaks(sequence30T)

shift = peak30T - peak10T

print(peak10T)
print(peak30T)
# Apply this shift to all sequences in channels A, G, C and T
sequence10A = align_sequences_based_on_shift(sequence10A, shift)
sequence10G = align_sequences_based_on_shift(sequence10G, shift)
sequence10C = align_sequences_based_on_shift(sequence10C, shift)
sequence10T = align_sequences_based_on_shift(sequence10T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_KRAS_61]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] >= sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 2:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak11T = find_peaks(sequence11T)
peak30T = find_peaks(sequence30T)

shift = peak30T - peak11T

print(peak11T)
print(peak30T)
# Apply this shift to all sequences in channels A, G, C and T
sequence11A = align_sequences_based_on_shift(sequence11A, shift)
sequence11G = align_sequences_based_on_shift(sequence11G, shift)
sequence11C = align_sequences_based_on_shift(sequence11C, shift)
sequence11T = align_sequences_based_on_shift(sequence11T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=40, start_point=sorted_peak_positions_all[0][length_ref-length_APC_1450]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] >= sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 1:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak12T = find_peaks(sequence12T)
peak30T = find_peaks(sequence30T)

shift = peak30T - peak12T

print(peak12T)
print(peak30T)
# Apply this shift to all sequences in channels A, G, C and T
sequence12A = align_sequences_based_on_shift(sequence12A, shift)
sequence12G = align_sequences_based_on_shift(sequence12G, shift)
sequence12C = align_sequences_based_on_shift(sequence12C, shift)
sequence12T = align_sequences_based_on_shift(sequence12T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_KRAS_146]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] >= sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 2:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak13T = find_peaks(sequence13T)
peak30T = find_peaks(sequence30T)

shift = peak30T - peak13T

print(peak13T)
print(peak30T)
# Apply this shift to all sequences in channels A, G, C and T
sequence13A = align_sequences_based_on_shift(sequence13A, shift)
sequence13G = align_sequences_based_on_shift(sequence13G, shift)
sequence13C = align_sequences_based_on_shift(sequence13C, shift)
sequence13T = align_sequences_based_on_shift(sequence13T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_ERBB2_310]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] >= sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 2:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak14T = find_peaks(sequence14T)
peak30T = find_peaks(sequence30T)

shift = peak30T - peak14T

print(peak14T)
print(peak30T)
# Apply this shift to all sequences in channels A, G, C and T
sequence14A = align_sequences_based_on_shift(sequence14A, shift)
sequence14G = align_sequences_based_on_shift(sequence14G, shift)
sequence14C = align_sequences_based_on_shift(sequence14C, shift)
sequence14T = align_sequences_based_on_shift(sequence14T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_APC_876]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] >= sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 2:
                return peaks[0][0]  # Return the index of the first peak
    return start_point  # Return the starting point if no first peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

peak15T = find_peaks(sequence15T)
peak30T = find_peaks(sequence30T)

shift = peak30T - peak15T

print(peak15T)
print(peak30T)
# Apply this shift to all sequences in channels A, G, C and T
sequence15A = align_sequences_based_on_shift(sequence15A, shift)
sequence15G = align_sequences_based_on_shift(sequence15G, shift)
sequence15C = align_sequences_based_on_shift(sequence15C, shift)
sequence15T = align_sequences_based_on_shift(sequence15T, shift)  # For consistency

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(ref_sequenceA, color='green', label='mixed data A lines', marker='', linestyle='-')
plt.plot(ref_sequenceG, color='black', label='mixed data G lines', marker='', linestyle='-')
plt.plot(ref_sequenceC, color='blue', label='mixed data C lines', marker='', linestyle='-')
plt.plot(ref_sequenceT, color='red', label='mixed data T lines', marker='', linestyle='-')
plt.ylim(0, 100)
plt.ylabel('Value')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(sequence30A, color='green', label='UP data A lines', marker='', linestyle='-')
plt.plot(sequence30G, color='black', label='UP data G lines', marker='', linestyle='-')
plt.plot(sequence30C, color='blue', label='UP data C lines', marker='', linestyle='-')
plt.plot(sequence30T, color='red', label='UP data T lines', marker='', linestyle='-')
# plt.title('UP sequence')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
# Plot the alignment path for APC_213,APC_2132 and AD peak lines
plt.subplot(3, 1, 1)
plt.plot(sequence10A, color='green', label='mixed data A lines', marker='', linestyle='-')
plt.plot(sequence10G, color='black', label='mixed data G lines', marker='', linestyle='-')
plt.plot(sequence10C, color='blue', label='mixed data C lines', marker='', linestyle='-')
plt.plot(sequence10T, color='red', label='mixed data T lines', marker='', linestyle='-')
# plt.title('mixed sequence')
plt.ylim(0, 100)
plt.ylabel('Value')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(sequence15A, color='green', label='mixed data A lines', marker='', linestyle='-')
plt.plot(sequence15G, color='black', label='mixed data G lines', marker='', linestyle='-')
plt.plot(sequence15C, color='blue', label='mixed data C lines', marker='', linestyle='-')
plt.plot(sequence15T, color='red', label='mixed data T lines', marker='', linestyle='-')
# plt.title('mixed sequence')
plt.ylim(0, 100)
plt.ylabel('Value')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(sequence30A, color='green', label='UP data A lines', marker='', linestyle='-')
plt.plot(sequence30G, color='black', label='UP data G lines', marker='', linestyle='-')
plt.plot(sequence30C, color='blue', label='UP data C lines', marker='', linestyle='-')
plt.plot(sequence30T, color='red', label='UP data T lines', marker='', linestyle='-')
# plt.title('UP sequence')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
# Plot the alignment path for APC_213,APC_2132 and AD peak lines
plt.subplot(3, 1, 1)
plt.plot(sequence6A, color='green', label='mixed data A lines', marker='', linestyle='-')
plt.plot(sequence6G, color='black', label='mixed data G lines', marker='', linestyle='-')
plt.plot(sequence6C, color='blue', label='mixed data C lines', marker='', linestyle='-')
plt.plot(sequence6T, color='red', label='mixed data T lines', marker='', linestyle='-')
# plt.title('mixed sequence')
plt.ylim(0, 100)
plt.ylabel('Value')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(sequence7A, color='green', label='mixed data A lines', marker='', linestyle='-')
plt.plot(sequence7G, color='black', label='mixed data G lines', marker='', linestyle='-')
plt.plot(sequence7C, color='blue', label='mixed data C lines', marker='', linestyle='-')
plt.plot(sequence7T, color='red', label='mixed data T lines', marker='', linestyle='-')
# plt.title('mixed sequence')
plt.ylim(0, 100)
plt.ylabel('Value')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(sequence30A, color='green', label='UP data A lines', marker='', linestyle='-')
plt.plot(sequence30G, color='black', label='UP data G lines', marker='', linestyle='-')
plt.plot(sequence30C, color='blue', label='UP data C lines', marker='', linestyle='-')
plt.plot(sequence30T, color='red', label='UP data T lines', marker='', linestyle='-')
# plt.title('UP sequence')
plt.legend()
plt.tight_layout()
plt.show()

# Define the sequences for each channel
sequencesA = [sequence1A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [sequence1G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G,sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G, sequence15G]  # Define all sequences for channel G
sequencesC = [sequence1C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C,sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [sequence1T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T,sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A
target_sequenceG = sequence30G
target_sequenceC = sequence30C
target_sequenceT = sequence30T

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    for i, seq in enumerate(sequences, start=1):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = n * seq  
            distance = dtw_distance_1(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 1 as the best sequence number
channels_with_sequence1 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 1]

APC_213 = 0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence1) >= 2:  # At least two channels with sequence 1
    valid_n_count = sum(1 for _, n in channels_with_sequence1 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        APC_213 = 1  # Store APC_213 = 1

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")

if length_ref != length_APC_213:
    APC_213 = 0
# Print the value of APC_213
print(f"Value of APC_213: {APC_213}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = n * sequence1A  # Combine with the first sequence in the channel
    distance = dtw_distance_1(combined_sequence, sequence30A)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A = best_n
print("Best n for sequence1A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = n * sequence1G  # Combine with the first sequence in the channel
    distance = dtw_distance_1(combined_sequence, sequence30G)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_G = best_n
print("Best n for sequence1G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = n * sequence1C  # Combine with the first sequence in the channel
    distance = dtw_distance_1(combined_sequence, sequence30C)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_C = best_n
print("Best n for sequence1C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = n * sequence1T  # Combine with the first sequence in the channel
    distance = dtw_distance_1(combined_sequence, sequence30T)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_T = best_n
print("Best n for sequence1T:", best_results_T)

# Calculate APC_213 channel A weight
if APC_213==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif APC_213==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence1A = sequence1A*results_A
sequence1A_new=sequence1A*results_A

# Calculate APC_213 channel G weight
if APC_213==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif APC_213==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence1G = sequence1G*results_G
sequence1G_new=sequence1G*results_G

# Calculate APC_213 channel C weight
if APC_213==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif APC_213==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence1C = sequence1C*results_C
sequence1C_new=sequence1C*results_C

# Calculate APC_213 channel T weight
if APC_213==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif APC_213==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence1T = sequence1T*results_T
sequence1T_new=sequence1T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence1A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [combined_sequence1G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G, sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G,sequence15G]  # Define all sequences for channel G
sequencesC = [combined_sequence1C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C, sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [combined_sequence1T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T, sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A
target_sequenceG = sequence30G
target_sequenceC = sequence30C
target_sequenceT = sequence30T

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    first_sequence = sequences[0]
    for i, seq in enumerate(sequences[1:], start=2):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = first_sequence + n * seq  # Combine with the first sequence in the channel
            distance = dtw_distance_2(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 2 as the best sequence number
channels_with_sequence2 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 2]
KRAS_12=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence2) >= 2:  # At least two channels with sequence 2
    valid_n_count = sum(1 for _, n in channels_with_sequence2 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        KRAS_12 = 1  # Store KRAS_12 = 1

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")

if length_ref != length_APC_213 and length_ref != length_KRAS_12:
    KRAS_12 = 0

if length_ref == length_KRAS_12:
    KRAS_12 = 1

# Print the value of KRAS_12
print(f"Value of KRAS_12: {KRAS_12}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence1A + n * sequence2A
    distance = dtw_distance_2(combined_sequence, sequence30A)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A = best_n
print("Best n for sequence2A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence1G + n * sequence2G
    distance = dtw_distance_2(combined_sequence, sequence30G)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_G = best_n
print("Best n for sequence2G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence1C + n * sequence2C
    distance = dtw_distance_2(combined_sequence, sequence30C)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_C = best_n
print("Best n for sequence2C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence1T + n * sequence2T
    distance = dtw_distance_2(combined_sequence, sequence30T)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_T = best_n
print("Best n for sequence2T:", best_results_T)

# Calculate KRAS_12 channel A weight
if KRAS_12==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif KRAS_12==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence2A = combined_sequence1A+sequence2A*results_A
sequence2A_new=sequence2A*results_A

# Calculate KRAS_12 channel G weight
if KRAS_12==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif KRAS_12==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence2G = combined_sequence1G+sequence2G*results_G
sequence2G_new=sequence2G*results_G

# Calculate KRAS_12 channel C weight
if KRAS_12==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif KRAS_12==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence2C = combined_sequence1C+sequence2C*results_C
sequence2C_new=sequence2C*results_C

# Calculate KRAS_12 channel T weight
if KRAS_12==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif KRAS_12==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence2T = combined_sequence1T+sequence2T*results_T
sequence2T_new=sequence2T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence2A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [combined_sequence2G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G, sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G,sequence15G]  # Define all sequences for channel G
sequencesC = [combined_sequence2C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C, sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [combined_sequence2T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T, sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A 
target_sequenceG = sequence30G 
target_sequenceC = sequence30C
target_sequenceT = sequence30T 

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    first_sequence = sequences[0]
    for i, seq in enumerate(sequences[2:], start=3):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = first_sequence + n * seq  # Combine with the first sequence in the channel
            distance = dtw_distance_3(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 3 as the best sequence number
channels_with_sequence3 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 3]
PIK3CA_542=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence3) >= 2:  # At least two channels with sequence 3
    valid_n_count = sum(1 for _, n in channels_with_sequence3 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        PIK3CA_542 = 1  

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")

if length_ref != length_APC_213 and length_ref != length_KRAS_12 and length_ref != length_PIK3CA_542:
    PIK3CA_542 = 0
if length_ref == length_PIK3CA_542:
    PIK3CA_542 = 1
# Print the value of APC_213
print(f"Value of PIK3CA_542: {PIK3CA_542}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence2A + n * sequence3A
    distance = dtw_distance_3(combined_sequence, target_sequenceA)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A = best_n
print("Best n for sequence3A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence2G + n * sequence3G
    distance = dtw_distance_3(combined_sequence, sequence30G)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_G =  best_n
print("Best n for sequence3G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence2C + n * sequence3C
    distance = dtw_distance_3(combined_sequence, sequence30C)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_C = best_n
print("Best n for sequence3C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence2T + n * sequence3T
    distance = dtw_distance_3(combined_sequence, sequence30T)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_T = best_n
print("Best n for sequence3T:", best_results_T)

# Calculate PIK3CA_542 channel A weight
if PIK3CA_542==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif PIK3CA_542==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence3A = combined_sequence2A+sequence3A*results_A
sequence3A_new=sequence3A*results_A

# Calculate PIK3CA_542 channel G weight
if PIK3CA_542==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif PIK3CA_542==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence3G = combined_sequence2G+sequence3G*results_G
sequence3G_new=sequence3G*results_G

# Calculate PIK3CA_542 channel C weight
if PIK3CA_542==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif PIK3CA_542==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence3C = combined_sequence2C+sequence3C*results_C
sequence3C_new=sequence3C*results_C

# Calculate PIK3CA_542 channel T weight
if PIK3CA_542==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif PIK3CA_542==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence3T = combined_sequence2T+sequence3T*results_T
sequence3T_new=sequence3T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence3A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [combined_sequence3G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G, sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G,sequence15G]  # Define all sequences for channel G
sequencesC = [combined_sequence3C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C, sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [combined_sequence3T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T, sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A 
target_sequenceG = sequence30G
target_sequenceC = sequence30C
target_sequenceT = sequence30T

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    first_sequence = sequences[0]
    for i, seq in enumerate(sequences[3:], start=4):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = first_sequence + n * seq  # Combine with the first sequence in the channel
            distance = dtw_distance_4(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 4 as the best sequence number
channels_with_sequence4 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 4]
PIK3CA_1047=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence4) >= 2:  # At least two channels with sequence 4
    valid_n_count = sum(1 for _, n in channels_with_sequence4 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        PIK3CA_1047 = 1  # Store PIK3CA_1047= 1

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")

if length_ref == length_PIK3CA_1047:
    PIK3CA_1047 = 1

print(f"Value of PIK3CA_1047: {PIK3CA_1047}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequenceA = combined_sequence3A + n * sequence4A
    distance = dtw_distance_4(combined_sequenceA, target_sequenceA)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A = best_n
print("Best n for sequence4A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequenceG = combined_sequence3G + n * sequence4G 
    distance = dtw_distance_4(combined_sequenceG, target_sequenceG)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result1
best_results_G = best_n
print("Best n for sequence4G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequenceC = combined_sequence3C + n * sequence4C
    distance = dtw_distance_4(combined_sequenceC, target_sequenceC)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result1
best_results_C = best_n
print("Best n for sequence4C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequenceT = combined_sequence3T + n * sequence4T
    distance = dtw_distance_4(combined_sequenceT, target_sequenceT)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result1
best_results_T = best_n
print("Best n for sequence4T:", best_results_T)

# Calculate PIK3CA_1047 channel A weight
if PIK3CA_1047==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif PIK3CA_1047==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence4A = combined_sequence3A+sequence4A*results_A
sequence4A_new=sequence4A*results_A

# Calculate PIK3CA_1047 channel G weight
if PIK3CA_1047==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif PIK3CA_1047==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence4G = combined_sequence3G+sequence4G*results_G
sequence4G_new=sequence4G*results_G

# Calculate PIK3CA_1047 channel C weight
if PIK3CA_1047==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif PIK3CA_1047==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence4C = combined_sequence3C+sequence4C*results_C
sequence4C_new=sequence4C*results_C

# Calculate PIK3CA_1047 channel T weight
if PIK3CA_1047==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif PIK3CA_1047==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence4T = combined_sequence3T+sequence4T*results_T
sequence4T_new=sequence4T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence4A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [combined_sequence4G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G, sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G,sequence15G]  # Define all sequences for channel G
sequencesC = [combined_sequence4C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C, sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [combined_sequence4T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T, sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A
target_sequenceG = sequence30G
target_sequenceC = sequence30C
target_sequenceT = sequence30T

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    first_sequence = sequences[0]
    for i, seq in enumerate(sequences[4:], start=5):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = first_sequence + n * seq  # Combine with the first sequence in the channel
            distance = dtw_distance_5(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 5 as the best sequence number
channels_with_sequence5 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 5]
TP53_273=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence5) >= 2:  # At least two channels with sequence 5
    valid_n_count = sum(1 for _, n in channels_with_sequence5 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        TP53_273 = 1

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")

if length_ref == length_TP53_273:
    TP53_273 = 1

print(f"Value of TP53_273: {TP53_273}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence4A + n * sequence5A
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A =best_n
print("Best n for sequence5A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence4G + n * sequence5G
    distance = dtw_distance_5(combined_sequence, target_sequenceG)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_G = best_n
print("Best n for sequence5G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence4C + n * sequence5C
    distance = dtw_distance_5(combined_sequence, target_sequenceC)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_C = best_n
print("Best n for sequence5C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence4T + n * sequence5T
    distance = dtw_distance_5(combined_sequence, target_sequenceT)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_T =  best_n
print("Best n for sequence5T:", best_results_T)

# Calculate TP53_273 channel A weight
if TP53_273==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif TP53_273==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence5A = combined_sequence4A+sequence5A*results_A
sequence5A_new=sequence5A*results_A

# Calculate TP53_273 channel G weight
if TP53_273==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif TP53_273==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence5G = combined_sequence4G+sequence5G*results_G
sequence5G_new=sequence5G*results_G

# Calculate TP53_273 channel C weight
if TP53_273==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif TP53_273==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence5C = combined_sequence4C+sequence5C*results_C
sequence5C_new=sequence5C*results_C

# Calculate TP53_273 channel T weight
if TP53_273==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif TP53_273==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence5T = combined_sequence4T+sequence5T*results_T
sequence5T_new=sequence5T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence5A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [combined_sequence5G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G, sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G,sequence15G]  # Define all sequences for channel G
sequencesC = [combined_sequence5C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C, sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [combined_sequence5T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T, sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A
target_sequenceG = sequence30G
target_sequenceC = sequence30C
target_sequenceT = sequence30T

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    first_sequence = sequences[0]
    for i, seq in enumerate(sequences[4:], start=5):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = first_sequence + n * seq  # Combine with the first sequence in the channel
            distance = dtw_distance_6(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 6 as the best sequence number
channels_with_sequence6 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 6]

BRAF_600=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence6) >= 2:  # At least two channels with sequence 6
    valid_n_count = sum(1 for _, n in channels_with_sequence6 if n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        BRAF_600 = 1  
# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")

if length_ref == length_BRAF_600:
    BRAF_600 = 1

print(f"Value of BRAF_600: {BRAF_600}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence5A + n * sequence6A
    distance = dtw_distance_6(combined_sequence, target_sequenceA)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A = best_n
print("Best n for sequence6A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence5G + n * sequence6G
    distance = dtw_distance_6(combined_sequence, target_sequenceG)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_G = best_n
print("Best n for sequence6G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence5C + n * sequence6C
    distance = dtw_distance_6(combined_sequence, target_sequenceC)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_C =  best_n
print("Best n for sequence6C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence5T + n * sequence6T
    distance = dtw_distance_6(combined_sequence, target_sequenceT)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_T =  best_n
print("Best n for sequence6T:", best_results_T)

# Calculate BRAF_600 channel A weight
if BRAF_600==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif BRAF_600==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence6A = combined_sequence5A+sequence6A*results_A
sequence6A_new=sequence6A*results_A

# Calculate BRAF_600 channel G weight
if BRAF_600==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif BRAF_600==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence6G = combined_sequence5G+sequence6G*results_G
sequence6G_new=sequence6G*results_G

# Calculate BRAF_600 channel C weight
if BRAF_600==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif BRAF_600==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence6C = combined_sequence5C+sequence6C*results_C
sequence6C_new=sequence6C*results_C

# Calculate BRAF_600 channel T weight
if BRAF_600==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif BRAF_600==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence6T = combined_sequence5T+sequence6T*results_T
sequence6T_new=sequence6T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence6A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [combined_sequence6G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G, sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G,sequence15G]  # Define all sequences for channel G
sequencesC = [combined_sequence6C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C, sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [combined_sequence6T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T, sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A
target_sequenceG = sequence30G
target_sequenceC = sequence30C
target_sequenceT = sequence30T

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    first_sequence = sequences[0]
    for i, seq in enumerate(sequences[4:], start=5):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = first_sequence + n * seq  # Combine with the first sequence in the channel
            distance = dtw_distance_7(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 7 as the best sequence number
channels_with_sequence7 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 7]

ERBB2_842=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence7) >= 2:  # At least two channels with sequence 7
    valid_n_count = sum(1 for _, n in channels_with_sequence7 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        ERBB2_842 = 1  

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")

if length_ref == length_ERBB2_842:
    ERBB2_842 = 1

print(f"Value of ERBB2_842: {ERBB2_842}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence6A + n * sequence7A
    distance = dtw_distance_7(combined_sequence, target_sequenceA)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A = best_n
print("Best n for sequence7A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence6G + n * sequence7G
    distance = dtw_distance_7(combined_sequence, target_sequenceG)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_G = best_n
print("Best n for sequence7G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence6C + n * sequence7C
    distance = dtw_distance_7(combined_sequence, target_sequenceC)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_C =  best_n
print("Best n for sequence7C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence6T + n * sequence7T
    distance = dtw_distance_7(combined_sequence, target_sequenceT)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_T =best_n
print("Best n for sequence7T:", best_results_T)

# Calculate ERBB2_842 channel A weight
if ERBB2_842==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif ERBB2_842==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3]or [0.2]), 0.2)
else:
    results_A=0

combined_sequence7A = combined_sequence6A+sequence7A*results_A
sequence7A_new=sequence7A*results_A

# Calculate ERBB2_842 channel G weight
if ERBB2_842==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif ERBB2_842==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3]or [0.2]), 0.2)
else:
    results_G=0

combined_sequence7G = combined_sequence6G+sequence7G*results_G
sequence7G_new=sequence7G*results_G

# Calculate ERBB2_842 channel C weight
if ERBB2_842==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif ERBB2_842==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence7C = combined_sequence6C+sequence7C*results_C
sequence7C_new=sequence7C*results_C

# Calculate ERBB2_842 channel T weight
if ERBB2_842==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif ERBB2_842==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3]or [0.2]), 0.2)
else:
    results_T=0

combined_sequence7T = combined_sequence6T+sequence7T*results_T
sequence7T_new=sequence7T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence7A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [combined_sequence7G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G, sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G,sequence15G]  # Define all sequences for channel G
sequencesC = [combined_sequence7C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C, sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [combined_sequence7T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T, sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A
target_sequenceG = sequence30G
target_sequenceC = sequence30C
target_sequenceT = sequence30T

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    first_sequence = sequences[0]
    for i, seq in enumerate(sequences[4:], start=5):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = first_sequence + n * seq  # Combine with the first sequence in the channel
            distance = dtw_distance_8(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 8 as the best sequence number
channels_with_sequence8 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 8]
# print(channels_with_sequence6)
TP53_196 = 0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence8) >= 2:  # At least two channels with sequence 8
    valid_n_count = sum(1 for _, n in channels_with_sequence8 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        TP53_196 = 1  

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")
if length_ref == length_TP53_196:
    TP53_196 = 1

print(f"Value of TP53_196: {TP53_196}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence7A + n * sequence8A
    distance = dtw_distance_8(combined_sequence, target_sequenceA)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A =  best_n
print("Best n for sequence8A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence7G + n * sequence8G
    distance = dtw_distance_8(combined_sequence, target_sequenceG)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_G = best_n
print("Best n for sequence8G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence7C + n * sequence8C
    distance = dtw_distance_8(combined_sequence, target_sequenceC)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_C = best_n
print("Best n for sequence8C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence7T + n * sequence8T
    distance = dtw_distance_8(combined_sequence, target_sequenceT)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_T = best_n
print("Best n for sequence8T:", best_results_T)

if not np.any(np.all(ref_sequenceA == np.array([sequence1A, sequence2A, sequence3A, sequence4A, 
                                                sequence5A, sequence6A, sequence7A, sequence8A]), axis=1)):
    TP53_196 = 0

# Calculate TP53_196 channel A weight
if TP53_196==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif TP53_196==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3]or [0.2]), 0.2)
else:
    results_A=0

combined_sequence8A = combined_sequence7A+sequence8A*results_A
sequence8A_new=sequence8A*results_A

# Calculate TP53_196 channel G weight
if TP53_196==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif TP53_196==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3]or [0.2]), 0.2)
else:
    results_G=0

combined_sequence8G = combined_sequence7G+sequence8G*results_G
sequence8G_new=sequence8G*results_G

# Calculate TP53_196 channel C weight
if TP53_196==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif TP53_196==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3]or [0.2]), 0.2)
else:
    results_C=0

combined_sequence8C = combined_sequence7C+sequence8C*results_C
sequence8C_new=sequence8C*results_C

# Calculate TP53_196 channel T weight
if TP53_196==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif TP53_196==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3]or [0.2]), 0.2)
else:
    results_T=0

combined_sequence8T = combined_sequence7T+sequence8T*results_T
sequence8T_new=sequence8T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence8A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [combined_sequence8G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G, sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G,sequence15G]  # Define all sequences for channel G
sequencesC = [combined_sequence8C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C, sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [combined_sequence8T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T, sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A
target_sequenceG = sequence30G
target_sequenceC = sequence30C
target_sequenceT = sequence30T

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    first_sequence = sequences[0]
    for i, seq in enumerate(sequences[4:], start=5):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = first_sequence + n * seq  # Combine with the first sequence in the channel
            distance = dtw_distance_9(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 9 as the best sequence number
channels_with_sequence9 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 9]
# print(channels_with_sequence6)
ERBB2_777 = 0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence9) >= 2:  # At least two channels with sequence 9
    valid_n_count = sum(1 for _, n in channels_with_sequence9 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        ERBB2_777 = 1  

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")
if length_ref == length_ERBB2_777:
    ERBB2_777 = 1

print(f"Value of ERBB2_777: {ERBB2_777}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence8A + n * sequence9A
    distance = dtw_distance_9(combined_sequence, target_sequenceA)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A =  best_n
print("Best n for sequence9A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence8G + n * sequence9G
    distance = dtw_distance_9(combined_sequence, target_sequenceG)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_G =  best_n
print("Best n for sequence9G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence8C + n * sequence9C
    distance = dtw_distance_9(combined_sequence, target_sequenceC)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_C =  best_n
print("Best n for sequence9C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence8T + n * sequence9T
    distance = dtw_distance_9(combined_sequence, target_sequenceT)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_T =  best_n
print("Best n for sequence9T:", best_results_T)

# Calculate ERBB2_777 channel A weight
if ERBB2_777==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif ERBB2_777==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3]or [0.2]), 0.2)
else:
    results_A=0

combined_sequence9A = combined_sequence8A+sequence9A*results_A
sequence9A_new=sequence9A*results_A

# Calculate ERBB2_777 channel G weight
if ERBB2_777==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif ERBB2_777==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3]or [0.2]), 0.2)
else:
    results_G=0

combined_sequence9G = combined_sequence8G+sequence9G*results_G
sequence9G_new=sequence9G*results_G

# Calculate ERBB2_777 channel C weight
if ERBB2_777==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif ERBB2_777==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3]or [0.2]), 0.2)
else:
    results_C=0

combined_sequence9C = combined_sequence8C+sequence9C*results_C
sequence9C_new=sequence9C*results_C

# Calculate ERBB2_777 channel T weight
if ERBB2_777==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif ERBB2_777==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3]or [0.2]), 0.2)
else:
    results_T=0

combined_sequence9T = combined_sequence8T+sequence9T*results_T
sequence9T_new=sequence9T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence9A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [combined_sequence9G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G, sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G,sequence15G]  # Define all sequences for channel G
sequencesC = [combined_sequence9C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C, sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [combined_sequence9T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T, sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A
target_sequenceG = sequence30G
target_sequenceC = sequence30C
target_sequenceT = sequence30T

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    first_sequence = sequences[0]
    for i, seq in enumerate(sequences[4:], start=5):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = first_sequence + n * seq  # Combine with the first sequence in the channel
            distance = dtw_distance_10(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 10 as the best sequence number
channels_with_sequence10 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 10]
APC_1114=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence10) >= 2:  # At least two channels with sequence 10
    valid_n_count = sum(1 for _, n in channels_with_sequence10 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        APC_1114 = 1  

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")
if length_ref == length_APC_1114:
    APC_1114 = 1

print(f"Value of APC_1114: {APC_1114}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence9A + n * sequence10A
    distance = dtw_distance_10(combined_sequence, target_sequenceA)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A =best_n
print("Best n for sequence10A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence9G + n * sequence10G
    distance = dtw_distance_10(combined_sequence, target_sequenceG)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_G =  best_n
print("Best n for sequence10G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence9C + n * sequence10C
    distance = dtw_distance_10(combined_sequence, target_sequenceC)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_C =  best_n
print("Best n for sequence10C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence9T + n * sequence10T
    distance = dtw_distance_10(combined_sequence, target_sequenceT)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_T =  best_n
print("Best n for sequence10T:", best_results_T)

# Calculate APC_1114 channel A weight
if APC_1114==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif APC_1114==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence10A = combined_sequence9A+sequence10A*results_A
sequence10A_new=sequence10A*results_A

# Calculate APC_1114 channel G weight
if APC_1114==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif APC_1114==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence10G = combined_sequence9G+sequence10G*results_G
sequence10G_new=sequence10G*results_G

# Calculate APC_1114 channel C weight
if APC_1114==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif APC_1114==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence10C = combined_sequence9C+sequence10C*results_C
sequence10C_new=sequence10C*results_C

# Calculate APC_1114 channel T weight
if APC_1114==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif APC_1114==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence10T = combined_sequence9T+sequence10T*results_T
sequence10T_new=sequence10T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence10A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [combined_sequence10G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G, sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G,sequence15G]  # Define all sequences for channel G
sequencesC = [combined_sequence10C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C, sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [combined_sequence10T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T, sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A
target_sequenceG = sequence30G
target_sequenceC = sequence30C
target_sequenceT = sequence30T

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    first_sequence = sequences[0]
    for i, seq in enumerate(sequences[4:], start=5):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = first_sequence + n * seq  # Combine with the first sequence in the channel
            distance = dtw_distance_11(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 11 as the best sequence number
channels_with_sequence11 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 11]
KRAS_61=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence11) >= 2:  # At least two channels with sequence 11
    valid_n_count = sum(1 for _, n in channels_with_sequence11 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        KRAS_61 = 1  

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")
if length_ref == length_KRAS_61:
    KRAS_61 = 1

print(f"Value of KRAS_61: {KRAS_61}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence10A + n * sequence11A
    distance = dtw_distance_11(combined_sequence, target_sequenceA)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A =  best_n
print("Best n for sequence11A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence10G + n * sequence11G
    distance = dtw_distance_11(combined_sequence, target_sequenceG)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_G =  best_n
print("Best n for sequence11G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence10C + n * sequence11C
    distance = dtw_distance_11(combined_sequence, target_sequenceC)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_C = best_n
print("Best n for sequence11C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence10T + n * sequence11T
    distance = dtw_distance_11(combined_sequence, target_sequenceT)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_T =  best_n
print("Best n for sequence11T:", best_results_T)

# Calculate KRAS_61 channel A weight
if KRAS_61==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif KRAS_61==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence11A = combined_sequence10A+sequence11A*results_A
sequence11A_new=sequence11A*results_A

# Calculate KRAS_61 channel G weight
if KRAS_61==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif KRAS_61==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence11G = combined_sequence10G+sequence11G*results_G
sequence11G_new=sequence11G*results_G

# Calculate KRAS_61 channel C weight
if KRAS_61==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif KRAS_61==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence11C = combined_sequence10C+sequence11C*results_C
sequence11C_new=sequence11C*results_C

# Calculate KRAS_61 channel T weight
if KRAS_61==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif KRAS_61==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence11T = combined_sequence10T+sequence11T*results_T
sequence11T_new=sequence11T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence11A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [combined_sequence11G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G, sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G,sequence15G]  # Define all sequences for channel G
sequencesC = [combined_sequence11C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C, sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [combined_sequence11T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T, sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A
target_sequenceG = sequence30G
target_sequenceC = sequence30C
target_sequenceT = sequence30T

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    first_sequence = sequences[0]
    for i, seq in enumerate(sequences[4:], start=5):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = first_sequence + n * seq  # Combine with the first sequence in the channel
            distance = dtw_distance_12(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 12 as the best sequence number
channels_with_sequence12 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 12]
APC_1450=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence12) >= 2:  # At least two channels with sequence 12
    valid_n_count = sum(1 for _, n in channels_with_sequence12 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        APC_1450 = 1  

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")
if length_ref == length_APC_1450:
    APC_1450 = 1

print(f"Value of APC_1450: {APC_1450}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence11A + n * sequence12A
    distance = dtw_distance_12(combined_sequence, target_sequenceA)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A =  best_n
print("Best n for sequence12A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence11G + n * sequence12G
    distance = dtw_distance_12(combined_sequence, target_sequenceG)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_G =  best_n
print("Best n for sequence12G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence11C + n * sequence12C
    distance = dtw_distance_12(combined_sequence, target_sequenceC)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_C = best_n
print("Best n for sequence12C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence11T + n * sequence12T
    distance = dtw_distance_12(combined_sequence, target_sequenceT)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_T = best_n
print("Best n for sequence12T:", best_results_T)

# Calculate APC_1450 channel A weight
if APC_1450==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif APC_1450==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3]or [0.2]), 0.2)
else:
    results_A=0

combined_sequence12A = combined_sequence11A+sequence12A*results_A
sequence12A_new=sequence12A*results_A

# Calculate APC_1450 channel G weight
if APC_1450==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif APC_1450==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence12G = combined_sequence11G+sequence12G*results_G
sequence12G_new=sequence12G*results_G

# Calculate APC_1450 channel C weight
if APC_1450==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif APC_1450==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence12C = combined_sequence11C+sequence12C*results_C
sequence12C_new=sequence12C*results_C

# Calculate APC_1450 channel T weight
if APC_1450==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif APC_1450==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence12T = combined_sequence11T+sequence12T*results_T
sequence12T_new=sequence12T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence12A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [combined_sequence12G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G, sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G,sequence15G]  # Define all sequences for channel G
sequencesC = [combined_sequence12C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C, sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [combined_sequence12T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T, sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A
target_sequenceG = sequence30G
target_sequenceC = sequence30C
target_sequenceT = sequence30T

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    first_sequence = sequences[0]
    for i, seq in enumerate(sequences[4:], start=5):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = first_sequence + n * seq  # Combine with the first sequence in the channel
            distance = dtw_distance_13(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 13 as the best sequence number
channels_with_sequence13 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 13]
KRAS_146=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence13) >= 2:  # At least two channels with sequence 13
    valid_n_count = sum(1 for _, n in channels_with_sequence13 if 1.5 >= n > 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        KRAS_146 = 1  

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")
if length_ref == length_KRAS_146:
    KRAS_146 = 1

print(f"Value of KRAS_146: {KRAS_146}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence12A + n * sequence13A
    distance = dtw_distance_13(combined_sequence, target_sequenceA)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A =  best_n
print("Best n for sequence13A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence12G + n * sequence13G
    distance = dtw_distance_13(combined_sequence, target_sequenceG)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_G =  best_n
print("Best n for sequence13G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence12C + n * sequence13C
    distance = dtw_distance_13(combined_sequence, target_sequenceC)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_C =  best_n
print("Best n for sequence13C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence12T + n * sequence13T
    distance = dtw_distance_13(combined_sequence, target_sequenceT)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_T = best_n
print("Best n for sequence13T:", best_results_T)

# Calculate KRAS_146 channel A weight
if KRAS_146==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif KRAS_146==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence13A = combined_sequence12A+sequence13A*results_A
sequence13A_new=sequence13A*results_A

# Calculate KRAS_146 channel G weight
if KRAS_146==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif KRAS_146==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence13G = combined_sequence12G+sequence13G*results_G
sequence13G_new=sequence13G*results_G

# Calculate KRAS_146 channel C weight
if KRAS_146==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif KRAS_146==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence13C = combined_sequence12C+sequence13C*results_C
sequence13C_new=sequence13C*results_C

# Calculate KRAS_146 channel T weight
if KRAS_146==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif KRAS_146==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence13T = combined_sequence12T+sequence13T*results_T
sequence13T_new=sequence13T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence13A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [combined_sequence13G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G, sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G,sequence15G]  # Define all sequences for channel G
sequencesC = [combined_sequence13C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C, sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [combined_sequence13T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T, sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A
target_sequenceG = sequence30G
target_sequenceC = sequence30C
target_sequenceT = sequence30T

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    first_sequence = sequences[0]
    for i, seq in enumerate(sequences[4:], start=5):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = first_sequence + n * seq  # Combine with the first sequence in the channel
            distance = dtw_distance_14(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 14 as the best sequence number
channels_with_sequence14 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 14]
ERBB2_310=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence14) >= 2:  # At least two channels with sequence 14
    valid_n_count = sum(1 for _, n in channels_with_sequence14 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        ERBB2_310 = 1  

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")
if length_ref == length_ERBB2_310:
    ERBB2_310 = 1
# Print the value of APC_213
print(f"Value of ERBB2_310: {ERBB2_310}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence13A + n * sequence14A
    distance = dtw_distance_14(combined_sequence, target_sequenceA)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A =  best_n
print("Best n for sequence14A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence13G + n * sequence14G
    distance = dtw_distance_14(combined_sequence, target_sequenceG)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_G =  best_n
print("Best n for sequence14G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence13C + n * sequence14C
    distance = dtw_distance_14(combined_sequence, target_sequenceC)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_C =  best_n
print("Best n for sequence14C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence13T + n * sequence14T
    distance = dtw_distance_14(combined_sequence, target_sequenceT)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_T =  best_n
print("Best n for sequence14T:", best_results_T)

# Calculate ERBB2_310 channel A weight
if ERBB2_310==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif ERBB2_310==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence14A = combined_sequence13A+sequence14A*results_A
sequence14A_new=sequence14A*results_A

# Calculate ERBB2_310 channel G weight
if ERBB2_310==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif ERBB2_310==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence14G = combined_sequence13G+sequence14G*results_G
sequence14G_new=sequence14G*results_G

# Calculate ERBB2_310 channel C weight
if ERBB2_310==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif ERBB2_310==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence14C = combined_sequence13C+sequence14C*results_C
sequence14C_new=sequence14C*results_C

# Calculate ERBB2_310 channel T weight
if ERBB2_310==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif ERBB2_310==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence14T = combined_sequence13T+sequence14T*results_T
sequence14T_new=sequence14T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence14A, sequence2A, sequence3A, sequence4A,sequence5A,sequence6A,sequence7A, sequence8A, sequence9A, sequence10A, sequence11A,sequence12A,sequence13A,sequence14A,sequence15A]  # Define all sequences for channel A
sequencesG = [combined_sequence14G, sequence2G, sequence3G, sequence4G, sequence5G,sequence6G,sequence7G, sequence8G, sequence9G, sequence10G, sequence11G,sequence12G,sequence13G,sequence14G,sequence15G]  # Define all sequences for channel G
sequencesC = [combined_sequence14C, sequence2C, sequence3C, sequence4C, sequence5C,sequence6C,sequence7C, sequence8C, sequence9C, sequence10C, sequence11C,sequence12C,sequence13C,sequence14C,sequence15C]  # Define all sequences for channel C
sequencesT = [combined_sequence14T, sequence2T, sequence3T, sequence4T, sequence5T,sequence6T,sequence7T, sequence8T, sequence9T, sequence10T, sequence11T,sequence12T,sequence13T,sequence14T,sequence15T]  # Define all sequences for channel T

# Define the target sequences for each channel
target_sequenceA = sequence30A
target_sequenceG = sequence30G
target_sequenceC = sequence30C
target_sequenceT = sequence30T

# Find the best value of n and sequence for each channel
best_results = {}

for channel, sequences, target_sequence in [("A", sequencesA, target_sequenceA), ("G", sequencesG, target_sequenceG), ("C", sequencesC, target_sequenceC), ("T", sequencesT, target_sequenceT)]:
    min_distance = float('inf')
    best_n = None
    best_sequence = None
    best_sequence_number = None
    first_sequence = sequences[0]
    for i, seq in enumerate(sequences[4:], start=5):
        for n in np.linspace(0, 2, 21):  # Try different values of n
            combined_sequence = first_sequence + n * seq  # Combine with the first sequence in the channel
            distance = dtw_distance_15(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 15 as the best sequence number
channels_with_sequence15 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 15]
# print(channels_with_sequence6)
APC_876=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence15) >= 2:  # At least two channels with sequence 15
    valid_n_count = sum(1 for _, n in channels_with_sequence15 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        APC_876 = 1  

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")
if length_ref == length_APC_876:
    APC_876 = 1

print(f"Value of APC_876: {APC_876}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence14A + n * sequence15A
    distance = dtw_distance_15(combined_sequence, target_sequenceA)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_A =  best_n
print("Best n for sequence15A:", best_results_A)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence14G + n * sequence15G
    distance = dtw_distance_15(combined_sequence, target_sequenceG)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_G = best_n
print("Best n for sequence15G:", best_results_G)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence14C + n * sequence15C
    distance = dtw_distance_15(combined_sequence, target_sequenceC)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_C =  best_n
print("Best n for sequence15C:", best_results_C)

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence14T + n * sequence15T
    distance = dtw_distance_15(combined_sequence, target_sequenceT)
    if distance < min_distance:
        min_distance = distance
        best_n = n
# Store the result
best_results_T =  best_n
print("Best n for sequence15T:", best_results_T)

# Calculate APC_876 channel A weight
if APC_876==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif APC_876==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence15A = combined_sequence14A+sequence15A*results_A
sequence15A_new=sequence15A*results_A

# Calculate APC_876 channel G weight
if APC_876==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif APC_876==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence15G = combined_sequence14G+sequence15G*results_G
sequence15G_new=sequence15G*results_G

# Calculate APC_876 channel C weight
if APC_876==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif APC_876==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence15C = combined_sequence14C+sequence15C*results_C
sequence15C_new=sequence15C*results_C

# Calculate APC_876 channel T weight
if APC_876==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif APC_876==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence15T = combined_sequence14T+sequence15T*results_T
sequence15T_new=sequence15T*results_T
    
# Create a list to store the names of single-plex genes
single_plex = []  
# Check each gene and add to the list if the condition is met
if APC_213 == 1:
    print("APC_213 : ✓")
    single_plex.append("APC_213")
else:
    print("APC_213 : ✗")

if KRAS_12 == 1:
    print("KRAS_12 : ✓")
    single_plex.append("KRAS_12")
else:
    print("KRAS_12 : ✗")

if PIK3CA_542 == 1:
    print("PIK3CA_542 : ✓")
    single_plex.append("PIK3CA_542")
else:
    print("PIK3CA_542 : ✗")

if PIK3CA_1047 == 1:
    print("PIK3CA_1047 : ✓")
    single_plex.append("PIK3CA_1047")
else:
    print("PIK3CA_1047 : ✗")

if TP53_273 == 1:
    print("TP53_273 : ✓")
    single_plex.append("TP53_273")
else:
    print("TP53_273 : ✗")

if BRAF_600 == 1:
    print("BRAF_600 : ✓")
    single_plex.append("BRAF_600")
else:
    print("BRAF_600 : ✗")

if ERBB2_842 == 1:
    print("ERBB2_842 : ✓")
    single_plex.append("ERBB2_842")
else:
    print("ERBB2_842 : ✗")

if TP53_196 == 1:
    print("TP53_196 : ✓")
    single_plex.append("TP53_196")
else:
    print("TP53_196 : ✗")

if ERBB2_777 == 1:
    print("ERBB2_777 : ✓")
    single_plex.append("ERBB2_777")
else:
    print("ERBB2_777 : ✗")

if APC_1114 == 1:
    print("APC_1114 : ✓")
    single_plex.append("APC_1114")
else:
    print("APC_1114 : ✗")

if KRAS_61 == 1:
    print("KRAS_61 : ✓")
    single_plex.append("KRAS_61")
else:
    print("KRAS_61 : ✗")

if APC_1450 == 1:
    print("APC_1450 : ✓")
    single_plex.append("APC_1450")
else:
    print("APC_1450 : ✗")

if KRAS_146 == 1:
    print("KRAS_146 : ✓")
    single_plex.append("KRAS_146")
else:
    print("KRAS_146 : ✗")

if ERBB2_310 == 1:
    print("ERBB2_310 : ✓")
    single_plex.append("ERBB2_310")
else:
    print("ERBB2_310 : ✗")

if APC_876 == 1:
    print("APC_876 : ✓")
    single_plex.append("APC_876")
else:
    print("APC_876 : ✗")

# Output the list of single-plex genes
print("\nSingle-plex obtained in Multi-plex: ", ', '.join(single_plex))

plt.figure(figsize=(12, 8))
# Plot final mixed sequence
plt.subplot(2, 1, 1)
plt.plot(combined_sequence15A, color='green', label='mixed data A lines', marker='', linestyle='-', linewidth=2)
plt.plot(combined_sequence15G, color='black', label='mixed data G lines', marker='', linestyle='-', linewidth=2)
plt.plot(combined_sequence15C, color='blue', label='mixed data C lines', marker='', linestyle='-', linewidth=2)
plt.plot(combined_sequence15T, color='red', label='mixed data T lines', marker='', linestyle='-', linewidth=2)
plt.ylim(0, 100)

# Plot sequence30 (multiplex sequence)
plt.subplot(2, 1, 2)
plt.plot(sequence30A, color='green', label='UP data A lines', marker='', linestyle='-', linewidth=2)
plt.plot(sequence30G, color='black', label='UP data G lines', marker='', linestyle='-', linewidth=2)
plt.plot(sequence30C, color='blue', label='UP data C lines', marker='', linestyle='-', linewidth=2)
plt.plot(sequence30T, color='red', label='UP data T lines', marker='', linestyle='-', linewidth=2)
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(sequence1A_new, color='green', label='APC-213 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence1G_new, color='black', label='APC-213 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence1C_new, color='blue', label='APC-213 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence1T_new, color='red', label='APC-213 data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('mixed sequence')
# plt.ylabel('Value')
# plt.ylim(0, 100)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(sequence2A_new, color='green', label='KRAS-12 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence2G_new, color='black', label='KRAS-12 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence2C_new, color='blue', label='KRAS-12 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence2T_new, color='red', label='KRAS-12 data T lines', marker='', linestyle='-', linewidth=2)

# # plt.title('UP sequence')
# plt.legend()
# plt.ylim(0, 100)
# plt.tight_layout()
# plt.show() 

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(sequence3A_new, color='green', label='PIK3CA-542 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence3G_new, color='black', label='PIK3CA-542 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence3C_new, color='blue', label='PIK3CA-542 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence3T_new, color='red', label='PIK3CA-542 data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('mixed sequence')
# plt.ylabel('Value')
# plt.ylim(0, 100)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(sequence4A_new, color='green', label='PIK3CA-1047 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence4G_new, color='black', label='PIK3CA-1047 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence4C_new, color='blue', label='PIK3CA-1047 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence4T_new, color='red', label='PIK3CA-1047 data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('UP sequence')
# plt.legend()
# plt.tight_layout()
# plt.ylim(0, 100)
# plt.show()  

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(sequence5A_new, color='green', label='TP53-273 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence5G_new, color='black', label='TP53-273 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence5C_new, color='blue', label='TP53-273 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence5T_new, color='red', label='TP53-273 data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('mixed sequence')
# plt.ylabel('Value')
# plt.ylim(0, 100)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(sequence6A_new, color='green', label='BRAF-600 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence6G_new, color='black', label='BRAF-600 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence6C_new, color='blue', label='BRAF-600 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence6T_new, color='red', label='BRAF-600 data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('UP sequence')
# plt.legend()
# plt.tight_layout()
# plt.ylim(0, 100)
# plt.show()  

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(sequence7A_new, color='green', label='ERBB2-842 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence7G_new, color='black', label='ERBB2-842 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence7C_new, color='blue', label='ERBB2-842 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence7T_new, color='red', label='ERBB2-842 data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('mixed sequence')
# plt.ylabel('Value')
# plt.ylim(0, 100)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(sequence8A_new, color='green', label='TP53-196 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence8G_new, color='black', label='TP53-196 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence8C_new, color='blue', label='TP53-196 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence8T_new, color='red', label='TP53-196 data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('UP sequence')
# plt.legend()
# plt.tight_layout()
# plt.ylim(0, 100)
# plt.show()  

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(sequence9A_new, color='green', label='ERBB2-777 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence9G_new, color='black', label='ERBB2-777 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence9C_new, color='blue', label='ERBB2-777 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence9T_new, color='red', label='ERBB2-777 data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('mixed sequence')
# plt.ylabel('Value')
# plt.ylim(0, 100)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(sequence10A_new, color='green', label='APC-1114 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence10G_new, color='black', label='APC-1114 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence10C_new, color='blue', label='APC-1114 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence10T_new, color='red', label='APC-1114 data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('UP sequence')
# plt.legend()
# plt.tight_layout()
# plt.ylim(0, 100)
# plt.show()  

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(sequence11A_new, color='green', label='KRAS-61 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence11G_new, color='black', label='KRAS-61 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence11C_new, color='blue', label='KRAS-61 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence11T_new, color='red', label='KRAS-61 data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('mixed sequence')
# plt.ylabel('Value')
# plt.ylim(0, 100)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(sequence12A_new, color='green', label='APC-1450 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence12G_new, color='black', label='APC-1450 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence12C_new, color='blue', label='APC-1450 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence12T_new, color='red', label='APC-1450 data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('UP sequence')
# plt.legend()
# plt.tight_layout()
# plt.ylim(0, 100)
# plt.show()  

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(sequence13A_new, color='green', label='KRAS-146 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence13G_new, color='black', label='KRAS-146 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence13C_new, color='blue', label='KRAS-146 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence13T_new, color='red', label='KRAS-146 data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('mixed sequence')
# plt.ylabel('Value')
# plt.ylim(0, 100)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(sequence14A_new, color='green', label='ERBB2-310 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence14G_new, color='black', label='ERBB2-310 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence14C_new, color='blue', label='ERBB2-310 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence14T_new, color='red', label='ERBB2-310 data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('UP sequence')
# plt.legend()
# plt.tight_layout()
# plt.ylim(0, 100)
# plt.show()  

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(sequence15A_new, color='green', label='APC-876 data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence15G_new, color='black', label='APC-876 data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence15C_new, color='blue', label='APC-876 data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(sequence15T_new, color='red', label='APC-876 data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('mixed sequence')
# plt.ylabel('Value')
# plt.ylim(0, 100)
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(combined_sequence15A, color='green', label='mixed data A lines', marker='', linestyle='-', linewidth=2)
# plt.plot(combined_sequence15G, color='black', label='mixed data G lines', marker='', linestyle='-', linewidth=2)
# plt.plot(combined_sequence15C, color='blue', label='mixed data C lines', marker='', linestyle='-', linewidth=2)
# plt.plot(combined_sequence15T, color='red', label='mixed data T lines', marker='', linestyle='-', linewidth=2)
# # plt.title('mixed sequence')
# plt.ylim(0, 100)
# plt.legend()
# plt.tight_layout()
# plt.show()  import numpy as np
