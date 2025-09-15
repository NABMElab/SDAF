import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from dtw import *
from fastdtw import fastdtw
from itertools import combinations
from scipy.signal import find_peaks
import pandas as pd

#input length of CRC 7plex
length_NRAS_12 = 149
length_APC_892 = 145
length_TP53_175 = 144
length_APC_302 = 138
length_TP53_245 = 136
length_NRAS_61 = 127
length_TP53_280 = 116
# 设置全局参数
# plt.rcParams['axes.linewidth'] = 3  # 坐标轴线宽
# plt.rcParams['axes.labelsize'] = 20  # 坐标轴标签字体大小
# plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签字体粗细
# plt.rcParams['xtick.labelsize'] = 30  # x轴刻度标签字体大小
# plt.rcParams['ytick.labelsize'] = 30  # y轴刻度标签字体大小
# plt.rcParams['xtick.major.width'] = 3  # x轴刻度线宽
# plt.rcParams['ytick.major.width'] = 3  # y轴刻度线宽

# Calculate the cost matrix for sequences x and y, considering start and end indices
def dtw_cost_matrix(x, y, start_index_x=0, end_index_x=None, start_index_y=0, end_index_y=None):
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

# Calculate dynamic time warping distance between sequences x and y
def dtw_distance(x, y):
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

# Calculate the DTW distance between sequences x and y, considering start and end indices
def dtw_distance_with_indices(x, y, start_index_x=0, end_index_x=None, start_index_y=0, end_index_y=None):
    if end_index_x is None:
        end_index_x = len(x) - 1
    if end_index_y is None:
        end_index_y = len(y) - 1
    x_subseq = x[start_index_x:end_index_x + 1]
    y_subseq = y[start_index_y:end_index_y + 1]
    return dtw_distance(x_subseq, y_subseq)

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
    plt.show()

# Define DTW distance functions for each combination of channels and plex name
def dtw_distance_1(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=23, end_index_x=173, start_index_y=23, end_index_y=173)

# def dtw_distance_2(x, y):
#     return dtw_distance_with_indices(x, y, start_index_x=30, end_index_x=293, start_index_y=28, end_index_y=293)

# def dtw_distance_2(x, y):
#     return dtw_distance_with_indices(x, y, start_index_x=253, end_index_x=353, start_index_y=253, end_index_y=353)

def dtw_distance_2(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=76, end_index_x=226, start_index_y=76, end_index_y=226)

def dtw_distance_3(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=88, end_index_x=238, start_index_y=88, end_index_y=238)

def dtw_distance_4(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=160, end_index_x=310, start_index_y=160, end_index_y=310)

def dtw_distance_5(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=181, end_index_x=331, start_index_y=181, end_index_y=331)

def dtw_distance_6(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=294, end_index_x=444, start_index_y=294, end_index_y=444)

def dtw_distance_7(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=423, end_index_x=573, start_index_y=423, end_index_y=573)


#plot graph with different channel
def dtw_cost_matrix_1A(x, y):
    return dtw_cost_matrix(x, y, start_index_x=16, end_index_x=516, start_index_y=16, end_index_y=536)

def dtw_cost_matrix_2A(x, y):
    return dtw_cost_matrix(x, y, start_index_x=265, end_index_x=765, start_index_y=265, end_index_y=785)

def dtw_cost_matrix_3A(x, y):
    return dtw_cost_matrix(x, y, start_index_x=458, end_index_x=958, start_index_y=458, end_index_y=978)

def dtw_cost_matrix_4A(x, y):
    return dtw_cost_matrix(x, y, start_index_x=279, end_index_x=779, start_index_y=279, end_index_y=799)

def dtw_cost_matrix_5A(x, y):
    return dtw_cost_matrix(x, y, start_index_x=421, end_index_x=921, start_index_y=421, end_index_y=941)

def dtw_cost_matrix_1G(x, y):
    return dtw_cost_matrix(x, y, start_index_x=16, end_index_x=516, start_index_y=16, end_index_y=536)

def dtw_cost_matrix_2G(x, y):
    return dtw_cost_matrix(x, y, start_index_x=265, end_index_x=765, start_index_y=265, end_index_y=785)

def dtw_cost_matrix_3G(x, y):
    return dtw_cost_matrix(x, y, start_index_x=458, end_index_x=958, start_index_y=458, end_index_y=978)

def dtw_cost_matrix_4G(x, y):
    return dtw_cost_matrix(x, y, start_index_x=279, end_index_x=779, start_index_y=279, end_index_y=799)

def dtw_cost_matrix_5G(x, y):
    return dtw_cost_matrix(x, y, start_index_x=421, end_index_x=921, start_index_y=421, end_index_y=941)

def dtw_cost_matrix_1C(x, y):
    return dtw_cost_matrix(x, y, start_index_x=16, end_index_x=516, start_index_y=16, end_index_y=536)

def dtw_cost_matrix_2C(x, y):
    return dtw_cost_matrix(x, y, start_index_x=265, end_index_x=765, start_index_y=265, end_index_y=785)

def dtw_cost_matrix_3C(x, y):
    return dtw_cost_matrix(x, y, start_index_x=458, end_index_x=958, start_index_y=458, end_index_y=978)

def dtw_cost_matrix_4C(x, y):
    return dtw_cost_matrix(x, y, start_index_x=279, end_index_x=779, start_index_y=279, end_index_y=799)

def dtw_cost_matrix_5C(x, y):
    return dtw_cost_matrix(x, y, start_index_x=421, end_index_x=921, start_index_y=421, end_index_y=941)

def dtw_cost_matrix_1T(x, y):
    return dtw_cost_matrix(x, y, start_index_x=16, end_index_x=516, start_index_y=16, end_index_y=536)

def dtw_cost_matrix_2T(x, y):
    return dtw_cost_matrix(x, y, start_index_x=265, end_index_x=765, start_index_y=265, end_index_y=785)

def dtw_cost_matrix_3T(x, y):
    return dtw_cost_matrix(x, y, start_index_x=458, end_index_x=958, start_index_y=458, end_index_y=978)

def dtw_cost_matrix_4T(x, y):
    return dtw_cost_matrix(x, y, start_index_x=279, end_index_x=779, start_index_y=279, end_index_y=799)

def dtw_cost_matrix_5T(x, y):
    return dtw_cost_matrix(x, y, start_index_x=421, end_index_x=921, start_index_y=421, end_index_y=941)

# Load the multiplex Excel file
df = pd.read_excel("C:/Users/Jack Leong/Downloads/20250618_补充组织解谱/20250618_tube2e-7P-WB-0.5XHB-250114-S6.xlsx", header=None)

# sequence30A = df.iloc[1, 7:1106].tolist()
# sequence30G = df.iloc[2, 7:1106].tolist()
# sequence30C = df.iloc[3, 7:1106].tolist()
# sequence30T = df.iloc[4, 7:1106].tolist()

sequence30A = df.iloc[1, 0:1099].tolist()
sequence30G = df.iloc[2, 0:1099].tolist()
sequence30C = df.iloc[3, 0:1099].tolist()
sequence30T = df.iloc[4, 0:1099].tolist()
# Convert sequence30A to a numpy array
sequence30A = np.array(sequence30A)
sequence30G = np.array(sequence30G)
sequence30C = np.array(sequence30C)
sequence30T = np.array(sequence30T)

# sequence30A = np.concatenate((np.zeros(2),sequence30A))
# sequence30G = np.concatenate((np.zeros(2),sequence30G))
# sequence30C = np.concatenate((np.zeros(2),sequence30C))
# sequence30T = np.concatenate((np.zeros(2),sequence30T))

# Load the Single Plex Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/CRC 22plex 单谱excel/Hieffblue_CRC7plex_NRAS-12.xlsx", header=None)

sequence1A = df.iloc[1, 7:1106].tolist()
sequence1G = df.iloc[2, 7:1106].tolist()
sequence1C = df.iloc[3, 7:1106].tolist()
sequence1T = df.iloc[4, 7:1106].tolist()

# sequence30A = df.iloc[1, 0:1099].tolist()
# sequence30G = df.iloc[2, 0:1099].tolist()
# sequence30C = df.iloc[3, 0:1099].tolist()
# sequence30T = df.iloc[4, 0:1099].tolist()

sequence1A = np.array(sequence1A)
sequence1G = np.array(sequence1G)
sequence1C = np.array(sequence1C)
sequence1T = np.array(sequence1T)

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/CRC 22plex 单谱excel/Hieffblue_CRC7plex_APC-892.xlsx", header=None)

sequence2A = df.iloc[1, 0:1099].tolist()
sequence2G = df.iloc[2, 0:1099].tolist()
sequence2C = df.iloc[3, 0:1099].tolist()
sequence2T = df.iloc[4, 0:1099].tolist()

sequence2A = np.array(sequence2A)
sequence2G = np.array(sequence2G)
sequence2C = np.array(sequence2C)
sequence2T = np.array(sequence2T)

sequence2A = np.concatenate((np.zeros(1),sequence2A))
sequence2G = np.concatenate((np.zeros(1),sequence2G))
sequence2C = np.concatenate((np.zeros(1),sequence2C))
sequence2T = np.concatenate((np.zeros(1),sequence2T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/CRC 22plex 单谱excel/Hieffblue_CRC7plex_TP53-175.xlsx", header=None)

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
df = pd.read_excel("C:/Users/Jack Leong/Downloads/CRC 22plex 单谱excel/Hieffblue_CRC7plex_APC-302.xlsx", header=None)

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
df = pd.read_excel("C:/Users/Jack Leong/Downloads/CRC 22plex 单谱excel/Hieffblue_CRC7plex_TP53-245.xlsx", header=None)

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
df = pd.read_excel("C:/Users/Jack Leong/Downloads/CRC 22plex 单谱excel/Hieffblue_CRC7plex_NRAS-61.xlsx", header=None)

sequence6A = df.iloc[1, 8:1107].tolist()
sequence6G = df.iloc[2, 8:1107].tolist()
sequence6C = df.iloc[3, 8:1107].tolist()
sequence6T = df.iloc[4, 8:1107].tolist()

sequence6A = np.array(sequence6A)
sequence6G = np.array(sequence6G)
sequence6C = np.array(sequence6C)
sequence6T = np.array(sequence6T)

# sequence6A = np.concatenate((np.zeros(1),sequence6A))
# sequence6G = np.concatenate((np.zeros(1),sequence6G))
# sequence6C = np.concatenate((np.zeros(1),sequence6C))
# sequence6T = np.concatenate((np.zeros(1),sequence6T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/CRC 22plex 单谱excel/Hieffblue_CRC7plex_TP53-280.xlsx", header=None)

sequence7A = df.iloc[1, 0:1099].tolist()
sequence7G = df.iloc[2, 0:1099].tolist()
sequence7C = df.iloc[3, 0:1099].tolist()
sequence7T = df.iloc[4, 0:1099].tolist()

sequence7A = np.array(sequence7A)
sequence7G = np.array(sequence7G)
sequence7C = np.array(sequence7C)
sequence7T = np.array(sequence7T)

# sequence7A = np.concatenate((np.zeros(1),sequence7A))
# sequence7G = np.concatenate((np.zeros(1),sequence7G))
# sequence7C = np.concatenate((np.zeros(1),sequence7C))
# sequence7T = np.concatenate((np.zeros(1),sequence7T))

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
sequence30A = sequence30A/max_sequence30*100*1.5
sequence30G = sequence30G/max_sequence30*100*1.5
sequence30C = sequence30C/max_sequence30*100*1.5
sequence30T = sequence30T/max_sequence30*100*1.5

# Sort peaks above threshold value based on their peak position
def find_and_sort_peaks(sequence_data, segment_start, segment_end, height_threshold):
    peaks_all = []
    for sequence, name in sequence_data:
        peaks, _ = find_peaks(sequence[segment_start:segment_end], height=height_threshold)
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
sequencesA = [sequence1A, sequence2A, sequence3A, sequence4A, sequence5A, sequence6A, sequence7A, sequence0A]  # Define all sequences for channel A
sequencesG = [sequence1G, sequence2G, sequence3G, sequence4G, sequence5G, sequence6G, sequence7G, sequence0G]  # Define all sequences for channel G
sequencesC = [sequence1C, sequence2C, sequence3C, sequence4C, sequence5C, sequence6C, sequence7C, sequence0C]  # Define all sequences for channel C
sequencesT = [sequence1T, sequence2T, sequence3T, sequence4T, sequence5T, sequence6T, sequence7T, sequence0T]  # Define all sequences for channel T

# Set parameters
segment_start = 20
segment_end = 400
height_threshold = 15

sequences = [
    [(sequence1A, 'A'), (sequence1G, 'G'), (sequence1C, 'C'), (sequence1T, 'T')],
    [(sequence2A, 'A'), (sequence2G, 'G'), (sequence2C, 'C'), (sequence2T, 'T')],
    [(sequence3A, 'A'), (sequence3G, 'G'), (sequence3C, 'C'), (sequence3T, 'T')],
    [(sequence4A, 'A'), (sequence4G, 'G'), (sequence4C, 'C'), (sequence4T, 'T')],
    [(sequence5A, 'A'), (sequence5G, 'G'), (sequence5C, 'C'), (sequence5T, 'T')],
    [(sequence6A, 'A'), (sequence6G, 'G'), (sequence6C, 'C'), (sequence6T, 'T')],
    [(sequence7A, 'A'), (sequence7G, 'G'), (sequence7C, 'C'), (sequence7T, 'T')], 
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

sequence_set_8 = {'bases':sorted_bases_all[7], 'peaks': sorted_peak_positions_all[7]}

# Sequence sets to compare
sequence_sets = [
    {'bases': sorted_bases_all[0], 'peaks': sorted_peak_positions_all[0]},  # Sequence Set 1
    {'bases': sorted_bases_all[1], 'peaks': sorted_peak_positions_all[1]},   # Sequence Set 2
    {'bases': sorted_bases_all[2], 'peaks': sorted_peak_positions_all[2]},  # Sequence Set 3
    {'bases': sorted_bases_all[3], 'peaks': sorted_peak_positions_all[3]},  # Sequence Set 4
    {'bases': sorted_bases_all[4], 'peaks': sorted_peak_positions_all[4]},  # Sequence Set 5
    {'bases': sorted_bases_all[5], 'peaks': sorted_peak_positions_all[5]},  # Sequence Set 6
    {'bases': sorted_bases_all[6], 'peaks': sorted_peak_positions_all[6]},  # Sequence Set 7

]

# Calculate similarities to sequence sets
similarities = []
for i, seq_set in enumerate(sequence_sets[:7]):  # Only compare with Sequence Set 1 to 7
    matched_peaks = compare_peak_bases_with_tolerance(sequence_set_8, seq_set)
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
    length_ref= length_NRAS_12
    ref_sequenceA = sequence1A
    ref_sequenceG = sequence1G
    ref_sequenceC = sequence1C
    ref_sequenceT = sequence1T

elif most_similar_sequence['set'] == 2:
    length_ref= length_APC_892
    ref_sequenceA = sequence2A
    ref_sequenceG = sequence2G
    ref_sequenceC = sequence2C
    ref_sequenceT = sequence2T

elif most_similar_sequence['set'] == 3:
    length_ref= length_TP53_175
    ref_sequenceA = sequence3A
    ref_sequenceG = sequence3G
    ref_sequenceC = sequence3C
    ref_sequenceT = sequence3T

elif most_similar_sequence['set'] == 4:
    length_ref= length_APC_302
    ref_sequenceA = sequence4A
    ref_sequenceG = sequence4G
    ref_sequenceC = sequence4C
    ref_sequenceT = sequence4T
elif most_similar_sequence['set'] == 5:
    length_ref= length_TP53_245
    ref_sequenceA = sequence5A
    ref_sequenceG = sequence5G
    ref_sequenceC = sequence5C
    ref_sequenceT = sequence5T
elif most_similar_sequence['set'] == 6:
    length_ref= length_NRAS_61
    ref_sequenceA = sequence6A
    ref_sequenceG = sequence6G
    ref_sequenceC = sequence6C
    ref_sequenceT = sequence6T
elif most_similar_sequence['set'] == 7:
    length_ref= length_TP53_280
    ref_sequenceA = sequence7A
    ref_sequenceG = sequence7G
    ref_sequenceC = sequence7C
    ref_sequenceT = sequence7T
else:
    length_ref = None  # Handle the case where no ref_sequence is chosen

print(ref_sequenceA)
print(length_ref)

if length_ref == None:
    print("This is a chaos spectrum, no mutation in this sample")

# length_ref= length_APC_892
# ref_sequenceA = sequence2A
# ref_sequenceG = sequence2G
# ref_sequenceC = sequence2C
# ref_sequenceT = sequence2T
# print(length_ref)

# Set parameters
segment_start = 0
segment_end = 500
height_threshold = 10

sequences = [
    [(ref_sequenceA, 'ref_sequenceA'), (ref_sequenceG, 'ref_sequenceG'), (ref_sequenceC, 'ref_sequenceC'), (ref_sequenceT, 'ref_sequenceT')],
    [(sequence1A, 'sequence1A'), (sequence1G, 'sequence1G'), (sequence1C, 'sequence1C'), (sequence1T, 'sequence1T')],
    [(sequence2A, 'sequence2A'), (sequence2G, 'sequence2G'), (sequence2C, 'sequence2C'), (sequence2T, 'sequence2T')],
    [(sequence3A, 'sequence3A'), (sequence3G, 'sequence3G'), (sequence3C, 'sequence3C'), (sequence3T, 'sequence3T')],
    [(sequence4A, 'sequence4A'), (sequence4G, 'sequence4G'), (sequence4C, 'sequence4C'), (sequence4T, 'sequence4T')],
    [(sequence5A, 'sequence5A'), (sequence5G, 'sequence5G'), (sequence5C, 'sequence5C'), (sequence5T, 'sequence5T')],
    [(sequence6A, 'sequence6A'), (sequence6G, 'sequence6G'), (sequence6C, 'sequence6C'), (sequence6T, 'sequence6T')],
    [(sequence7A, 'sequence7A'), (sequence7G, 'sequence7G'), (sequence7C, 'sequence7C'), (sequence7T, 'sequence7T')]
]

# Stores the sequencing peak positions for all sequences
sorted_peak_positions_all = []
for i, seq_set in enumerate(sequences):
    sorted_peaks = find_and_sort_peaks(seq_set, segment_start, segment_end, height_threshold)
    print(f"\nPeaks for Sequence Set {i+1}:")
    print_peaks(sorted_peaks)
    sorted_peak_positions_all.append(get_sorted_peak_positions(sorted_peaks))

length_NRAS_12 = 149
length_APC_892 = 145
length_TP53_175 = 144
length_APC_302 = 138
length_TP53_245 = 136
length_NRAS_61 = 127
length_TP53_280 = 116

# Extract different zero padding values
zerospadding_NRAS_12 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_NRAS_12,0)+1] - sorted_peak_positions_all[1][1],0)
zerospadding_APC_892 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_APC_892,0)+1] - sorted_peak_positions_all[2][1],0)
zerospadding_TP53_175 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_TP53_175,0)+1] - sorted_peak_positions_all[3][1],0)
zerospadding_APC_302 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_APC_302,0)+1] - sorted_peak_positions_all[4][1],0)
zerospadding_TP53_245 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_TP53_245,0)+1] - sorted_peak_positions_all[5][1],0)
zerospadding_NRAS_61 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_NRAS_61,0)+1] - sorted_peak_positions_all[6][1],0)
zerospadding_TP53_280 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_TP53_280,0)+1] - sorted_peak_positions_all[7][1],0)

print(zerospadding_NRAS_12)
print(zerospadding_APC_892)
print(zerospadding_TP53_175)
print(zerospadding_APC_302)
print(zerospadding_TP53_245)
print(zerospadding_NRAS_61)
print(zerospadding_TP53_280)

sequence1A = np.concatenate((np.zeros(np.abs(zerospadding_NRAS_12)), sequence1A))
sequence1G = np.concatenate((np.zeros(np.abs(zerospadding_NRAS_12)), sequence1G))
sequence1C = np.concatenate((np.zeros(np.abs(zerospadding_NRAS_12)), sequence1C))
sequence1T = np.concatenate((np.zeros(np.abs(zerospadding_NRAS_12)), sequence1T))
sequence2A = np.concatenate((np.zeros(zerospadding_APC_892), sequence2A))
sequence2G = np.concatenate((np.zeros(zerospadding_APC_892), sequence2G))
sequence2C = np.concatenate((np.zeros(zerospadding_APC_892), sequence2C))
sequence2T = np.concatenate((np.zeros(zerospadding_APC_892), sequence2T))
sequence3A = np.concatenate((np.zeros(zerospadding_TP53_175), sequence3A))
sequence3G = np.concatenate((np.zeros(zerospadding_TP53_175), sequence3G))
sequence3C = np.concatenate((np.zeros(zerospadding_TP53_175), sequence3C))
sequence3T = np.concatenate((np.zeros(zerospadding_TP53_175), sequence3T))
sequence4A = np.concatenate((np.zeros(zerospadding_APC_302), sequence4A))
sequence4G = np.concatenate((np.zeros(zerospadding_APC_302), sequence4G))
sequence4C = np.concatenate((np.zeros(zerospadding_APC_302), sequence4C))
sequence4T = np.concatenate((np.zeros(zerospadding_APC_302), sequence4T))
sequence5A = np.concatenate((np.zeros(zerospadding_TP53_245), sequence5A))
sequence5G = np.concatenate((np.zeros(zerospadding_TP53_245), sequence5G))
sequence5C = np.concatenate((np.zeros(zerospadding_TP53_245), sequence5C))
sequence5T = np.concatenate((np.zeros(zerospadding_TP53_245), sequence5T))
sequence6A = np.concatenate((np.zeros(zerospadding_NRAS_61), sequence6A))
sequence6G = np.concatenate((np.zeros(zerospadding_NRAS_61), sequence6G))
sequence6C = np.concatenate((np.zeros(zerospadding_NRAS_61), sequence6C))
sequence6T = np.concatenate((np.zeros(zerospadding_NRAS_61), sequence6T))
sequence7A = np.concatenate((np.zeros(zerospadding_TP53_280), sequence7A))
sequence7G = np.concatenate((np.zeros(zerospadding_TP53_280), sequence7G))
sequence7C = np.concatenate((np.zeros(zerospadding_TP53_280), sequence7C))
sequence7T = np.concatenate((np.zeros(zerospadding_TP53_280), sequence7T))

max_sequence1 = max(max(sequence1A, default=0),max(sequence1G, default=0), max(sequence1C, default=0),max(sequence1T, default=0))
max_sequence2 = max(max(sequence2A, default=0),max(sequence2G, default=0), max(sequence2C, default=0),max(sequence2T, default=0))
max_sequence3 = max(max(sequence3A, default=0),max(sequence3G, default=0), max(sequence3C, default=0),max(sequence3T, default=0))
max_sequence4 = max(max(sequence4A, default=0),max(sequence4G, default=0), max(sequence4C, default=0),max(sequence4T, default=0))
max_sequence5 = max(max(sequence5A, default=0),max(sequence5G, default=0), max(sequence5C, default=0),max(sequence5T, default=0))
max_sequence6 = max(max(sequence6A, default=0),max(sequence6G, default=0), max(sequence6C, default=0),max(sequence6T, default=0))
max_sequence7 = max(max(sequence7A, default=0),max(sequence7G, default=0), max(sequence7C, default=0),max(sequence7T, default=0))
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
sequence30A = sequence30A/max_sequence30*100
sequence30G = sequence30G/max_sequence30*100
sequence30C = sequence30C/max_sequence30*100
sequence30T = sequence30T/max_sequence30*100

ref_sequenceA = ref_sequenceA/max_sequence_ref*100
ref_sequenceG = ref_sequenceG/max_sequence_ref*100
ref_sequenceC = ref_sequenceC/max_sequence_ref*100
ref_sequenceT = ref_sequenceT/max_sequence_ref*100

# Find the length of the shortest sequence
min_length = min(len(sequence1A), len(sequence2A),len(sequence3A),len(sequence4A),len(sequence5A),len(sequence6A),len(sequence7A))

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
height_threshold = 10
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

# Apply this shift to all sequences in channels A, G, and C
sequence30A = align_sequences_based_on_shift(sequence30A, shift)
sequence30G = align_sequences_based_on_shift(sequence30G, shift)
sequence30C = align_sequences_based_on_shift(sequence30C, shift)
sequence30T = align_sequences_based_on_shift(sequence30T, shift)

sequence30A = sequence30A[:min_length] 
sequence30G = sequence30G[:min_length] 
sequence30C = sequence30C[:min_length]
sequence30T = sequence30T[:min_length]

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

length_NRAS_12 = 149
length_APC_892 = 145
length_TP53_175 = 144
length_APC_302 = 138
length_TP53_245 = 136
length_NRAS_61 = 127
length_TP53_280 = 116
print(sorted_peak_positions_all[0][length_ref-length_NRAS_12]-5)
print(sorted_peak_positions_all[0][length_ref-length_APC_892]-5)
print(sorted_peak_positions_all[0][length_ref-length_TP53_175]-5)
print(sorted_peak_positions_all[0][length_ref-length_APC_302]-5)
print(sorted_peak_positions_all[0][length_ref-length_TP53_245]-5)
print(sorted_peak_positions_all[0][length_ref-length_NRAS_61]-5)
print(sorted_peak_positions_all[0][length_ref-length_TP53_280]-5)

# Define DTW distance functions for each combination of channels and plex name
def dtw_distance_ref(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=15, end_index_x=165, start_index_y=15, end_index_y=165)

def dtw_distance_1(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_NRAS_12]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_NRAS_12]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_NRAS_12]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_NRAS_12]+70)

def dtw_distance_2(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_APC_892]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_APC_892]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_APC_892]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_APC_892]+70)

def dtw_distance_3(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_TP53_175]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_TP53_175]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_TP53_175]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_TP53_175]+70)

def dtw_distance_4(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_APC_302]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_APC_302]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_APC_302]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_APC_302]+70)

def dtw_distance_5(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_TP53_245]-5, end_index_x=sorted_peak_positions_all[0][length_ref-length_TP53_245]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_TP53_245]-5, end_index_y=sorted_peak_positions_all[0][length_ref-length_TP53_245]+70)

def dtw_distance_6(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_NRAS_61]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_NRAS_61]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_NRAS_61]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_NRAS_61]+70)

def dtw_distance_7(x, y):
    return dtw_distance_with_indices(x, y, start_index_x=sorted_peak_positions_all[0][length_ref-length_TP53_280]-10, end_index_x=sorted_peak_positions_all[0][length_ref-length_TP53_280]+70, start_index_y=sorted_peak_positions_all[0][length_ref-length_TP53_280]-10, end_index_y=sorted_peak_positions_all[0][length_ref-length_TP53_280]+70)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_NRAS_12]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] > sequence[i + 1]:
            peaks.append((i, sequence[i]))
            if len(peaks) == 1:
                return peaks[0][0]  # Return the index of the second peak
    return start_point  # Return the starting point if no second peak is found

# Function to align sequences based on the peak alignment of a reference sequence
def align_sequences_based_on_shift(ref_sequence, shift):
    if shift > 0:
        aligned_seq = np.pad(ref_sequence, (shift, 0), 'constant')[:len(ref_sequence)]
    else:
        aligned_seq = np.pad(ref_sequence[-shift:], (0, -shift), 'constant')
    return aligned_seq

# Find the first peaks in sequence1T and sequence3T
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

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_APC_892]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] > sequence[i + 1]:
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

peak2T = find_peaks(sequence2T)
peak30T = find_peaks(sequence30T)

print(peak2T)
print(peak30T)

shift = peak30T - peak2T

# Apply this shift to all sequences in channels A, G, C and T
sequence2A = align_sequences_based_on_shift(sequence2A, shift)
sequence2G = align_sequences_based_on_shift(sequence2G, shift)
sequence2C = align_sequences_based_on_shift(sequence2C, shift)
sequence2T = align_sequences_based_on_shift(sequence2T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_TP53_175]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] > sequence[i + 1]:
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

peak3T = find_peaks(sequence3T)
peak30T = find_peaks(sequence30T)

print(peak3T)
print(peak30T)

shift = peak30T - peak3T

# Apply this shift to all sequences in channels A, G, C and T
sequence3A = align_sequences_based_on_shift(sequence3A, shift)
sequence3G = align_sequences_based_on_shift(sequence3G, shift)
sequence3C = align_sequences_based_on_shift(sequence3C, shift)
sequence3T = align_sequences_based_on_shift(sequence3T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_APC_302]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] > sequence[i + 1]:
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

peak4T = find_peaks(sequence4T)
peak30T = find_peaks(sequence30T)

print(peak4T)
print(peak30T)

shift = peak30T - peak4T

# Apply this shift to all sequences in channels A, G, C and T
sequence4A = align_sequences_based_on_shift(sequence4A, shift)
sequence4G = align_sequences_based_on_shift(sequence4G, shift)
sequence4C = align_sequences_based_on_shift(sequence4C, shift)
sequence4T = align_sequences_based_on_shift(sequence4T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_TP53_245]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] > sequence[i + 1]:
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

peak5T = find_peaks(sequence5T)
peak30T = find_peaks(sequence30T)

shift = peak30T - peak5T

print(peak5T)
print(peak30T)
# Apply this shift to all sequences in channels A, G, C and T
sequence5A = align_sequences_based_on_shift(sequence5A, shift)
sequence5G = align_sequences_based_on_shift(sequence5G, shift)
sequence5C = align_sequences_based_on_shift(sequence5C, shift)
sequence5T = align_sequences_based_on_shift(sequence5T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_NRAS_61]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] > sequence[i + 1]:
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

peak6T = find_peaks(sequence6T)
peak30T = find_peaks(sequence30T)

shift = peak30T - peak6T

print(peak6T)
print(peak30T)
# Apply this shift to all sequences in channels A, G, C and T
sequence6A = align_sequences_based_on_shift(sequence6A, shift)
sequence6G = align_sequences_based_on_shift(sequence6G, shift)
sequence6C = align_sequences_based_on_shift(sequence6C, shift)
sequence6T = align_sequences_based_on_shift(sequence6T, shift)

# Function to find the first peak with a specified threshold and starting point
def find_peaks(sequence, threshold=10, start_point=sorted_peak_positions_all[0][length_ref-length_TP53_280]-5):
    peaks = []
    for i in range(start_point, len(sequence) - 1):
        if sequence[i] > threshold and sequence[i] > sequence[i - 1] and sequence[i] > sequence[i + 1]:
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

peak7T = find_peaks(sequence7T)
peak30T = find_peaks(sequence30T)

shift = peak30T - peak7T

print(peak7T)
print(peak30T)
# Apply this shift to all sequences in channels A, G, C and T
sequence7A = align_sequences_based_on_shift(sequence7A, shift)
sequence7G = align_sequences_based_on_shift(sequence7G, shift)
sequence7C = align_sequences_based_on_shift(sequence7C, shift)
sequence7T = align_sequences_based_on_shift(sequence7T, shift)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(sequence5A, color='green', label='mixed data A lines', marker='', linestyle='-')
plt.plot(sequence5G, color='black', label='mixed data G lines', marker='', linestyle='-')
plt.plot(sequence5C, color='blue', label='mixed data C lines', marker='', linestyle='-')
plt.plot(sequence5T, color='red', label='mixed data T lines', marker='', linestyle='-')
plt.ylim(0, 100)
plt.ylabel('Value')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(sequence6A, color='green', label='mixed data A lines', marker='', linestyle='-')
plt.plot(sequence6G, color='black', label='mixed data G lines', marker='', linestyle='-')
plt.plot(sequence6C, color='blue', label='mixed data C lines', marker='', linestyle='-')
plt.plot(sequence6T, color='red', label='mixed data T lines', marker='', linestyle='-')
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

# Define the sequences for each channel
sequencesA = [sequence1A, sequence2A,sequence3A, sequence4A, sequence5A,sequence6A,sequence7A,sequence0A]  # Define all sequences for channel A
sequencesG = [sequence1G, sequence2G,sequence3G, sequence4G, sequence5G,sequence6G,sequence7G,sequence0G]  # Define all sequences for channel G
sequencesC = [sequence1C, sequence2C,sequence3C, sequence4C, sequence5C,sequence6C,sequence7C,sequence0C]  # Define all sequences for channel C
sequencesT = [sequence1T, sequence2T,sequence3T, sequence4T, sequence5T,sequence6T,sequence7T,sequence0T]  # Define all sequences for channel T

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

NRAS_12 = 0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence1) >= 2:  # At least two channels with sequence 1
    valid_n_count = sum(1 for _, n in channels_with_sequence1 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        NRAS_12 = 1  # Store NRAS_12 = 1

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")

if length_ref != length_NRAS_12:
    NRAS_12 = 0
# Print the value of NRAS_12
print(f"Value of NRAS_12: {NRAS_12}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = n * sequence1A
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
    combined_sequence = n * sequence1G 
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
    combined_sequence = n * sequence1C
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
    combined_sequence = n * sequence1T
    distance = dtw_distance_1(combined_sequence, sequence30T)
    if distance < min_distance:
        min_distance = distance
        best_n = n

# Store the result
best_results_T = best_n
print("Best n for sequence1T:", best_results_T)

# Calculate NRAS_12 channel A weight
if NRAS_12==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif NRAS_12==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence1A = sequence1A*results_A
sequence1A_new=sequence1A*results_A

# Calculate NRAS_12 channel G weight
if NRAS_12==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif NRAS_12==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence1G = sequence1G*results_G
sequence1G_new=sequence1G*results_G

# Calculate NRAS_12 channel C weight
if NRAS_12==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif NRAS_12==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence1C = sequence1C*results_C
sequence1C_new=sequence1C*results_C

# Calculate NRAS_12 channel T weight
if NRAS_12==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif NRAS_12==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence1T = sequence1T*results_T
sequence1T_new=sequence1T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence1A, sequence2A,sequence3A, sequence4A, sequence5A,sequence6A,sequence7A]  # Define all sequences for channel A
sequencesG = [combined_sequence1G, sequence2G,sequence3G, sequence4G, sequence5G,sequence6G,sequence7G]  # Define all sequences for channel G
sequencesC = [combined_sequence1C, sequence2C,sequence3C, sequence4C, sequence5C,sequence6C,sequence7C]  # Define all sequences for channel C
sequencesT = [combined_sequence1T, sequence2T,sequence3T, sequence4T, sequence5T,sequence6T,sequence7T]  # Define all sequences for channel T

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
APC_892=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence2) >= 2:  # At least two channels with sequence 2
    valid_n_count = sum(1 for _, n in channels_with_sequence2 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        APC_892 = 1  # Store APC_892 = 1

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")

print(f"Value of APC_892: {APC_892}")

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

# Calculate APC_892 channel A weight
if APC_892==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif APC_892==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence2A = combined_sequence1A+sequence2A*results_A
sequence2A_new=sequence2A*results_A

# Calculate APC_892 channel G weight
if APC_892==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif APC_892==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence2G = combined_sequence1G+sequence2G*results_G
sequence2G_new=sequence2G*results_G

# Calculate APC_892 channel C weight
if APC_892==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif APC_892==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence2C = combined_sequence1C+sequence2C*results_C
sequence2C_new=sequence2C*results_C

# Calculate APC_892 channel T weight
if APC_892==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif APC_892==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence2T = combined_sequence1T+sequence2T*results_T
sequence2T_new=sequence2T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence2A, sequence2A,sequence3A, sequence4A, sequence5A,sequence6A,sequence7A]  # Define all sequences for channel A
sequencesG = [combined_sequence2G, sequence2G,sequence3G, sequence4G, sequence5G,sequence6G,sequence7G]  # Define all sequences for channel G
sequencesC = [combined_sequence2C, sequence2C,sequence3C, sequence4C, sequence5C,sequence6C,sequence7C]  # Define all sequences for channel C
sequencesT = [combined_sequence2T, sequence2T,sequence3T, sequence4T, sequence5T,sequence6T,sequence7T]  # Define all sequences for channel T

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
            distance = dtw_distance_3(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 3 as the best sequence number
channels_with_sequence3 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 3]
TP53_175=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence3) >= 2:  # At least two channels with sequence 3
    valid_n_count = sum(1 for _, n in channels_with_sequence3 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        TP53_175 = 1  

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")

print(f"Value of TP53_175: {TP53_175}")

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

# Calculate TP53_175 channel A weight
if TP53_175==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif TP53_175==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence3A = combined_sequence2A+sequence3A*results_A
sequence3A_new=sequence3A*results_A

# Calculate TP53_175 channel G weight
if TP53_175==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif TP53_175==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence3G = combined_sequence2G+sequence3G*results_G
sequence3G_new=sequence3G*results_G

# Calculate TP53_175 channel C weight
if TP53_175==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif TP53_175==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence3C = combined_sequence2C+sequence3C*results_C
sequence3C_new=sequence3C*results_C

# Calculate TP53_175 channel T weight
if TP53_175==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif TP53_175==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence3T = combined_sequence2T+sequence3T*results_T
sequence3T_new=sequence3T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence3A, sequence2A,sequence3A, sequence4A, sequence5A,sequence6A,sequence7A]  # Define all sequences for channel A
sequencesG = [combined_sequence3G, sequence2G,sequence3G, sequence4G, sequence5G,sequence6G,sequence7G]  # Define all sequences for channel G
sequencesC = [combined_sequence3C, sequence2C,sequence3C, sequence4C, sequence5C,sequence6C,sequence7C]  # Define all sequences for channel C
sequencesT = [combined_sequence3T, sequence2T,sequence3T, sequence4T, sequence5T,sequence6T,sequence7T]  # Define all sequences for channel T

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
            distance = dtw_distance_4(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 4 as the best sequence number
channels_with_sequence4 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 4]
APC_302=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence4) >= 2:  # At least two channels with sequence 4
    valid_n_count = sum(1 for _, n in channels_with_sequence4 if 1.5 >= n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        APC_302 = 1  # Store APC_302= 1

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")

print(f"Value of APC_302: {APC_302}")

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

# Calculate APC_302 channel A weight
if APC_302==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif APC_302==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence4A = combined_sequence3A+sequence4A*results_A
sequence4A_new=sequence4A*results_A

# Calculate APC_302 channel G weight
if APC_302==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif APC_302==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence4G = combined_sequence3G+sequence4G*results_G
sequence4G_new=sequence4G*results_G

# Calculate APC_302 channel C weight
if APC_302==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif APC_302==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence4C = combined_sequence3C+sequence4C*results_C
sequence4C_new=sequence4C*results_C

# Calculate APC_302 channel T weight
if APC_302==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif APC_302==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence4T = combined_sequence3T+sequence4T*results_T
sequence4T_new=sequence4T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence4A, sequence2A,sequence3A, sequence4A, sequence5A,sequence6A,sequence7A]  # Define all sequences for channel A
sequencesG = [combined_sequence4G, sequence2G,sequence3G, sequence4G, sequence5G,sequence6G,sequence7G]  # Define all sequences for channel G
sequencesC = [combined_sequence4C, sequence2C,sequence3C, sequence4C, sequence5C,sequence6C,sequence7C]  # Define all sequences for channel C
sequencesT = [combined_sequence4T, sequence2T,sequence3T, sequence4T, sequence5T,sequence6T,sequence7T]  # Define all sequences for channel T

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
            distance = dtw_distance_5(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 5 as the best sequence number
channels_with_sequence5 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 5]
TP53_245=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence5) >= 2:  # At least two channels with sequence 5
    valid_n_count = sum(1 for _, n in channels_with_sequence5 if n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        TP53_245 = 1

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")

print(f"Value of TP53_245: {TP53_245}")

# Find the best value of n
min_distance = float('inf')
best_n = None
for n in np.linspace(0, 2, 21):
    combined_sequence = combined_sequence4A + n * sequence5A
    distance = dtw_distance_5(combined_sequence, target_sequenceA)
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

# Calculate TP53_245 channel A weight
if TP53_245==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif TP53_245==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence5A = combined_sequence4A+sequence5A*results_A
sequence5A_new=sequence5A*results_A

# Calculate TP53_245 channel G weight
if TP53_245==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif TP53_245==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence5G = combined_sequence4G+sequence5G*results_G
sequence5G_new=sequence5G*results_G

# Calculate TP53_245 channel C weight
if TP53_245==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif TP53_245==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3]or [0.2]), 0.2)
else:
    results_C=0

combined_sequence5C = combined_sequence4C+sequence5C*results_C
sequence5C_new=sequence5C*results_C

# Calculate TP53_245 channel T weight
if TP53_245==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif TP53_245==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence5T = combined_sequence4T+sequence5T*results_T
sequence5T_new=sequence5T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence5A, sequence2A,sequence3A, sequence4A, sequence5A,sequence6A,sequence7A]  # Define all sequences for channel A
sequencesG = [combined_sequence5G, sequence2G,sequence3G, sequence4G, sequence5G,sequence6G,sequence7G]  # Define all sequences for channel G
sequencesC = [combined_sequence5C, sequence2C,sequence3C, sequence4C, sequence5C,sequence6C,sequence7C]  # Define all sequences for channel C
sequencesT = [combined_sequence5T, sequence2T,sequence3T, sequence4T, sequence5T,sequence6T,sequence7T]  # Define all sequences for channel T

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
            distance = dtw_distance_6(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 6 as the best sequence number
channels_with_sequence6 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 6]
NRAS_61=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence6) >= 2:  # At least two channels with sequence 6
    valid_n_count = sum(1 for _, n in channels_with_sequence6 if n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        NRAS_61 = 1

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")

print(f"Value of NRAS_61: {NRAS_61}")

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
best_results_A =best_n
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
best_results_C = best_n
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

# Calculate NRAS_61 channel A weight
if NRAS_61==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif NRAS_61==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence6A = combined_sequence5A+sequence6A*results_A
sequence6A_new=sequence6A*results_A

# Calculate NRAS_61 channel G weight
if NRAS_61==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif NRAS_61==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence6G = combined_sequence5G+sequence6G*results_G
sequence6G_new=sequence6G*results_G

# Calculate NRAS_61 channel C weight
if NRAS_61==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif NRAS_61==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
    results_C=0

combined_sequence6C = combined_sequence5C+sequence6C*results_C
sequence6C_new=sequence6C*results_C

# Calculate NRAS_61 channel T weight
if NRAS_61==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif NRAS_61==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence6T = combined_sequence5T+sequence6T*results_T
sequence6T_new=sequence6T*results_T

# Define the sequences for each channel
sequencesA = [combined_sequence6A, sequence2A,sequence3A, sequence4A, sequence5A,sequence6A,sequence7A]  # Define all sequences for channel A
sequencesG = [combined_sequence6G, sequence2G,sequence3G, sequence4G, sequence5G,sequence6G,sequence7G]  # Define all sequences for channel G
sequencesC = [combined_sequence6C, sequence2C,sequence3C, sequence4C, sequence5C,sequence6C,sequence7C]  # Define all sequences for channel C
sequencesT = [combined_sequence6T, sequence2T,sequence3T, sequence4T, sequence5T,sequence6T,sequence7T]  # Define all sequences for channel T

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
            distance = dtw_distance_7(combined_sequence, target_sequence)
            if distance < min_distance:
                min_distance = distance
                best_n = n
                best_sequence = seq
                best_sequence_number = i
    
    best_results[channel] = {"best_n": best_n, "best_sequence": best_sequence, "min_distance": min_distance, "best_sequence_number": best_sequence_number}

# Check if at least two channels have sequence 7 as the best sequence number
channels_with_sequence7 = [(channel, result['best_n']) for channel, result in best_results.items() if result['best_sequence_number'] == 7]
TP53_280=0
# Check if at least two of them have n >= 0.2
if len(channels_with_sequence7) >= 2:  # At least two channels with sequence 7
    valid_n_count = sum(1 for _, n in channels_with_sequence7 if n >= 0.2)
    if valid_n_count >= 2:  # At least two of them have n >= 0.2
        TP53_280 = 1

# Print the results for each channel
for channel, result in best_results.items():
    print(f"Best value of n for channel {channel}: {result['best_n']}")
    print(f"Best sequence for channel {channel}: sequence{result['best_sequence_number']}{channel}")
    print(f"Minimum DTW distance for channel {channel}: {result['min_distance']}")

print(f"Value of TP53_280: {TP53_280}")

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
best_results_A =best_n
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
best_results_C = best_n
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
best_results_T =  best_n
print("Best n for sequence7T:", best_results_T)

# Calculate TP53_280 channel A weight
if TP53_280==1 and best_results_A<1.6 and best_results_A< 1.5*max(best_results_G,best_results_C,best_results_T) and 1.5*best_results_A> min(best_results_G,best_results_C,best_results_T):
    results_A=best_results_A
elif TP53_280==1 and (best_results_A>=1.6 or best_results_A>= 1.5*max(best_results_G,best_results_C,best_results_T) or 1.5*best_results_A<= min(best_results_G,best_results_C,best_results_T)):
    results_A=max(min([x for x in [best_results_G, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_A=0

combined_sequence7A = combined_sequence6A+sequence7A*results_A
sequence7A_new=sequence7A*results_A

# Calculate TP53_280 channel G weight
if TP53_280==1 and best_results_G<1.6 and best_results_G< 1.5*max(best_results_A,best_results_C,best_results_T) and 1.5*best_results_G> min(best_results_A,best_results_C,best_results_T):
    results_G=best_results_G
elif TP53_280==1 and (best_results_G>=1.6 or best_results_G>= 1.5*max(best_results_A,best_results_C,best_results_T) or 1.5*best_results_G<= min(best_results_A,best_results_C,best_results_T)):
    results_G=max(min([x for x in [best_results_A, best_results_C, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_G=0

combined_sequence7G = combined_sequence6G+sequence7G*results_G
sequence7G_new=sequence7G*results_G

# Calculate TP53_280 channel C weight
if TP53_280==1 and best_results_C<1.6 and best_results_C< 1.5*max(best_results_A,best_results_G,best_results_T) and 1.5*best_results_C> min(best_results_A,best_results_G,best_results_T):
    results_C=best_results_C
elif TP53_280==1 and (best_results_C>=1.6 or best_results_C>= 1.5*max(best_results_A,best_results_G,best_results_T) or 1.5*best_results_C<= min(best_results_A,best_results_G,best_results_T)):
    results_C=max(min([x for x in [best_results_A, best_results_G, best_results_T] if x >= 0.3] or [0.2]), 0.2)
else:
    results_C=0

combined_sequence7C = combined_sequence6C+sequence7C*results_C
sequence7C_new=sequence7C*results_C

# Calculate TP53_280 channel T weight
if TP53_280==1 and best_results_T<1.6 and best_results_T< 1.5*max(best_results_A,best_results_G,best_results_C) and 1.5*best_results_T> min(best_results_A,best_results_G,best_results_C):
    results_T=best_results_T
elif TP53_280==1 and (best_results_T>=1.6 or best_results_T>= 1.5*max(best_results_A,best_results_G,best_results_C) or 1.5*best_results_T<= min(best_results_A,best_results_G,best_results_C)):
    results_T=max(min([x for x in [best_results_A, best_results_G, best_results_C] if x >= 0.3] or [0.2]), 0.2)
else:
    results_T=0

combined_sequence7T = combined_sequence6T+sequence7T*results_T
sequence7T_new=sequence7T*results_T

# Create a list to store the names of single-plex genes
single_plex = []  
# Check each gene and add to the list if the condition is met
if NRAS_12==1:
    print ("NRAS_12 : ✓")
    single_plex.append("NRAS_12")
else:
    print ("NRAS_12 : ✗")

if APC_892==1:
    print ("APC_892 : ✓")
    single_plex.append("APC_892")
else:
    print ("APC_892 : ✗")

if TP53_175==1:
    print ("TP53_175 : ✓")
    single_plex.append("TP53_175")
else:
    print ("TP53_175 : ✗") 

if APC_302==1:
    print ("APC_302 : ✓")
    single_plex.append("APC_302")
else:
    print ("APC_302 : ✗")

if TP53_245==1:
    print ("TP53_245 : ✓")
    single_plex.append("TP53_245")
else:
    print ("TP53_245 : ✗")
 
if NRAS_61==1:
    print ("NRAS_61 : ✓")
    single_plex.append("NRAS_61")
else:
    print ("NRAS_61 : ✗")

if TP53_280==1:
    print ("TP53_280 : ✓")
    single_plex.append("TP53_280")
else:
    print ("TP53_280 : ✗")

# Output the list of single-plex genes
print("\nSingle-plex obtained in Multi-plex: ", ', '.join(single_plex))

plt.figure(figsize=(12, 8))
# Plot final mixed sequence
plt.subplot(2, 1, 1)
plt.plot(combined_sequence7A, color='green', label='mixed data A lines', marker='', linewidth=3, linestyle='-')
plt.plot(combined_sequence7G, color='black', label='mixed data G lines', marker='', linewidth=3, linestyle='-')
plt.plot(combined_sequence7C, color='blue', label='mixed data C lines', marker='', linewidth=3, linestyle='-')
plt.plot(combined_sequence7T, color='red', label='mixed data T lines', marker='', linewidth=3, linestyle='-')
plt.ylim(0, 100)
plt.ylabel('Value')
plt.yticks(range(0, 101, 25))

# Plot sequence30 (multiplex sequence)
plt.subplot(2, 1, 2)
plt.plot(sequence30A, color='green', label='UP data A lines', marker='', linewidth=3, linestyle='-')
plt.plot(sequence30G, color='black', label='UP data G lines', marker='', linewidth=3, linestyle='-')
plt.plot(sequence30C, color='blue', label='UP data C lines', marker='', linewidth=3, linestyle='-')
plt.plot(sequence30T, color='red', label='UP data T lines', marker='', linewidth=3, linestyle='-')
plt.ylim(0, 100)
plt.yticks(range(0, 101, 25))
plt.tight_layout()
plt.show()    

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(sequence1A_new, color='green', label='NRAS-12 data A lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence1G_new, color='black', label='NRAS-12 data G lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence1C_new, color='blue', label='NRAS-12 data C lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence1T_new, color='red', label='NRAS-12 data T lines', marker='', linewidth=3, linestyle='-')
# plt.ylabel('Value')
# plt.ylim(0, 100)
# plt.yticks(range(0, 101, 25))

# plt.subplot(2, 1, 2)
# plt.plot(sequence2A_new, color='green', label='APC-892 data A lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence2G_new, color='black', label='APC-892 data G lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence2C_new, color='blue', label='APC-892 data C lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence2T_new, color='red', label='APC-892 data T lines', marker='', linewidth=3, linestyle='-')
# plt.yticks(range(0, 101, 25))
# plt.ylim(0, 100)
# plt.tight_layout()
# plt.show() 

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(sequence3A_new, color='green', label='TP53-175 data A lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence3G_new, color='black', label='TP53-175 data G lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence3C_new, color='blue', label='TP53-175 data C lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence3T_new, color='red', label='TP53-175 data T lines', marker='', linewidth=3, linestyle='-')
# plt.ylabel('Value')
# plt.ylim(0, 100)
# plt.yticks(range(0, 101, 25))

# plt.subplot(2, 1, 2)
# plt.plot(sequence4A_new, color='green', label='APC-302 data A lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence4G_new, color='black', label='APC-302 data G lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence4C_new, color='blue', label='APC-302 data C lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence4T_new, color='red', label='APC-302 data T lines', marker='', linewidth=3, linestyle='-')
# plt.yticks(range(0, 101, 25))
# plt.tight_layout()
# plt.ylim(0, 100)
# plt.show()  

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(sequence5A_new, color='green', label='TP53-245 data A lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence5G_new, color='black', label='TP53-245 data G lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence5C_new, color='blue', label='TP53-245 data C lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence5T_new, color='red', label='TP53-245 data T lines', marker='', linewidth=3, linestyle='-')
# plt.ylabel('Value')
# plt.ylim(0, 100)
# plt.yticks(range(0, 101, 25))

# plt.subplot(2, 1, 2)
# plt.plot(sequence6A_new, color='green', label='NRAS-61 data A lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence6G_new, color='black', label='NRAS-61 data G lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence6C_new, color='blue', label='NRAS-61 data C lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence6T_new, color='red', label='NRAS-61 data T lines', marker='', linewidth=3, linestyle='-')
# # plt.title('UP sequence')
# plt.yticks(range(0, 101, 25))
# # plt.legend()
# plt.tight_layout()
# plt.ylim(0, 100)
# plt.show()  

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(sequence7A_new, color='green', label='TP53-280 data A lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence7G_new, color='black', label='TP53-280 data G lines', marker='', linewidth=3, linestyle='-')
# plt.plot(sequence7C_new, color='blue', label='TP53-280 data C lines', marker='', linewidth=3, linestyle='-')
# plt.ylabel('Value')
# plt.ylim(0, 100)
# plt.yticks(range(0, 101, 25))

# plt.subplot(2, 1, 2)
# plt.plot(combined_sequence7A, color='green', label='mixed data A lines', marker='', linewidth=3, linestyle='-')
# plt.plot(combined_sequence7G, color='black', label='mixed data G lines', marker='', linewidth=3, linestyle='-')
# plt.plot(combined_sequence7C, color='blue', label='mixed data C lines', marker='', linewidth=3, linestyle='-')
# plt.plot(combined_sequence7T, color='red', label='mixed data T lines', marker='', linewidth=3, linestyle='-')
# plt.yticks(range(0, 101, 25))
# plt.tight_layout()
# plt.ylim(0, 100)
# plt.show()  