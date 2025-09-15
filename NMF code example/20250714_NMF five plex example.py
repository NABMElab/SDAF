import pandas as pd
import numpy as np
import os
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from dtw import *
from fastdtw import fastdtw
from itertools import combinations
from scipy.signal import find_peaks
import math
from scipy.optimize import nnls  # 导入非负最小二乘法函数
from fastdtw import fastdtw
from itertools import combinations

length_m2 = 168
length_m24 = 147
length_mp70 = 146
length_mp79 = 134
length_mp12 = 131

m2 = 0
m24 = 0
mp70 = 0
mp79 = 0
mp12 = 0

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/NA18537_5plex/5plex_S3/20250324_5plex_s3_1(0.01).xlsx", header=None)

# Read the first row (index 0) and store it in an array
sequence30A = df.iloc[1, 6:1105].tolist()
sequence30G = df.iloc[2, 6:1105].tolist()
sequence30C = df.iloc[3, 6:1105].tolist()
sequence30T = df.iloc[4, 6:1105].tolist()
# Convert sequence30A to a numpy array
sequence30A = np.array(sequence30A)
sequence30G = np.array(sequence30G)
sequence30C = np.array(sequence30C)
sequence30T = np.array(sequence30T)

# sequence30A = np.concatenate((np.zeros(13),sequence30A))
# sequence30G = np.concatenate((np.zeros(13),sequence30G))
# sequence30C = np.concatenate((np.zeros(13),sequence30C))
# sequence30T = np.concatenate((np.zeros(13),sequence30T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/5plex单谱信息/S3-M2.xlsx", header=None)

sequence1A = df.iloc[1, 0:1099].tolist()
sequence1G = df.iloc[2, 0:1099].tolist()
sequence1C = df.iloc[3, 0:1099].tolist()
sequence1T = df.iloc[4, 0:1099].tolist()

sequence1A = np.array(sequence1A)
sequence1G = np.array(sequence1G)
sequence1C = np.array(sequence1C)
sequence1T = np.array(sequence1T)

sequence1A = np.concatenate((np.zeros(12),sequence1A))
sequence1G = np.concatenate((np.zeros(12),sequence1G))
sequence1C = np.concatenate((np.zeros(12),sequence1C))
sequence1T = np.concatenate((np.zeros(12),sequence1T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/5plex单谱信息/S3-M24.xlsx", header=None)

sequence2A = df.iloc[1, 0:1099].tolist()
sequence2G = df.iloc[2, 0:1099].tolist()
sequence2C = df.iloc[3, 0:1099].tolist()
sequence2T = df.iloc[4, 0:1099].tolist()

sequence2A = np.array(sequence2A)
sequence2G = np.array(sequence2G)
sequence2C = np.array(sequence2C)
sequence2T = np.array(sequence2T)

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/5plex单谱信息/S3-MP70.xlsx", header=None)

sequence3A = df.iloc[1, 0:1099].tolist()
sequence3G = df.iloc[2, 0:1099].tolist()
sequence3C = df.iloc[3, 0:1099].tolist()
sequence3T = df.iloc[4, 0:1099].tolist()

sequence3A = np.array(sequence3A)
sequence3G = np.array(sequence3G)
sequence3C = np.array(sequence3C)
sequence3T = np.array(sequence3T)

sequence3A = np.concatenate((np.zeros(15),sequence3A))
sequence3G = np.concatenate((np.zeros(15),sequence3G))
sequence3C = np.concatenate((np.zeros(15),sequence3C))
sequence3T = np.concatenate((np.zeros(15),sequence3T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/5plex单谱信息/S3-MP79.xlsx", header=None)

sequence4A = df.iloc[1, 0:1099].tolist()
sequence4G = df.iloc[2, 0:1099].tolist()
sequence4C = df.iloc[3, 0:1099].tolist()
sequence4T = df.iloc[4, 0:1099].tolist()

sequence4A = np.array(sequence4A)
sequence4G = np.array(sequence4G)
sequence4C = np.array(sequence4C)
sequence4T = np.array(sequence4T)

# sequence4A = np.concatenate((np.zeros(1),sequence4A))
# sequence4G = np.concatenate((np.zeros(1),sequence4G))
# sequence4C = np.concatenate((np.zeros(1),sequence4C))
# sequence4T = np.concatenate((np.zeros(1),sequence4T))

# Load the Excel file (assuming the first sheet and no header)
df = pd.read_excel("C:/Users/Jack Leong/Downloads/5plex单谱信息/S3-MP12.xlsx", header=None)

sequence5A = df.iloc[1, 0:1099].tolist()
sequence5G = df.iloc[2, 0:1099].tolist()
sequence5C = df.iloc[3, 0:1099].tolist()
sequence5T = df.iloc[4, 0:1099].tolist()

sequence5A = np.array(sequence5A)
sequence5G = np.array(sequence5G)
sequence5C = np.array(sequence5C)
sequence5T = np.array(sequence5T)

# sequence5A = np.concatenate((np.zeros(1),sequence5A))
# sequence5G = np.concatenate((np.zeros(1),sequence5G))
# sequence5C = np.concatenate((np.zeros(1),sequence5C))
# sequence5T = np.concatenate((np.zeros(1),sequence5T))


sequence0A = sequence1A*0
sequence0G = sequence1G*0
sequence0C = sequence1C*0
sequence0T = sequence1T*0

max_sequence1 = max(max(sequence1A, default=0),max(sequence1G, default=0), max(sequence1C, default=0),max(sequence1T, default=0))
max_sequence2 = max(max(sequence2A, default=0),max(sequence2G, default=0), max(sequence2C, default=0),max(sequence2T, default=0))
max_sequence3 = max(max(sequence3A, default=0),max(sequence3G, default=0), max(sequence3C, default=0),max(sequence3T, default=0))
max_sequence4 = max(max(sequence4A, default=0),max(sequence4G, default=0), max(sequence4C, default=0),max(sequence4T, default=0))
max_sequence5 = max(max(sequence5A, default=0),max(sequence5G, default=0), max(sequence5C, default=0),max(sequence5T, default=0))
max_sequence30 = max(max(sequence30A, default=0),max(sequence30G, default=0), max(sequence30C, default=0),max(sequence30T, default=0))

sequence1A = sequence1A/max_sequence1*100
sequence2A = sequence2A/max_sequence2*100
sequence3A = sequence3A/max_sequence3*100
sequence4A = sequence4A/max_sequence4*100
sequence5A = sequence5A/max_sequence5*100
sequence1G = sequence1G/max_sequence1*100
sequence2G = sequence2G/max_sequence2*100
sequence3G = sequence3G/max_sequence3*100
sequence4G = sequence4G/max_sequence4*100
sequence5G = sequence5G/max_sequence5*100
sequence1C = sequence1C/max_sequence1*100
sequence2C = sequence2C/max_sequence2*100
sequence3C = sequence3C/max_sequence3*100
sequence4C = sequence4C/max_sequence4*100
sequence5C = sequence5C/max_sequence5*100
sequence1T = sequence1T/max_sequence1*100
sequence2T = sequence2T/max_sequence2*100
sequence3T = sequence3T/max_sequence3*100
sequence4T = sequence4T/max_sequence4*100
sequence5T = sequence5T/max_sequence5*100
sequence30A = sequence30A/max_sequence30*100
sequence30G = sequence30G/max_sequence30*100
sequence30C = sequence30C/max_sequence30*100
sequence30T = sequence30T/max_sequence30*100

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(sequence1A, color='green', label='mixed data A lines', marker='', linestyle='-')
plt.plot(sequence1G, color='black', label='mixed data G lines', marker='', linestyle='-')
plt.plot(sequence1C, color='blue', label='mixed data C lines', marker='', linestyle='-')
plt.plot(sequence1T, color='red', label='mixed data T lines', marker='', linestyle='-')
plt.ylabel('Value')
plt.ylim(0, 100)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(sequence30A, color='green', label='UP data A lines', marker='', linestyle='-')
plt.plot(sequence30G, color='black', label='UP data G lines', marker='', linestyle='-')
plt.plot(sequence30C, color='blue', label='UP data C lines', marker='', linestyle='-')
plt.plot(sequence30T, color='red', label='UP data T lines', marker='', linestyle='-')
plt.legend()
plt.tight_layout()
plt.show()

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
sequencesA = [sequence1A, sequence2A, sequence3A, sequence4A, sequence5A, sequence0A]  # Define all sequences for channel A
sequencesG = [sequence1G, sequence2G, sequence3G, sequence4G, sequence5G, sequence0G]  # Define all sequences for channel G
sequencesC = [sequence1C, sequence2C, sequence3C, sequence4C, sequence5C, sequence0C]  # Define all sequences for channel C
sequencesT = [sequence1T, sequence2T, sequence3T, sequence4T, sequence5T, sequence0T]  # Define all sequences for channel T

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

sequence_set_6 = {'bases':sorted_bases_all[5], 'peaks': sorted_peak_positions_all[5]}

# Sequence sets to compare
sequence_sets = [
    {'bases': sorted_bases_all[0], 'peaks': sorted_peak_positions_all[0]},  # Sequence Set 1
    {'bases': sorted_bases_all[1], 'peaks': sorted_peak_positions_all[1]},   # Sequence Set 2
    {'bases': sorted_bases_all[2], 'peaks': sorted_peak_positions_all[2]},  # Sequence Set 3
    {'bases': sorted_bases_all[3], 'peaks': sorted_peak_positions_all[3]},  # Sequence Set 4
    {'bases': sorted_bases_all[4], 'peaks': sorted_peak_positions_all[4]},  # Sequence Set 5
]

# Calculate similarities to sequence sets
similarities = []
for i, seq_set in enumerate(sequence_sets[:5]):  # Only compare with Sequence Set 1 to 5
    matched_peaks = compare_peak_bases_with_tolerance(sequence_set_6, seq_set)
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
    length_ref= length_m2
    ref_sequenceA = sequence1A
    ref_sequenceG = sequence1G
    ref_sequenceC = sequence1C
    ref_sequenceT = sequence1T

elif most_similar_sequence['set'] == 2:
    length_ref= length_m24
    ref_sequenceA = sequence2A
    ref_sequenceG = sequence2G
    ref_sequenceC = sequence2C
    ref_sequenceT = sequence2T

elif most_similar_sequence['set'] == 3:
    length_ref= length_mp70
    ref_sequenceA = sequence3A
    ref_sequenceG = sequence3G
    ref_sequenceC = sequence3C
    ref_sequenceT = sequence3T

elif most_similar_sequence['set'] == 4:
    length_ref= length_mp79
    ref_sequenceA = sequence4A
    ref_sequenceG = sequence4G
    ref_sequenceC = sequence4C
    ref_sequenceT = sequence4T
elif most_similar_sequence['set'] == 5:
    length_ref= length_mp12
    ref_sequenceA = sequence5A
    ref_sequenceG = sequence5G
    ref_sequenceC = sequence5C
    ref_sequenceT = sequence5T
else:
    length_ref = None  # Handle the case where no ref_sequence is chosen

print(ref_sequenceA)
print(length_ref)

if length_ref == None:
    print("This is a chaos spectrum, no mutation in this sample")

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
]

sorted_peak_positions_all = []
for i, seq_set in enumerate(sequences):
    sorted_peaks = find_and_sort_peaks(seq_set, segment_start, segment_end, height_threshold)
    print(f"\nPeaks for Sequence Set {i+1}:")
    print_peaks(sorted_peaks)
    sorted_peak_positions_all.append(get_sorted_peak_positions(sorted_peaks))

# Extract different zero padding values
zerospadding_m2 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_m2,0)+1] - sorted_peak_positions_all[1][1],0)
zerospadding_m24 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_m24,0)+1] - sorted_peak_positions_all[2][1],0)
zerospadding_mp70 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_mp70,0)+1] - sorted_peak_positions_all[3][1],0)
zerospadding_mp79 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_mp79,0)+1] - sorted_peak_positions_all[4][1],0)
zerospadding_mp12 = np.maximum(sorted_peak_positions_all[0][np.maximum(length_ref-length_mp12,0)+1] - sorted_peak_positions_all[5][1],0)

# print(sorted_peak_positions_all[0][length_m2-length_m24])
print(zerospadding_m2)
print(zerospadding_m24)
print(zerospadding_mp70)
print(zerospadding_mp79)
print(zerospadding_mp12)

sequence1A = np.concatenate((np.zeros(zerospadding_m2), sequence1A))
sequence1G = np.concatenate((np.zeros(zerospadding_m2), sequence1G))
sequence1C = np.concatenate((np.zeros(zerospadding_m2), sequence1C))
sequence1T = np.concatenate((np.zeros(zerospadding_m2), sequence1T))
sequence2A = np.concatenate((np.zeros(zerospadding_m24), sequence2A))
sequence2G = np.concatenate((np.zeros(zerospadding_m24), sequence2G))
sequence2C = np.concatenate((np.zeros(zerospadding_m24), sequence2C))
sequence2T = np.concatenate((np.zeros(zerospadding_m24), sequence2T))
sequence3A = np.concatenate((np.zeros(zerospadding_mp70), sequence3A))
sequence3G = np.concatenate((np.zeros(zerospadding_mp70), sequence3G))
sequence3C = np.concatenate((np.zeros(zerospadding_mp70), sequence3C))
sequence3T = np.concatenate((np.zeros(zerospadding_mp70), sequence3T))
sequence4A = np.concatenate((np.zeros(zerospadding_mp79), sequence4A))
sequence4G = np.concatenate((np.zeros(zerospadding_mp79), sequence4G))
sequence4C = np.concatenate((np.zeros(zerospadding_mp79), sequence4C))
sequence4T = np.concatenate((np.zeros(zerospadding_mp79), sequence4T))
sequence5A = np.concatenate((np.zeros(zerospadding_mp12), sequence5A))
sequence5G = np.concatenate((np.zeros(zerospadding_mp12), sequence5G))
sequence5C = np.concatenate((np.zeros(zerospadding_mp12), sequence5C))
sequence5T = np.concatenate((np.zeros(zerospadding_mp12), sequence5T))

max_sequence1 = max(max(sequence1A, default=0),max(sequence1G, default=0), max(sequence1C, default=0),max(sequence1T, default=0))
max_sequence2 = max(max(sequence2A, default=0),max(sequence2G, default=0), max(sequence2C, default=0),max(sequence2T, default=0))
max_sequence3 = max(max(sequence3A, default=0),max(sequence3G, default=0), max(sequence3C, default=0),max(sequence3T, default=0))
max_sequence4 = max(max(sequence4A, default=0),max(sequence4G, default=0), max(sequence4C, default=0),max(sequence4T, default=0))
max_sequence5 = max(max(sequence5A, default=0),max(sequence5G, default=0), max(sequence5C, default=0),max(sequence5T, default=0))
max_sequence30 = max(max(sequence30A, default=0),max(sequence30G, default=0), max(sequence30C, default=0),max(sequence30T, default=0))

sequence1A = sequence1A/max_sequence1*100
sequence2A = sequence2A/max_sequence2*100
sequence3A = sequence3A/max_sequence3*100
sequence4A = sequence4A/max_sequence4*100
sequence5A = sequence5A/max_sequence5*100
sequence1G = sequence1G/max_sequence1*100
sequence2G = sequence2G/max_sequence2*100
sequence3G = sequence3G/max_sequence3*100
sequence4G = sequence4G/max_sequence4*100
sequence5G = sequence5G/max_sequence5*100
sequence1C = sequence1C/max_sequence1*100
sequence2C = sequence2C/max_sequence2*100
sequence3C = sequence3C/max_sequence3*100
sequence4C = sequence4C/max_sequence4*100
sequence5C = sequence5C/max_sequence5*100
sequence1T = sequence1T/max_sequence1*100
sequence2T = sequence2T/max_sequence2*100
sequence3T = sequence3T/max_sequence3*100
sequence4T = sequence4T/max_sequence4*100
sequence5T = sequence5T/max_sequence5*100
sequence30A = sequence30A/max_sequence30*100
sequence30G = sequence30G/max_sequence30*100
sequence30C = sequence30C/max_sequence30*100
sequence30T = sequence30T/max_sequence30*100
# Find the length of the shorter sequence
min_length = min(len(sequence1A), len(sequence2A),len(sequence3A),len(sequence4A),len(sequence5A),len(sequence30A))
#min_length = 80

# Shorten the longer sequence to match the length of the shorter sequence
sequence1A = sequence1A[:min_length]  #reduce length of sequence mp22 to length 1200
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
sequence30A = sequence30A[:min_length] 
sequence30G = sequence30G[:min_length] 
sequence30C = sequence30C[:min_length]
sequence30T = sequence30T[:min_length]
sequence0A = sequence0A[:min_length] 
sequence0G = sequence0G[:min_length] 
sequence0C = sequence0C[:min_length]
sequence0T = sequence0T[:min_length]

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

# Apply this shift to all sequences in channels A, G, C and T
sequence30A = align_sequences_based_on_shift(sequence30A, shift)
sequence30G = align_sequences_based_on_shift(sequence30G, shift)
sequence30C = align_sequences_based_on_shift(sequence30C, shift)
sequence30T = align_sequences_based_on_shift(sequence30T, shift)

sequence30A = sequence30A[:min_length] 
sequence30G = sequence30G[:min_length] 
sequence30C = sequence30C[:min_length]
sequence30T = sequence30T[:min_length]
plt.figure(figsize=(12, 8))
# Plot the alignment path for m2,m22 and AD peak lines
plt.subplot(2, 1, 1)
plt.plot(sequence1A, color='green', label='mixed data A lines', marker='', linestyle='-')
plt.plot(sequence1G, color='black', label='mixed data G lines', marker='', linestyle='-')
plt.plot(sequence1C, color='blue', label='mixed data C lines', marker='', linestyle='-')
plt.plot(sequence1T, color='red', label='mixed data T lines', marker='', linestyle='-')
# plt.title('mixed sequence')
plt.ylabel('Value')
plt.ylim(0, 100)
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

print(sequence1A)
# Helper function to calculate NNLS weights
def calculate_weights(F, mixed_data, sequence_type):
    try:
        W, _ = nnls(F, mixed_data)
        sequence_names = [f'sequence{i}{sequence_type}' for i in range(1, 6)]
        for sequence_name, weight in zip(sequence_names, W):
            print(f'Weight for {sequence_name}: {weight}')
        return W
    except Exception as e:
        print(f"Error in solving for weights: {e}")
        return None

# Combine all sequences into matrices
A = np.vstack([sequence1A, sequence2A, sequence3A, sequence4A, sequence5A]).T
G = np.vstack([sequence1G, sequence2G, sequence3G, sequence4G, sequence5G]).T
C = np.vstack([sequence1C, sequence2C, sequence3C, sequence4C, sequence5C]).T
T = np.vstack([sequence1T, sequence2T, sequence3T, sequence4T, sequence5T]).T

# Calculate weights for each matrix
weights_A = calculate_weights(A, sequence30A, "A")
weights_G = calculate_weights(G, sequence30G, "G")
weights_C = calculate_weights(C, sequence30C, "C")
weights_T = calculate_weights(T, sequence30T, "T")

# Initialize a list to store the names of single-plexes that are detected
existing_singleplexes = []
# Mapping from sequence number to name
sequence_name_map = {
        1: "m2",
        2: "m24",
        3: "mp70",
        4: "mp79",
        5: "mp12"
}
# Function to check if at least two weights > 0.2 for a given sequence
def check_sequence_existence(seq_num, weights_A, weights_G, weights_C, weights_T):
    weightA = weights_A[seq_num - 1]  # Weight for sequenceX_A (1-based index)
    weightG = weights_G[seq_num - 1]  # Weight for sequenceX_G
    weightC = weights_C[seq_num - 1]  # Weight for sequenceX_C
    weightT = weights_T[seq_num - 1]  # Weight for sequenceX_T

    weights_above_threshold = [w for w in [weightA, weightG, weightC, weightT] if w > 0.2]

    if len(weights_above_threshold) >= 2:
        print(f"Sequence {seq_num} exists in the mixed data (at least two weights > 0.2).")
        print(f"{sequence_name_map[seq_num]} exists")
        existing_singleplexes.append(sequence_name_map[seq_num])
    else:
        print(f"Sequence {seq_num} does not exist in the mixed data (fewer than two weights > 0.2).")

# Check for each sequence from 1 to 5
if weights_A is not None and weights_G is not None and weights_C is not None and weights_T is not None:
    for seq_num in range(1, 6):  # Loop through sequence 1 to sequence 5
        check_sequence_existence(seq_num, weights_A, weights_G, weights_C, weights_T)
    # Final summary output of detected single-plexes
    if existing_singleplexes:
        print("\n Single-plexes that exist in mixed data:")
        print(", ".join(existing_singleplexes))
    else:
        print("\n No single-plexes detected in mixed data.")