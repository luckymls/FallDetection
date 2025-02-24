import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


directory = "SIS/"

def load_data(directory):

    data = []
    g_data_all = []
    norm_data_all = []
    gyro_norm_data_all = []
    labels = []
    file_paths = []

    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)

        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if file_name.endswith('.txt') and "Readme" not in file_name:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()

                    cleaned_data = []
                    g_data = [] 
                    norm_data = []
                    gyro_norm_data = []
                    for line in lines:
                        cleaned_line = line.strip().replace(';','')
                        if cleaned_line:
                            try:
                                values = list(map(float, cleaned_line.split(',')))
                                cleaned_data.append(values[:6])
                                values = convert_data_to_g_acc(values)
                                g_data.append(values[:6])
                                ax,ay,az,gx,gy,gz = values[:6] # g values 
                                acc_norm = math.sqrt(ax**2+ay**2+az**2)
                                gyro_norm = math.sqrt(gx**2+gy**2+gz**2)

                                
                                norm_data.append(acc_norm)
                                gyro_norm_data.append(gyro_norm)
                                
                            except ValueError as e:
                                print(f'Errore convertendo la linea a float: {cleaned_line}. Errore: {e}')

                    if cleaned_data:
                        label = 1 if file_name.startswith('F') else 0
                        labels.append(label)
                        file_paths.append(file_path)
                        data.append(cleaned_data)
                        g_data_all.append(g_data)
                        norm_data_all.append(norm_data)
                        gyro_norm_data_all.append(gyro_norm_data)
                        
    return data, g_data_all, norm_data_all, gyro_norm_data_all, np.array(labels), file_paths





def view_data(magnitude, threshold, original_index):
    import matplotlib.pyplot as plt

    hz_frequency = 200  # Sampling frequency in Hz
    time = np.arange(len(magnitude)) / hz_frequency  # Time axis for visualization

    # Adjust index
    trigger_time = original_index / hz_frequency

    plt.figure(figsize=(10, 6))
    plt.plot(time, magnitude, label="Acceleration Magnitude (G)")
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold} G)", zorder=2)

    # Plot trigger point only if within bounds
    if 0 <= original_index < len(magnitude):
        plt.scatter(trigger_time, magnitude[original_index], color="green",
                    label=f"Trigger Point (Time: {trigger_time:.2f}s)", zorder=5)
    else:
        print(f"Warning: Index {original_index} is out of bounds for magnitude data of size {len(magnitude)}.")

    plt.title("Acceleration Magnitude with Threshold")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude (G)")
    plt.legend()
    plt.grid()
    plt.show()






def convert_data_to_g_acc(data):

    converted_data = []
    # Conversion factors
    ACCEL_SCALE = 0.00390625  # ±16g range 
    GYRO_SCALE = 0.06103515625

    ac_x = data[0] * ACCEL_SCALE
    ac_y = data[1] * ACCEL_SCALE
    ac_z = data[2] * ACCEL_SCALE

    gc_x = data[3] * GYRO_SCALE
    gc_y = data[4] * GYRO_SCALE
    gc_z = data[5] * GYRO_SCALE


    converted_data = [ac_x,ac_y,ac_z,gc_x,gc_y,gc_z]
    
    return converted_data



def find_threshold_index(magnitude, threshold):
    try:
        # Find the first index where magnitude exceeds the threshold
        index = next(i for i, val in enumerate(magnitude) if val > threshold)
    except StopIteration:
        # No value exceeds the threshold; handle appropriately
        index = len(magnitude) - 1  # Default to the last index
        print("No value exceeded the threshold. Defaulting to the last index.")
    return index



def limit_data_based_on_threshold_line(data, pre_th_time=2, post_th_time=3, mode="manual", file_name=None):
    print("Original Data Shape:", data.shape)

    # Convert to g and degrees/second
    data = convert_data_to_g_acc(data)

    # Extract acceleration data
    accel_data = data[:, :3]
    accel_magnitude = np.sqrt(np.sum(accel_data**2, axis=1))

    # Threshold detection
    threshold = 1.5  # G threshold
    index = find_threshold_index(accel_magnitude, threshold)

    if index >= len(accel_magnitude):
        print(f"Index {index} exceeds data size. Returning full data.")
        return data

    hz_frequency = 200  # Sampling frequency
    left_data_index = max(0, int(index - pre_th_time * hz_frequency))
    right_data_index = min(len(accel_magnitude), int(index + post_th_time * hz_frequency))

    print(f"Data Range: {left_data_index} to {right_data_index}")

    # Extract limited data
    limited_data = data[left_data_index:right_data_index]

    if mode == "manual" and file_name:
        adjusted_index = index - left_data_index
        view_data(accel_magnitude[left_data_index:right_data_index], threshold, adjusted_index)

    return limited_data



# Load dataset
data, g_data, norm_data, gyro_data, labels, paths = load_data('SIS/')

# Lists to store max acceleration values
fall_max_values = []
non_fall_max_values = []
fall_labels = []
non_fall_labels = []

# Extract max acceleration for each file
for index, file_path in enumerate(paths):
    file_data = norm_data[index]
    fileMax = max(file_data)  
    label = labels[index]

    if label == 1:  # Fall
        fall_max_values.append(fileMax)
        fall_labels.append(index)
    else:  # Non-fall
        non_fall_max_values.append(fileMax)
        non_fall_labels.append(index)

# Compute histograms with common bins
common_bins = np.linspace(
    min(fall_max_values + non_fall_max_values), 
    max(fall_max_values + non_fall_max_values), 
    121  # 120 bins
)

fall_hist, bin_edges = np.histogram(fall_max_values, bins=common_bins)

first_fall_index = np.where(fall_hist > 0)[0][0]
threshold = bin_edges[first_fall_index]

print(f"Threshold at start of fall region: {threshold:.2f} g")

plt.figure(figsize=(12, 7))
plt.hist(fall_max_values, bins=common_bins, alpha=0.7, label='Falls', color='red', linewidth=2)
plt.hist(non_fall_max_values, bins=common_bins, alpha=0.7, label='Non-falls', color='blue', linewidth=2)

plt.axvline(x=threshold, color='black', linestyle='--', linewidth=3, label=f'Threshold = {threshold:.2f} g')

plt.xlabel('Max Acceleration (g)', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title('Distribution of Maximum Acceleration Between Falls and Non-Falls', fontsize=18)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([min(fall_max_values + non_fall_max_values) - 0.1, max(fall_max_values + non_fall_max_values) + 0.1])
plt.legend(loc='upper right', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Scatter plot
plt.figure(figsize=(12, 7))
plt.scatter(fall_max_values, fall_labels, color='red', label='Falls', alpha=0.7, s=80)  
plt.scatter(non_fall_max_values, non_fall_labels, color='blue', label='Non-falls', alpha=0.7, s=80) 

plt.xlabel('Massima Accelerazione (g)', fontsize=16)
plt.ylabel('Indice File', fontsize=16)
plt.title('Scatter Plot della Massima Accelerazione tra Cadute e Non Cadute', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper right', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()





# ------ FALLO ANCHE PER IL GIROSCOPIO ------



fall_max_values = []
non_fall_max_values = []
fall_labels = []
non_fall_labels = []

for file_path in paths:
    
    index = paths.index(file_path)
    file_data = gyro_data[index]
    fileMax = max(file_data)
    
    label = labels[index]
    if label == 1:  # Fall
        fall_max_values.append(fileMax)
        fall_labels.append(index)
    else:  # Non-fall
        non_fall_max_values.append(fileMax)
        non_fall_labels.append(index)

common_bins = np.linspace(
    min(min(fall_max_values), min(non_fall_max_values)),
    max(max(fall_max_values), max(non_fall_max_values)),
    121
)

fall_hist, bin_edges = np.histogram(fall_max_values, bins=common_bins)
non_fall_hist, _ = np.histogram(non_fall_max_values, bins=common_bins)

overlap_indices = np.where((fall_hist > 0) & (non_fall_hist > 0))[0]
overlap_points = bin_edges[overlap_indices]
first_overlap_point = overlap_points[0] if len(overlap_points) > 0 else None

plt.figure(figsize=(12, 7))
plt.hist(fall_max_values, bins=common_bins, alpha=0.7, label='Falls', color='red', linewidth=2)
plt.hist(non_fall_max_values, bins=common_bins, alpha=0.7, label='Non-falls', color='blue', linewidth=2)

if first_overlap_point:
    plt.axvline(x=first_overlap_point, color='green', linestyle='--', linewidth=3, 
                label=f'Overlap = {first_overlap_point:.2f} °/s')

plt.xlabel('Maximum Rotation (°/s)', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title('Distribution of Maximum Rotation Between Falls and Non-Falls', fontsize=18)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([min(fall_max_values + non_fall_max_values) - 0.1, max(fall_max_values + non_fall_max_values) + 0.1])
plt.legend(loc='upper right', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
