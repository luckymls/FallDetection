import os, ast
import numpy as np
import joblib  # Per salvare e caricare il modello
import random, math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importo i modelli
from sklearn.ensemble import RandomForestClassifier # Random forest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  # SVM classifier
from sklearn.ensemble import BaggingClassifier

# Metriche di misura dell'accuratezza
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA # Ridurre la dimensione del problema
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV


directory = "SIS/"

# Flag per decidere se salvare il modello
save = True
flag=True
while flag:
    saveChoice = "n"#input("Do you wanna save the trained model? Y/N\n> ")
    
    if saveChoice.lower() in ['y', 'n']:
        flag=False
        if saveChoice.lower() == 'y':
            save = True

    else:
        print("Error, you should choose one of the two options.\n")
    

flag = True
while flag:
    modelChoice = "1"#input("Choose a model\n[1] Random Forest\n[2] KNN\n[3] SVM\n[4] EBT\n> ")
    
    if(modelChoice == '1' or modelChoice == '2' or modelChoice == '3' or modelChoice == '4'):
        flag=False
        if modelChoice == '1':
            model='RF'
        if modelChoice == '2':
            model='KNN'
        if modelChoice == '3':
            model='SVM'
        if modelChoice == '4':
            model='EBT'
    else:
        print("Error, you should choose between the proposed model")



# FUNZIONE PER LA VISUALIZZAZIONE DEL KNN

def visualize_knn_decision_boundary(model, X_train, y_train):
    # Define the range of values for visualization
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.75)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, edgecolor='k')
    plt.title("KNN Decision Boundary")
    plt.show()





def visualize_data(accel_data, threshold, index, sampling_frequency=200, title=None):

    # Calculate magnitude of acceleration
    magnitude = np.sqrt(np.sum(accel_data**2, axis=1))

    print("Acceleration Magnitude:", magnitude[:5])
    # Calculate time in seconds for each sample
    time = np.arange(len(magnitude)) / sampling_frequency

    # Plot magnitude over time
    plt.figure(figsize=(10, 6))
    plt.plot(time, magnitude, label="Acceleration Magnitude", color="blue")
    
    # Add threshold line
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold = {threshold}G")
    
    # Highlight the triggering point
    if index >= 0:
        trigger_time = index / sampling_frequency
        plt.scatter(trigger_time, magnitude[index], color="green", label=f"Trigger Point (Time: {trigger_time:.2f}s)", zorder=5)

    # Add labels, legend, and title
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration Magnitude (G)")
    plt.title(title if title else "Threshold Visualization")
    plt.legend()
    plt.grid()
    plt.show()



def max_sum_vector_magnitude(accel_data):

    
    norms = np.linalg.norm(accel_data, axis=1)  # Calcola la norma per ogni riga
    max_value = np.max(norms)                   # Trova il valore massimo
    return np.argmax(norms)                # Trova l'indice della riga con il massimo valore
    

def convert_data_to_g_acc(accel_data):
    '''
    # Constants for ADXL345
    accel_resolution = 13  # bits
    accel_range = 8  # ±16g

    # Handle signed 13-bit values
    max_value = 2**(accel_resolution - 1)  # Max positive value (8192)
    accel_data = accel_data.astype(np.int16)  # signed 16-bit interpretation

    # Calculate the scaling factor
    accel_scale = (2 * accel_range) / (2**accel_resolution)
    '''
    accel_scale = 0.00390625
    g_accel = accel_scale * accel_data
    return g_accel



def convert_all_data_to_international_system(data):

    accel_scale = 0.00390625
    rot_scale = 0.06103515625
    accel_data = data[:, :3]
    rot_data = data[:, 3:6]

    accel_data = accel_scale*accel_data*9.81
    rot_data = rot_scale*rot_data

    converted_data = np.hstack((accel_data, rot_data))

    return converted_data

def find_threshold_index(data, threshold):
  
    # Compute the magnitude of the acceleration vector
    magnitude = np.sqrt(np.sum(data**2, axis=1))

    # Find the index where the magnitude exceeds the threshold
    indices = np.where(magnitude > threshold)[0]

    # Return the first index, or -1 if none found
    return indices[0] if len(indices) > 0 else -1



avg_detection_time = []

def analyze_acceleration(file_name, data, window_size=1, window_max=0.5, sample_rate=200):
    
    if file_name.startswith('D'):
        pass
    
    accel_data = data[:, :3]
    accel_data = convert_data_to_g_acc(accel_data)
    accSVM = np.linalg.norm(accel_data, axis=1)
    total_frames = len(accSVM)
    time = np.arange(total_frames) / sample_rate
    window_frames = int(window_size * sample_rate)
    


    # Calcolo il centro del riferimento della finesta per la deviazione standard
    half_window = int(window_frames // 2)
    std_values = []
    
    for i in range(total_frames):

        start_idx = max(0, i - half_window)
        end_idx = min(total_frames, i + half_window+1)
        window_std = np.std(accSVM[start_idx:end_idx])
        std_values.append(window_std)
        
    std_values = np.array(std_values)
    
    baseline = np.median(std_values)
    
    significant_points = []
    in_significant_window = False
    start_idx = None

    
    for i in range(len(std_values)):
        if std_values[i] >= baseline * 1 and not in_significant_window:  # Check for significant increase
            in_significant_window = True
            start_idx = i
        elif std_values[i] <= baseline * 1 and in_significant_window:  # End of the significant variation
            in_significant_window = False
            significant_points.append((start_idx, i))



    # Calcolo l'area e tengo quella con valore maggiore (maggiore varianza)
    filtered_points = []
    max_area = 0
    
    for start, end in significant_points:
        area = sum(std_values[start:end])
        
        
        if area > max_area:
            max_area = area              
            filtered_points = [(start, end)]

    significant_points = filtered_points # Contiene solo una coppia. 

    reduced_points = []
    if len(significant_points) > 0:
        curr_window_size = (significant_points[0][1]-significant_points[0][0])/sample_rate


        
        if curr_window_size > window_max: # Se la finestra corrente è maggiore della finestra massima consentita
            
            
            p1_index = significant_points[0][0] # Start index
            p2_index = significant_points[0][1] # End index
            
            # Calcolo proporzioni di riduzione finestra (converge verso il massimo della deviazione)
            max_std_index = np.argmax(accSVM[p1_index:p2_index])+p1_index


            p1_std_size_time = (max_std_index - p1_index)/sample_rate # Tempo dall'inizio della finestra al massimo della deviazione standard
            p2_std_size_time = (p2_index-max_std_index)/sample_rate # Tempo dal massimo della deviazione standard e fine della finesta


            p1_x_proportion = (p1_std_size_time*window_max)/curr_window_size
            p2_x_proportion = (p2_std_size_time*window_max)/curr_window_size


            win_to_reduce = curr_window_size-window_max

            p1_to_reduce = (win_to_reduce*p1_x_proportion)/window_max
            p2_to_reduce = (win_to_reduce*p2_x_proportion)/window_max
            
            p1_index += math.ceil(p1_to_reduce*sample_rate)
            p2_index -= math.ceil(p2_to_reduce*sample_rate)


            reduced_points = [(p1_index, p2_index)]

    
    plot=True
    zoomed_graph=False
    
    if plot:

        plt.figure(figsize=(12, 8))

        # Original Acceleration Graph
        if zoomed_graph:
            plt.subplot(4, 1, 1)
        else:
            plt.subplot(2, 1, 1)
        plt.plot(time, accSVM, label='AccSVM')
        for start, end in significant_points:
            pass
            #plt.axvline(x=time[start], color='green', linestyle='--', label='Significant Start' if start == significant_points[0][0] else "")
            #plt.axvline(x=time[end], color='red', linestyle='--', label='Significant End' if end == significant_points[0][1] else "")
            
        if len(reduced_points) > 0:

            for start, end in reduced_points:
                pass
                #plt.axvline(x=time[start], color='grey', linestyle='--', label='Reduced Start')
                #plt.axvline(x=time[end], color='purple', linestyle='--', label='Reduced End')
        
        plt.title(f'Original Signal (File: {file_name})')
        plt.xlabel('Time (s)')
        plt.ylabel('AccSVM')
        plt.legend()


        # Significant Parts Graph
        if zoomed_graph:
            plt.subplot(4, 1, 2)
        else:
            plt.subplot(2, 1, 2)
        
        plt.plot(time[:len(std_values)], std_values, label='Standard Deviation', color='orange')
        plt.axhline(baseline, color='black', linestyle='--', label='Baseline')

        for start, end in significant_points:
            plt.axvline(x=time[start], color='green', linestyle='--', label='Significant Start' if start == significant_points[0][0] else "")
            plt.axvline(x=time[end], color='red', linestyle='--', label='Significant End' if end == significant_points[0][1] else "")

        if len(reduced_points) > 0:
    
            for start, end in reduced_points:
                plt.axvline(x=time[start], color='grey', linestyle='--', label='Reduced Start' if start == reduced_points[0][0] else "")
                plt.axvline(x=time[end], color='purple', linestyle='--', label='Reduced End' if end == reduced_points[0][1] else "")
            
        plt.title('Standard Deviation')
        plt.xlabel('Time (s)')
        plt.ylabel('Standard Deviation')
        plt.legend()


        # Zoomed Parts Graph

        if len(reduced_points) > 0 and zoomed_graph:

            for start, end in reduced_points:
                
                plt.subplot(4, 1, 3)
                plt.plot(time[:len(accSVM[reduced_points[0][0]:reduced_points[0][1]])], accSVM[reduced_points[0][0]:reduced_points[0][1]], label='AccSVM', color='blue')
                plt.title('Zoomed Acceleration Region')
                plt.xlabel('Time (s)')
                plt.ylabel('Standard Deviation')
                plt.legend()

            for start, end in reduced_points:
                
                plt.subplot(4, 1, 4)
                plt.plot(time[:len(std_values[reduced_points[0][0]:reduced_points[0][1]])], std_values[reduced_points[0][0]:reduced_points[0][1]], label='Standard Deviation', color='orange')
                plt.axhline(baseline, color='black', linestyle='--', label='Baseline')
                plt.title('Zoomed Standard Deviation Region')
                plt.xlabel('Time (s)')
                plt.ylabel('Standard Deviation')
                plt.legend()


        plt.tight_layout()
        plt.show()

    if len(reduced_points) > 0:
        start = reduced_points[0][0]
        end = reduced_points[0][1]
        avg_detection_time.append(end-start)
        
        return data[start:end]    
    else:

        if len(significant_points) > 0:
           start = significant_points[0][0]
           end = significant_points[0][1]
           avg_detection_time.append(end-start)
           return data[start:end]
        else:
            avg_detection_time.append(len(data))
            return data




#avg_detection_time = []

def analyze_acceleration_w_threshold(file_name, data, window_size=1, window_max=0.5, sample_rate=200, threshold=1.66):
    
    if file_name.startswith('D'):
        pass
    
    accel_data = data[:, :3]
    accel_data = convert_data_to_g_acc(accel_data)
    accSVM = np.linalg.norm(accel_data, axis=1)
    total_frames = len(accSVM)
    time = np.arange(total_frames) / sample_rate
    window_frames = int(window_size * sample_rate)
    


    # Calcolo il centro del riferimento della finesta per la deviazione standard
    half_window = int(window_frames // 2)
    std_values = []
    
    for i in range(total_frames):

        start_idx = max(0, i - half_window)
        end_idx = min(total_frames, i + half_window+1)
        window_std = np.std(accSVM[start_idx:end_idx])
        std_values.append(window_std)
        
    std_values = np.array(std_values)
    
    baseline = np.median(std_values)
    
    significant_points = []
    in_significant_window = False
    start_idx = None

    
    for i in range(len(std_values)):
        if std_values[i] >= baseline * 1 and not in_significant_window:
            in_significant_window = True
            start_idx = i
        elif std_values[i] <= baseline * 1 and in_significant_window:
            in_significant_window = False
            significant_points.append((start_idx, i))


    '''
    # Calcolo l'area e tengo quella con valore maggiore (maggiore varianza)
    filtered_points = []
    max_area = 0
    threshold_inside = False
    
    for start, end in significant_points:
        area = sum(std_values[start:end])
        
        
        for i in range(start, end): # Se il threshold è all'interno dell'area
            if accSVM[i] >= threshold:
                threshold_inside = True
                
        if area > max_area and threshold_inside:
            max_area = area              
            filtered_points = [(start, end)]
            print("Threshold interno")
            threshold_inside = False
        elif area > max_area and not threshold_inside:
            max_area = area
            filtered_points = [(start, end)]
    '''

    filtered_points = None
    for start, end in significant_points:
        
        threshold_inside = any(accSVM[i] >= threshold for i in range(start, end))
        
        if threshold_inside:
        
            filtered_points = [(start, end)]
            #print("Metodo threshold")
            break
    filtered_points = None

    if filtered_points is None: # Otherwise chose the largest area
        max_area = 0
        for start, end in significant_points:
            area = sum(std_values[start:end])
            if area > max_area:
                max_area = area
                filtered_points = [(start, end)]
        #print("Metodo area massima.")
            
    
    if filtered_points:
        significant_points = filtered_points # Contiene solo una coppia. 

    try:
        curr_window_size = (significant_points[0][1]-significant_points[0][0])/sample_rate
    except:
        return

    reduced_points = []
    if curr_window_size > window_max: # Se la finestra corrente è maggiore della finestra massima consentita
        
        
        p1_index = significant_points[0][0] # Start index
        p2_index = significant_points[0][1] # End index
        
        # Calcolo proporzioni di riduzione finestra (converge verso il massimo della deviazione)
        max_std_index = np.argmax(accSVM[p1_index:p2_index])+p1_index
        
        for i in range(p1_index,p2_index): # Cerca di centrare sulla prima di intersezione dell'accelerazione con il threshold
            if accSVM[i] >= threshold:
                max_std_index = i
                #print("Trovata intersezione threshold nella finestra ridotta")
                break
        
        p1_std_size_time = (max_std_index - p1_index)/sample_rate # Tempo dall'inizio della finestra al massimo della deviazione standard
        p2_std_size_time = (p2_index-max_std_index)/sample_rate # Tempo dal massimo della deviazione standard e fine della finesta


        p1_x_proportion = (p1_std_size_time*window_max)/curr_window_size
        p2_x_proportion = (p2_std_size_time*window_max)/curr_window_size


        win_to_reduce = curr_window_size-window_max

        p1_to_reduce = (win_to_reduce*p1_x_proportion)/window_max
        p2_to_reduce = (win_to_reduce*p2_x_proportion)/window_max
        
        p1_index += math.ceil(p1_to_reduce*sample_rate)
        p2_index -= math.ceil(p2_to_reduce*sample_rate)


        reduced_points = [(p1_index, p2_index)]


    threshold_x = None
    if len(reduced_points) > 0:
        for i in range(len(accSVM[reduced_points[0][0]:reduced_points[0][1]])):
            if accSVM[reduced_points[0][0]+i] >= threshold:  # Check for significant increase
                threshold_x = (reduced_points[0][0]+i)/sample_rate
                break
    elif len(significant_points) > 0:
        for i in range(len(accSVM[significant_points[0][0]:significant_points[0][1]])):
            if accSVM[significant_points[0][0]+i] >= threshold:  # Check for significant increase
                threshold_x = (significant_points[0][0]+i)/sample_rate
                break
    else:
        for i in range(len(accSVM)):
            if accSVM[i] >= threshold:  # Check for significant increase
                threshold_x = i/sample_rate
                break
    
    plot=threshold_x
    plot=True # Override
    zoomed_graph=False
    
    '''
    if plot and 1==0:
        plt.figure(figsize=(12,8))
        
        plt.subplot(2, 1, 1)
        plt.plot(time, accSVM, label='Acceleration')
        plt.title(f'Original Signal (File: {file_name})')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (g)')
        plt.legend()
        
        plt.subplot(1, 1, 1)
        plt.plot(time[:len(std_values)], std_values, label='Standard Deviation', color='orange')
        plt.axhline(baseline, color='black', linestyle='--', label='Baseline')
        plt.title('Standard Deviation')
        plt.xlabel('Time (s)')
        plt.ylabel('Standard Deviation')
        plt.legend()
        plt.show()

    '''

    
    if plot:

        plt.figure(figsize=(12, 8))

        # Original Acceleration Graph
        if zoomed_graph:
            plt.subplot(4, 1, 1)
        else:
            plt.subplot(2, 1, 1)
        plt.plot(time, accSVM, label='AccSVM (g)')
        for start, end in significant_points:
            plt.axvline(x=time[start], color='green', linestyle='--', label='Significant Start' if start == significant_points[0][0] else "")
            plt.axvline(x=time[end], color='red', linestyle='--', label='Significant End' if end == significant_points[0][1] else "")
            plt.axhline(y=threshold, color='black', linestyle='dotted', label='Threshold')

            if threshold_x != None:
                plt.axvline(x=threshold_x, color='black', linewidth=2, linestyle='dotted', label='Threshold V')
                plt.plot(threshold_x,threshold,'ro', label='Th intersect') 
        if len(reduced_points) > 0:

            for start, end in reduced_points:
                plt.axvline(x=time[start], color='grey', linestyle='--', label='Reduced Start')
                plt.axvline(x=time[end], color='purple', linestyle='--', label='Reduced End')
        
        plt.title(f'Original Signal (File: {file_name})')
        plt.xlabel('Time (s)')
        plt.ylabel('AccSVM')
        plt.legend()


        # Significant Parts Graph
        if zoomed_graph:
            plt.subplot(4, 1, 2)
        else:
            plt.subplot(2, 1, 2)
        
        plt.plot(time[:len(std_values)], std_values, label='Standard Deviation', color='orange')
        plt.axhline(baseline, color='black', linestyle='--', label='Baseline')

        for start, end in significant_points:
            plt.axvline(x=time[start], color='green', linestyle='--', label='Significant Start' if start == significant_points[0][0] else "")
            plt.axvline(x=time[end], color='red', linestyle='--', label='Significant End' if end == significant_points[0][1] else "")

        if len(reduced_points) > 0:
    
            for start, end in reduced_points:
                plt.axvline(x=time[start], color='grey', linestyle='--', label='Reduced Start' if start == reduced_points[0][0] else "")
                plt.axvline(x=time[end], color='purple', linestyle='--', label='Reduced End' if end == reduced_points[0][1] else "")
            
        plt.title('Standard Deviation')
        plt.xlabel('Time (s)')
        plt.ylabel('Standard Deviation')
        plt.legend()


        # Zoomed Parts Graph

        if len(reduced_points) > 0 and zoomed_graph:

            for start, end in reduced_points:
                
                plt.subplot(4, 1, 3)
                plt.plot(time[:len(accSVM[reduced_points[0][0]:reduced_points[0][1]])], accSVM[reduced_points[0][0]:reduced_points[0][1]], label='AccSVM', color='blue')
                plt.title('Zoomed Acceleration Region')
                plt.xlabel('Time (s)')
                plt.ylabel('Standard Deviation')
                plt.legend()

            for start, end in reduced_points:
                
                plt.subplot(4, 1, 4)
                plt.plot(time[:len(std_values[reduced_points[0][0]:reduced_points[0][1]])], std_values[reduced_points[0][0]:reduced_points[0][1]], label='Standard Deviation', color='orange')
                plt.axhline(baseline, color='black', linestyle='--', label='Baseline')
                plt.title('Zoomed Standard Deviation Region')
                plt.xlabel('Time (s)')
                plt.ylabel('Standard Deviation')
                plt.legend()


        plt.tight_layout()
        plt.show()

    if len(reduced_points) > 0:
        start = reduced_points[0][0]
        end = reduced_points[0][1]
        avg_detection_time.append(end-start)
        
        return data[start:end]    
    else:

        if len(significant_points) > 0:
           start = significant_points[0][0]
           end = significant_points[0][1]
           avg_detection_time.append(end-start)
           return data[start:end]
        else:
            avg_detection_time.append(len(data))
            return data


# Funzione per limitare i dati alla durata massima
def limit_data_to_duration(data, max_duration=2400): # Limitato a 12 secondi (2400 letture ogni 5 ms (200 Hz))
    return data[:max_duration]  # Limita la durata a max_duration campioni

# Funzione per estrarre le feature da un file di dati
def extract_features_from_file(data):

    accel_data = data[:, :3]  # Accelerazione
    rot_data = data[:, 3:6]   # Rotazione
    
    mean = np.mean(data, axis=0) # Media
    std_dev = np.std(data, axis=0) # Deviazione standard
    min_val = np.min(data, axis=0) # Valore minimo
    max_val = np.max(data, axis=0) # Valore massimo
    range_val = max_val - min_val # Range valore


    sum_accel = np.sum(np.linalg.norm(accel_data, axis=1)) # Norma 2 - accelerazione
    sum_rot = np.sum(np.linalg.norm(rot_data, axis=1)) # Norma 2 - rotazione

    # Calcola il tempo relativo in millisecondi
    num_samples = accel_data.shape[0] # Numero di campioni
    relative_time = np.arange(0, num_samples * 5, 5)  # Ogni campione è a 5 ms di distanza

    # Aggiungi una statistica del tempo (come la media e deviazione standard) - rivedere perché non ha senso (Il tempo medio in questo caso sarà sempre 5ms)
    mean_time = np.mean(relative_time) # Tempo medio
    std_dev_time = np.std(relative_time) # Deviazione standard del tempo

    # Costruisci le feature come concatenazione
    #features = np.concatenate([x,y,z,gx,gy,gz,[sum_accel, sum_rot]) # Features

    features = np.concatenate([mean, std_dev, range_val, [sum_accel, sum_rot, mean_time, std_dev_time]]) # Features
    
    return features

# ----- NEW DEVICE FUNCTION ------

def load_device_data_and_extract_features(folder_path):

    data = []
    labels = []
    file_paths = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if file_path.endswith('.txt'):

            # CODICE
            
            with open(file_path, 'r', encoding='ascii', errors='ignore') as f:
                text = f.read().strip()

            cleaned_data = ast.literal_eval(text)

            if len(cleaned_data) == 300:

                fileData = np.array(cleaned_data)

                features = extract_features_from_file(fileData)
                
                #if features.ndim == 1:
                #    features = features.reshape(1, -1)

                data.append(features)

                label = 1 if file_name.startswith('F') else 0
                labels.append(label)
                file_paths.append(file_path)
            else:
                print("\nSkipped...")
        
    return np.array(data), np.array(labels), file_paths





# ----- END NEW DEVICE FUNCTION -----





# Funzione per leggere i file e calcolare le feature
def load_fall_data_and_extract_features(directory, max_duration=2400, window_size_test=1):
    
    data = []
    labels = []
    file_paths = []

    
    for subject_folder in os.listdir(directory): 
        folder_path = os.path.join(directory, subject_folder)
        
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                if file_name.endswith('.txt') and "Readme" not in file_name:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    cleaned_data = []
                    for line in lines:
                        cleaned_line = line.strip().replace(';', '')
                        if cleaned_line:
                            try:
                                values = list(map(float, cleaned_line.split(',')))
                                cleaned_data.append(values[:6])
                            except ValueError as e:
                                print(f'Error converting line to float: {cleaned_line}. Error: {e}')
                    
                    if cleaned_data:
                        cleaned_data = np.array(cleaned_data)
                        
                        limited_data = limit_data_to_duration(cleaned_data, max_duration=max_duration)
                        
                        
                        limited_data2 = analyze_acceleration(file_name, limited_data, window_max=window_size_test)
                        
                        if limited_data2 is None:
                            limited_data2 = limited_data
                        limited_data2 = convert_all_data_to_international_system(limited_data2)
                        
                        features = extract_features_from_file(limited_data2)  
                        data.append(features)
                        label = 1 if file_name.startswith('F') else 0
                        labels.append(label)
                        file_paths.append(file_path)
    print("Tempo finestra: ")
    print((sum(avg_detection_time)/len(avg_detection_time))/200)
    return np.array(data), np.array(labels), file_paths


def save_model(model, filename):
    global save
    if save:
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")


def load_model(filename):
    global save
    if save:
        if os.path.exists(filename):
            return joblib.load(filename)
        else:
            print(f"Model file {filename} does not exist.")
            return None

# Funzione per classificare nuovi dati da un secondo percorso
def classify_new_data(directory, model, scaler, max_duration=2400):
    new_data = []
    file_names = []
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []

        print(f"Scanning directory: {directory}")

    for subject_folder in os.listdir(directory):
        folder_path = os.path.join(directory, subject_folder)

        if os.path.isdir(folder_path):
            print(f"Found folder: {subject_folder}")
            for file_name in os.listdir(folder_path):
                try:
                    file_path = os.path.join(folder_path, file_name)
                    
                    if file_name.endswith('.txt') and "Readme" not in file_name:
                        print(f"Processing file: {file_name}")
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        
                        cleaned_data = []
                        for line in lines:
                            cleaned_line = line.strip().replace(';', '')
                            if cleaned_line:
                                try:
                                    values = list(map(float, cleaned_line.split(',')))
                                    cleaned_data.append(values[:6])
                                except ValueError as e:
                                    print(f'Error converting line to float: {cleaned_line}. Error: {e}')
                        
                        if cleaned_data:
                            cleaned_data = np.array(cleaned_data)
                            limited_data = limit_data_to_duration(cleaned_data, max_duration=max_duration)
                            limited_data2 = analyze_acceleration_w_threshold(file_name, limited_data, window_max=1.5)
                            features = extract_features_from_file(limited_data2)
                            new_data.append(features)
                            file_names.append(file_name)
                except Exception as e:
                    print("Exception, no data valid", e)
            else:
                print(f"No valid data in file: {file_name}")
        else:
            print(f"Skipping non-folder: {subject_folder}")

    if not new_data:
        print("No data found in the second directory.")
        return []

    new_data = np.array(new_data)
    print(f"Shape of new data before scaling: {new_data.shape}")

    new_data = scaler.transform(new_data)
    predictions = model.predict(new_data)
    
    return list(zip(file_names, predictions))

# Classe per gestire il caricamento e la classificazione con il modello
class RandomTree:
    model_filename = "random_forest_model.joblib"
    scaler_filename = "scaler.joblib"
    
    def __init__(self, max_duration=2400):
        self.max_duration = max_duration
        self.load()

    def load(self):
        self.model = load_model(self.model_filename)
        if self.model is None:
            print(f"Model file {self.model_filename} does not exist.")
        self.scaler = joblib.load(self.scaler_filename) if os.path.exists(self.scaler_filename) else None

    def load_file(self, file_path):
        self.file = file_path

    def classify(self):
        return classify_single_file(self.file, self.model, self.scaler, self.max_duration)


# TEST - ALLENATO SU DATASET PERSONALE, TEST SU SISFALL

'''
modelName = "Trained/Device/Models/model_KNN_1500.joblib"
scalerName = "Trained/Device/Scalers/scaler_1500.joblib"   
model = joblib.load(modelName)
scaler = joblib.load(scalerName)


res = classify_new_data("SIS/", model, scaler, 2400)
print(res)


exit()

'''

# FINE TEST - ALLENATO SU DATASET PERSONALE, TEST SU SISFALL


# TEST - DECISIONE BASATA SU SOFT VOTING, TEST SU SISFALL

'''

deviceDirectory = "DEVICE_DATASET/"
device = False

window_size_test = 1.5

if device:           
    data, labels, file_paths = load_device_data_and_extract_features(deviceDirectory)
else:
    data, labels, file_paths = load_fall_data_and_extract_features(directory, 2400, window_size_test)
#print(labels, file_paths)

X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(data, labels, file_paths, test_size=0.3, random_state=42)

        
# Normalizzazione
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


rfModel = RandomForestClassifier(n_estimators=100, random_state=42)

knnModel = KNeighborsClassifier(n_neighbors=3)

#choosenModel = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)  # Radial basis kernel (utile per ridurre la dimensione del problema)
svmModel = SVC(C=1, gamma='scale', class_weight='balanced', random_state=42, probability=True)

base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
ebtModel = BaggingClassifier(estimator=base_estimator, n_estimators=50, random_state=42)


voting_clf = VotingClassifier(
    estimators=[
        ('knn', knnModel),
        ('ebt', ebtModel),
        ('rf', rfModel),
        ('svm', svmModel)
    ],
    voting='soft',  # Use 'hard' for majority vote if you prefer
    weights=[4, 4, 4, 1]
)

#voting_clf.fit(X_train, y_train)
#y_pred = voting_clf.predict(X_test)

# Define a parameter grid for weights
param_grid = {
    'weights': [
        [w_knn, w_ebt, w_rf, w_svm]
        for w_knn in range(1, 5)
        for w_ebt in range(1, 5)
        for w_rf in range(1, 5)
        for w_svm in range(1, 5)
    ]
}
grid_search = GridSearchCV(voting_clf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best weights found
best_weights = grid_search.best_params_['weights']
print("Best weights found:", best_weights)

# Use the best estimator from grid search
best_voting_clf = grid_search.best_estimator_
y_pred = best_voting_clf.predict(X_test)

# Salva il modello e lo scaler

text_win_size = int(window_size_test*1000)
model = "bestSoftVoting"
if device:
    model_filename = "Trained/Device/Models/model_%s_%s.joblib" % (model, text_win_size)
else:
    model_filename = "Trained/Models/model_%s_%s.joblib" % (model, text_win_size)
if not os.path.exists(model_filename):
    save_model(best_voting_clf, model_filename)
    #print("%s model saved." % model)


if device:
    scaler_filename = "Trained/Device/Scalers/scaler_%s.joblib" % text_win_size
else:
    scaler_filename = "Trained/Scalers/scaler_%s.joblib" % text_win_size
if not os.path.exists(scaler_filename):
    joblib.dump(scaler, scaler_filename)
    #print("Scaler saved: for window %s" % window_size_test)


# Valutazione del modello
cm = confusion_matrix(y_test, y_pred)

# Estrai i valori dalla Confusion Matrix
TN, FP, FN, TP = cm.ravel()


accuracy = accuracy_score(y_test, y_pred)
print(f"\nPunteggio di accuratezza: {accuracy*100:.2f}%")
print("----------")
'''
# FINE TEST - DECISIONE BASTA SU SOFT VOTING, TEST SU SISFALL


# INIZIO TEST - TRAINING SU ENTRAMBI I DATASET

deviceDirectory = "DEVICE_DATASET/"
device = False

window_size_to_test = [1.5]
models_to_choose = ["RF", "KNN", "SVM", "EBT"]

for window_size_test in window_size_to_test:
    print("-----------")
    for model in models_to_choose:
        avg_detection_time = []
        print("Modello: "+model)
        print("Finestra: "+str(window_size_test)+"s")


        
        dataDevice, labelsDevice, file_pathsDevice = load_device_data_and_extract_features(deviceDirectory)

        data, labels, file_paths = load_fall_data_and_extract_features(directory, 2400, window_size_test)

        data = np.concatenate([dataDevice, data], axis=0)
        labels = np.concatenate([labelsDevice, labels], axis=0)
        file_paths.extend(file_pathsDevice)

        #print(labels, file_paths)

        
        X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(data, labels, file_paths, test_size=0.3, random_state=42)

                
        # Normalizzazione
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Scelta del modello
        if model == 'RF': # Random Forest
            choosenModel = RandomForestClassifier(n_estimators=100, random_state=42)

        if model == 'KNN':
            choosenModel = KNeighborsClassifier(n_neighbors=3)

        if model == 'SVM':
            #choosenModel = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)  # Radial basis kernel (utile per ridurre la dimensione del problema)
            choosenModel = SVC(C=1, gamma='scale', class_weight='balanced', random_state=42)

        if model == 'EBT':  # Ensemble Bagging Trees
            base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            choosenModel = BaggingClassifier(estimator=base_estimator, n_estimators=50, random_state=42)

        
        
        choosenModel.fit(X_train, y_train)
        
        # Salva il modello e lo scaler

        text_win_size = int(window_size_test*1000)

        
        model_filename = "Trained/DeviceAndSisfall/Models/model_%s_%s.joblib" % (model, text_win_size)
        if not os.path.exists(model_filename):
            save_model(choosenModel, model_filename)
            #print("%s model saved." % model)


        scaler_filename = "Trained/DeviceAndSisfall/Scalers/scaler_%s.joblib" % text_win_size
        if not os.path.exists(scaler_filename):
            joblib.dump(scaler, scaler_filename)
            #print("Scaler saved: for window %s" % window_size_test)
        

        # Predizione e valutazione sul test set
        y_pred = choosenModel.predict(X_test)

        # Valutazione del modello
        cm = confusion_matrix(y_test, y_pred)

        # Estrai i valori dalla Confusion Matrix
        TN, FP, FN, TP = cm.ravel()

        print("TN: ", TN)
        print("FP: ", FP)
        print("FN: ", FN)
        print("TP: ", TP)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nPunteggio di accuratezza: {accuracy*100:.2f}%")
        print("----------")


# FINE TEST - TRAINING SU ENTRAMBI I DATASET
'''
deviceDirectory = "DEVICE_DATASET/"
device = False
window_size_to_test = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5]

window_size_to_test = [1.5]
models_to_choose = ["RF", "KNN", "SVM", "EBT"]

for window_size_test in window_size_to_test:
    print("-----------")
    for model in models_to_choose:
        avg_detection_time = []
        print("Modello: "+model)
        print("Finestra: "+str(window_size_test)+"s")


        if device:           
            data, labels, file_paths = load_device_data_and_extract_features(deviceDirectory)
        else:
            data, labels, file_paths = load_fall_data_and_extract_features(directory, 2400, window_size_test)
        #print(labels, file_paths)

        
        X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(data, labels, file_paths, test_size=0.3, random_state=42)

                
        # Normalizzazione
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Scelta del modello
        if model == 'RF': # Random Forest
            choosenModel = RandomForestClassifier(n_estimators=100, random_state=42)

        if model == 'KNN':
            choosenModel = KNeighborsClassifier(n_neighbors=3)

        if model == 'SVM':
            #choosenModel = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)  # Radial basis kernel (utile per ridurre la dimensione del problema)
            choosenModel = SVC(C=1, gamma='scale', class_weight='balanced', random_state=42)

        if model == 'EBT':  # Ensemble Bagging Trees
            base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            choosenModel = BaggingClassifier(estimator=base_estimator, n_estimators=50, random_state=42)

        
        
        choosenModel.fit(X_train, y_train)
        
        # Salva il modello e lo scaler

        text_win_size = int(window_size_test*1000)

        if device:
            model_filename = "Trained/Device/Models/model_%s_%s.joblib" % (model, text_win_size)
        else:
            model_filename = "Trained/Models/model_%s_%s.joblib" % (model, text_win_size)
        if not os.path.exists(model_filename):
            save_model(choosenModel, model_filename)
            #print("%s model saved." % model)

        
        if device:
            scaler_filename = "Trained/Device/Scalers/scaler_%s.joblib" % text_win_size
        else:
            scaler_filename = "Trained/Scalers/scaler_%s.joblib" % text_win_size
        if not os.path.exists(scaler_filename):
            joblib.dump(scaler, scaler_filename)
            #print("Scaler saved: for window %s" % window_size_test)
        

        # Predizione e valutazione sul test set
        y_pred = choosenModel.predict(X_test)

        # Valutazione del modello
        cm = confusion_matrix(y_test, y_pred)

        # Estrai i valori dalla Confusion Matrix
        TN, FP, FN, TP = cm.ravel()

        print("TN: ", TN)
        print("FP: ", FP)
        print("FN: ", FN)
        print("TP: ", TP)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nPunteggio di accuratezza: {accuracy*100:.2f}%")
        print("----------")
'''


