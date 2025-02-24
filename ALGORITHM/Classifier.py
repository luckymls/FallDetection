import os, ast
import numpy as np
import joblib  # Per salvare e caricare il modello
import random, math, requests, json, hashlib

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


directory = "SIS/"



# Flag per decidere se salvare il modello
save = True
flag=True



def visualize_data(data, threshold, index=None, sampling_frequency=200, title=None):

    print(data)
    accel_data = [row[:3] for row in data]
    accel_data = np.array(accel_data)
    magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
    magnitude = convert_data_to_g_acc(magnitude)
    time = np.arange(len(magnitude)) / sampling_frequency

    # Plot magnitude over time
    plt.figure(figsize=(10, 6))
    plt.plot(time, magnitude, label="Acceleration Magnitude", color="blue")
    
    # Add labels, legend, and title
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration Magnitude (G)")
    plt.title(title if title else "Threshold Visualization")
    plt.legend()
    plt.grid()
    #plt.pause(0.1)
    plt.show()


def max_sum_vector_magnitude(accel_data):

    
    norms = np.linalg.norm(accel_data, axis=1)  # Calcola la norma per ogni riga
    max_value = np.max(norms)                   # Trova il valore massimo
    return np.argmax(norms)                # Trova l'indice della riga con il massimo valore
    

def convert_data_to_g_acc(accel_data):

    accel_scale = 1/9.81
    # Convert to G values
    g_accel = accel_scale * accel_data
    return g_accel



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




# Funzione per caricare il modello
def load_model(filename):
    global save
    if save:
        if os.path.exists(filename):
            return joblib.load(filename)
        else:
            print(f"Model file {filename} does not exist.")
            return None

def model_name(model, window_size, useDevice=False):
    window = int(window_size*1000)
    if useDevice:
        return "Trained/Device/Models/model_%s_%s.joblib" % (model, window)
    return "Trained/DeviceAndSisfall/Models/model_%s_%s.joblib" % (model, window) # OVERRIDE
    return "Trained/Models/model_%s_%s.joblib" % (model, window)

def scaler_name(window_size, useDevice=False):
    window = int(window_size*1000)
    if useDevice:
        return "Trained/Device/Scalers/scaler_%s.joblib" % (window)
    return "Trained/DeviceAndSisfall/Scalers/scaler_%s.joblib" % (window)
    return "Trained/Scalers/scaler_%s.joblib" % (window)

# Funzione per classificare nuovi dati da un secondo percorso
def classify_data(data, model, window_size, useDevice=False):
    
    model_filename = model_name(model, window_size, useDevice)
    #print(model_filename)
    scaler_filename = scaler_name(window_size, useDevice)
    #print(scaler_filename)
    model = load_model(model_filename)
    scaler = joblib.load(scaler_filename)
    data = np.array(data)
    
    features = extract_features_from_file(data)
     # Assicurati che features sia 2D
    if features.ndim == 1:
        features = features.reshape(1, -1)
    scaled_data = scaler.transform(features)
    predictions = model.predict(scaled_data)
    return predictions



def convertToData(text):

    jsData = json.loads(text)
    rawData = jsData["data"]
    accX = rawData["accX"]
    accY = rawData["accY"]
    accZ = rawData["accZ"]
    gyroX = rawData["gyroX"]
    gyroY = rawData["gyroY"]
    gyroZ = rawData["gyroZ"]
    
    data = list(zip(accX, accY, accZ, gyroX, gyroY, gyroZ))

    return data

def get_md5_of_string(input_string): 
    return hashlib.md5(input_string.encode()).hexdigest()

directory = "SIS/"

latestDataMD5 = 0

personal_dataset = "DEVICE_DATASET/"



def classify_device_data_on_SisFall(features, model, window_size):
    
    model_filename = model_name(model, window_size, False)
    scaler_filename = scaler_name(window_size, False)
    
    model = load_model(model_filename)
    scaler = joblib.load(scaler_filename)
    #data = np.array(data)
    
    
     # Assicurati che features sia 2D
    if features.ndim == 1:
        features = features.reshape(1, -1)
    scaled_data = scaler.transform(features)
    predictions = model.predict(scaled_data)
    return predictions

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
                pass
                #print("\nSkipped...")
        
    return np.array(data), np.array(labels), file_paths


data, labels, fall_path = load_device_data_and_extract_features(personal_dataset)
models = ["KNN", "EBT", "RF", "SVM"]#, "softVoting"]
#models = ["bestSoftVoting"]
for model in models:
    print(model)
    i=0
    correct = 0
    total = 0
    correct_fall = 0
    dataset_fall = 0
    dataset_non_fall = 0
    for feat in data:
        predict = classify_device_data_on_SisFall(feat, model, 1.5)
        #print("Risultato: %s, Predizione: %s, Reale: %s" % (predict == labels[i], predict, labels[i]))
        correct += predict == labels[i]
        if labels[i] == 1 and predict == labels[i]:
            correct_fall +=1
        if labels[i] == 1:
            dataset_fall +=1
        if labels[i] == 0:
            dataset_non_fall += 1
        total += 1
        i+=1

    correct_non_fall = correct-correct_fall
    print("Cadute: %s/%s" % (correct_fall, dataset_fall))
    print("Non cadute: %s/%s " % (correct_non_fall, dataset_non_fall))
    print("Totali: ", total)

'''
is_fall = False
skip = False
useDevice = True
while True:
    text = requests.get("SERVER URL").text
    datamd5 = get_md5_of_string(text)

     

    if datamd5 != latestDataMD5: # Contenuto nuovo da classificare

        
        print("NEW DATA: \n")
        data = convertToData(text)
        models = ["KNN", "SVM", "EBT", "RF"]
        wSize = 1.5
        
        for model in models:

            
            prediction = classify_data(data, model, wSize, useDevice)
            
            if prediction == 1:
                #print("Modello %s" % model)
                print("%s: Caduta" % model)
                
            elif prediction == 0:
                #print("Modello %s" % model)
                print("%s: Non caduta" % model)
                
        visualize_data(data, 1.66)
        latestDataMD5 = datamd5
        activity_map = {
                        "0": "Walking",
                        "1": "Jumping",
                        "2": "Sitting",
                        "3": "Stairs",
                        "4": "Running",
                        "5": "Frontal_Fall",
                        "6": "Lateral_Fall",
                        "7": "Backward_Fall"
        }

        user_fall_status = input("\nIs this a fall event?\n[0] Non Fall\n[1] Fall\n[2] Skip\n> ")
        #user_fall_status = "0"
        if user_fall_status == "2":
            skip = True
        else:
            
            user_activity = input("\nSelect the activity:\n"
                          "[0] Walking\n"
                          "[1] Jumping\n"
                          "[2] Sitting\n"
                          "[3] Stairs\n"
                          "[4] Running\n"
                          "[5] Frontal_Fall\n"
                          "[6] Lateral_Fall\n"
                          "[7] Backward_Fall\n> ")
            
            #user_activity = "1"
            if user_activity in activity_map:
                prefix = "F_" if user_fall_status == "1" else "D_"
                file_name = f"{prefix}{activity_map[user_activity]}_{latestDataMD5[:10]}.txt"
            else:
                print("Invalid activity selection.")

            
        if skip == False:
            file_path = os.path.join(personal_dataset, file_name)

            # Write data to file
            with open(file_path, "w") as file:
                file.write(str(data))

            print(f"Data written to {file_path}")
        else:
            print("\nSkipped.\n")
            skip=False

            
        print("-------------------------\n\n") 


'''
