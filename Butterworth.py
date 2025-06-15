import scipy.io
import json
import numpy as np
import os
import glob
from scipy.signal import butter, sosfiltfilt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.stats import median_abs_deviation

def konwertacja_mat_do_json(mat_file_path, json_output_path, is_adhd):
    nazwy_kanalow = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
    ]

    mat_dane = scipy.io.loadmat(mat_file_path)
    mat_dane_oczyszczona = {k: v for k, v in mat_dane.items() if not k.startswith('__')}
    finalne_dane = {}

    finalne_dane['adhd_label'] = 1 if is_adhd else 0
    finalne_dane['adhd_class'] = 'ADHD' if is_adhd else 'Control'

    for key, value in mat_dane_oczyszczona.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            num_channels = value.shape[1]
            for i in range(num_channels):
                if i < len(nazwy_kanalow):
                    channel_name = nazwy_kanalow[i]
                else:
                    channel_name = f"{key}_extra_{i+1}"
                finalne_dane[channel_name] = value[:, i].tolist()
        else:
            if isinstance(value, np.ndarray):
                finalne_dane[key] = value.tolist()
            else:
                finalne_dane[key] = value

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(finalne_dane, f, indent=4, ensure_ascii=False)

    return finalne_dane

def Filtr_butterwortha(dolna, gorna, czest_prob, rzad=4):

    nyquist_freq = czest_prob / 2.0
    low = dolna / nyquist_freq
    high = gorna / nyquist_freq
    sos = butter(rzad, [low, high], btype='bandpass', output='sos')
    return sos

def filter_channel(data, sos):

    try:
        filtered = sosfiltfilt(sos, data)
        return np.round(filtered, 2)
    except Exception as e:
        print(f"Błąd podczas filtrowania kanału: {e}")
        return data

def filter_all_channels(data_dict, channel_labels, lowcut, highcut, sampling_rate, order):

    sos = Filtr_butterwortha(lowcut, highcut, sampling_rate, order)

    filtered_data = data_dict.copy()
    for channel in channel_labels:
        if channel in filtered_data:
            filtered_data[channel] = filter_channel(np.array(filtered_data[channel]), sos).tolist()
    
    
    return filtered_data

def soft_clip(signal, threshold=7.0):
    mad = median_abs_deviation(signal)
    threshold_val = threshold * mad
    return threshold_val * np.tanh(signal/threshold_val)

def ICA(filtered_data, channel_labels, 
                          clip_threshold=7.0,
                          max_iter=4000, tol=1e-6):

    valid_channels = [ch for ch in channel_labels if ch in filtered_data]
    if len(valid_channels) < 2:
        print("ICA potrzebuje przynajminiej dwa kanały")
        return filtered_data.copy()
    signal_matrix = np.array([filtered_data[ch] for ch in valid_channels])

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(signal_matrix.T).T

    ica = FastICA(
        n_components=len(valid_channels),
        random_state=42,
        max_iter=max_iter,
        tol=tol,
        whiten='unit-variance'
    )
    try:

        sources = ica.fit_transform(data_scaled.T)
        reconstructed = ica.inverse_transform(sources).T

        reconstructed = scaler.inverse_transform(reconstructed.T).T

        cleaned_signals = []
        for ch_signal in reconstructed:
            clipped = soft_clip(ch_signal, threshold=clip_threshold)
            cleaned_signals.append(clipped)

        cleaned_data = {ch: cleaned_signals[i].tolist() 
            for i, ch in enumerate(valid_channels)}

        for ch in channel_labels:
            if ch not in cleaned_data and ch in filtered_data:
                cleaned_data[ch] = filtered_data[ch]      
        return cleaned_data
    except Exception as e:
            print(f"ICA fail")   

def batch_convert_mat_to_json(input_folder, output_folder, apply_filter=False, filter_params=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f" Created output folder: {output_folder}")
    mat_files = glob.glob(os.path.join(input_folder, "*.mat"))
    if not mat_files:
        print(f" No MAT files found in folder: {input_folder}")
        return
    print(f" Found {len(mat_files)} MAT files to convert:")
    for i, file_path in enumerate(mat_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    print("\n" + "="*60)
    print("ADHD CLASSIFICATION FOR ENTIRE FOLDER")
    print("="*60)
    print("ADHD?")
    print("  • YES (enter 'y', 'yes', '1', or 'adhd')")
    print("  • NO (enter 'n', 'no', '0', or 'control')")
    print("="*60)
    while True:
        user_input = input(f"Are ALL files in '{input_folder}' from ADHD children? (y/n): ").strip().lower()
        
        if user_input in ['y', 'yes', '1', 'adhd']:
            is_adhd = True
            label_text = "ADHD"
            break
        elif user_input in ['n', 'no', '0', 'control']:
            is_adhd = False
            label_text = "Control"
            break
        else:
            print(" Invalid input. Please enter 'y' (ADHD) or 'n' (Control)")
    
    if filter_params is None:
        filter_params = {
            'lowcut': 1,
            'highcut': 45,
            'sampling_rate': 128,
            'order': 4
        }
    
    print(f"\n  All files will be labeled as: {label_text}")
    if apply_filter:
        print(" Applying bandpass filter with parameters:")
        print(f"   - Lowcut: {filter_params['lowcut']} Hz")
        print(f"   - Highcut: {filter_params['highcut']} Hz")
        print(f"   - Sampling rate: {filter_params['sampling_rate']} Hz")
        print(f"   - Filter order: {filter_params['order']}")
    else:
        print(" No filtering will be applied")
    print(" Starting conversion...")
    
    converted_count = 0
    
    for mat_file_path in mat_files:
        filename = os.path.basename(mat_file_path)
        base_name = os.path.splitext(filename)[0]
        json_output_path = os.path.join(output_folder, f"{base_name}.json")
        
        try:
            data = konwertacja_mat_do_json(mat_file_path, json_output_path, is_adhd)
            if apply_filter:
                channel_labels = [
                     "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
                ]
                filtered_data = filter_all_channels(
                    data, 
                    channel_labels,
                    lowcut=filter_params['lowcut'],
                    highcut=filter_params['highcut'],
                    sampling_rate=filter_params['sampling_rate'],
                    order=filter_params['order']
                )
                
                cleaned_data = ICA(filtered_data,channel_labels)
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
                
                print(f" Converted & filtered: {filename} → {base_name}.json (Label: {label_text})")
            else:
                print(f" Converted: {filename} → {base_name}.json (Label: {label_text})")
                
            converted_count += 1
        except Exception as e:
            print(f" Error converting {filename}: {str(e)}")
    
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f" Total files found: {len(mat_files)}")
    print(f" Successfully converted: {converted_count}")
    print(f" Errors: {len(mat_files) - converted_count}")
    print(f"  All files labeled as: {label_text}")
    if apply_filter:
        print(" Filtering was applied to all EEG channels")
    else:
        print(" No filtering was applied")
    print(f" Output folder: {output_folder}")
    print("="*60)

def main():

    input_folder = input(" Path with MAT ").strip()
    if not input_folder:
        input_folder = "."  
    if not os.path.exists(input_folder):
        print(f"Input folder does not exist: {input_folder}")
        return
    output_folder = input(" Path for json ").strip()
    if not output_folder:
        output_folder = "converted_json"  
    apply_filter = input("\n Filter? (y/n): ").strip().lower() in ['y', 'yes']
    filter_params = None
    if apply_filter:
        print("\n  Params")
        lowcut = input(f"   - Lowcut  (def. 1 Hz): ").strip()
        highcut = input(f"   - Highcut  (def. 45 Hz): ").strip()
        sampling_rate = input(f"   - Sampling rate (def. 128 Hz): ").strip()
        order = input(f"   - Filter order (def. 4): ").strip()
        filter_params = {
            'lowcut': float(lowcut) if lowcut else 1,
            'highcut': float(highcut) if highcut else 45,
            'sampling_rate': float(sampling_rate) if sampling_rate else 128,
            'order': int(order) if order else 4
        }
    print(f"\n Starting conversion...")
    print(f"   Input folder: {input_folder}")
    print(f"   Output folder: {output_folder}")
    print(f"   Filtering: {'ON' if apply_filter else 'OFF'}")
    batch_convert_mat_to_json(input_folder, output_folder, apply_filter, filter_params)
if __name__ == "__main__":
    main()
