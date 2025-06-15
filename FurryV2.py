import json
import numpy as np
from scipy import signal
import os
import pickle
from pathlib import Path

class EEGProcessor:
    def __init__(self, sampling_rate=128, segment_length=256):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.frequency_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
    def load_json_data(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            print(f"Załadowano dane z pliku: {file_path}")
            return data
        except Exception as e:
            print(f"Błąd podczas ładowania pliku {file_path}: {str(e)}")
            return None
    
    def apply_bandpass_filter(self, data, low_freq, high_freq):
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Filtr Butterwortha 4. rzędu
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data)
        
        return filtered_data
    
    def segment_data(self, data):
        data = np.array(data)
        segments = []
        num_full_segments = len(data) // self.segment_length

        for i in range(num_full_segments):
            start_idx = i * self.segment_length
            end_idx = start_idx + self.segment_length
            segments.append(data[start_idx:end_idx])
        
        remaining_data = len(data) % self.segment_length
        if remaining_data > 0:
            start_idx = len(data) - self.segment_length
            segments.append(data[start_idx:])
        
        return segments
    
    def extract_mean_power_from_segment(self, segment_data):
        power_data = {}
        
        eeg_channels = {k: v for k, v in segment_data.items() if k != 'adhd_class'}
        
        for channel_name, channel_data in eeg_channels.items():
            if not isinstance(channel_data, (list, np.ndarray)):
                continue
                
            channel_data = np.array(channel_data)
            power_data[channel_name] = {}
            
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                filtered_signal = self.apply_bandpass_filter(channel_data, low_freq, high_freq)
                mean_power = np.mean(filtered_signal ** 2)
                power_data[channel_name][band_name] = mean_power
        
        return power_data
    
    def create_feature_vector(self, power_data, adhd_label):
        features = []
        feature_names = []
        sorted_channels = sorted(power_data.keys())
        
        for channel in sorted_channels:
            for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                if band in power_data[channel]:
                    mean_power = power_data[channel][band]
                    features.append(mean_power)
                    feature_names.append(f"{channel}_{band}_mean_power")
        
        result = {
            'features': np.array(features),
            'feature_names': feature_names,
            'adhd_label': adhd_label,
            'power_data': power_data
        }
        
        return result
    
    def process_single_file(self, input_file_path, output_dir, adhd_label):
        raw_data = self.load_json_data(input_file_path)
        if raw_data is None:
            return 0
        file_stem = Path(input_file_path).stem
        eeg_channels = {k: v for k, v in raw_data.items() if k != 'adhd_class'}
        
        if not eeg_channels:
            print("Brak kanałów EEG w pliku")
            return 0
        first_channel = list(eeg_channels.keys())[0]
        first_channel_data = eeg_channels[first_channel]
        segments_from_first_channel = self.segment_data(first_channel_data)
        num_segments = len(segments_from_first_channel)
        
        print(f"Dzielenie na {num_segments} segmentów po {self.segment_length} próbek")
        
        processed_segments = 0
        for segment_idx in range(num_segments):
            print(f"Przetwarzanie segmentu {segment_idx + 1}/{num_segments}")
            segment_data = {}
            for channel_name, channel_data in eeg_channels.items():
                channel_segments = self.segment_data(channel_data)
                if segment_idx < len(channel_segments):
                    segment_data[channel_name] = channel_segments[segment_idx]
            if segment_data:
                power_data = self.extract_mean_power_from_segment(segment_data)
                processed_segment = self.create_feature_vector(power_data, adhd_label)
                print(f"Segment {segment_idx + 1}: etykieta = {processed_segment['adhd_label']}")
                output_filename = f"{file_stem}_part_{segment_idx + 1}.pkl"
                output_path = Path(output_dir) / output_filename
                
                with open(output_path, 'wb') as f:
                    pickle.dump(processed_segment, f)
                
                print(f"Zapisano segment: {output_filename}")
                processed_segments += 1
        
        return processed_segments
    
    def get_user_label_input(self):
        while True:
            response = input("\nCzy wszystkie pliki w tym katalogu są od osób z ADHD? (tak/nie): ").strip().lower()
            print(f"Twoja odpowiedź: '{response}'") 
            if response in ['tak', 't', 'yes', 'y']:
                print("Ustawiono etykietę: ADHD (1)")
                return 1
            elif response in ['nie', 'n', 'no']:
                print("Ustawiono etykietę: Kontrola (0)")
                return 0
            else:
                print("Proszę odpowiedzieć 'tak' lub 'nie'")
    
    def process_directory(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        input_path = Path(input_dir)
        json_files = list(input_path.glob("*.json"))
        
        if not json_files:
            print(f"Nie znaleziono plików JSON w katalogu: {input_dir}")
            return
        
        print(f"Znaleziono {len(json_files)} plików JSON")
        adhd_label = self.get_user_label_input()
        print(f"\nUżywana etykieta dla wszystkich segmentów: {adhd_label}") 
        
        total_segments = 0
        
        for i, json_file in enumerate(json_files):
            print(f"\n=== Przetwarzanie pliku {i+1}/{len(json_files)}: {json_file.name} ===")
            
            segments_processed = self.process_single_file(str(json_file), output_dir, adhd_label)
            total_segments += segments_processed
            
            print(f"Przetworzono {segments_processed} segmentów z pliku {json_file.name}")
        label_description = "ADHD" if adhd_label == 1 else "Control"
        metadata = {
            'segment_length': self.segment_length,
            'sampling_rate': self.sampling_rate,
            'frequency_bands': self.frequency_bands,
            'total_files_processed': len(json_files),
            'total_segments_created': total_segments,
            'feature_description': 'Mean power for each frequency band per channel',
            'dataset_label': adhd_label,
            'dataset_label_description': label_description,
            'label_encoding': {'0': 'Control', '1': 'ADHD'}
        }
        
        metadata_path = Path(output_dir) / "processing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n=== PODSUMOWANIE ===")
        print(f"Przetworzono plików: {len(json_files)}")
        print(f"Utworzono segmentów: {total_segments}")
        print(f"Długość segmentu: {self.segment_length} próbek")
        print(f"Etykieta datasetu: {label_description} ({adhd_label})")
        print(f"Zapisano metadane: {metadata_path}")

def main():
    processor = EEGProcessor(sampling_rate=250, segment_length=256) 
    input_directory = "Control_filtered"  
    output_directory = "FurryV2" 
    print("=== PROCESOR DANYCH EEG - WERSJA SEGMENTOWANA ===")
    print(f"Katalog wejściowy: {input_directory}")
    print(f"Katalog wyjściowy: {output_directory}")
    print(f"Długość segmentu: {processor.segment_length} próbek")
    print(f"Pasma częstotliwościowe: {processor.frequency_bands}")
    print("Cechy: Średnia moc dla każdego pasma częstotliwościowego")
    print("Program zapyta o etykietę dla całego datasetu")
    print("Etykiety: 1 = ADHD, 0 = Kontrola")
    processor.process_directory(input_directory, output_directory)
    
    print("\n=== PRZETWARZANIE ZAKOŃCZONE ===")

if __name__ == "__main__":
    main()
