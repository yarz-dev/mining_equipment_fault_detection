import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')

def dataset_path(*parts):
    return os.path.join(DATASETS_DIR, *parts)

# Raw datasets
RAW_DATASETS_DIR = dataset_path('raw')
RAW_THERMAL_DATASET_DIR = dataset_path('raw', 'thermal')
RAW_ACOUSTIC_DATASET_DIR = dataset_path('raw', 'acoustic')
RAW_VIBRATION_DATASET_DIR = dataset_path('raw', 'vibration')
RAW_TEMPERATURE_DATASET_DIR = dataset_path('raw', 'temperature')

# Cleaned datasets
CLEANED_DATASETS_DIR = dataset_path('cleaned')
CLEANED_THERMAL_DATASET_DIR = dataset_path('cleaned', 'thermal')
CLEANED_ACOUSTIC_DATASET_DIR = dataset_path('cleaned', 'acoustic')
CLEANED_VIBRATION_DATASET_DIR = dataset_path('cleaned', 'vibration')
CLEANED_TEMPERATURE_DATASET_DIR = dataset_path('cleaned', 'temperature')