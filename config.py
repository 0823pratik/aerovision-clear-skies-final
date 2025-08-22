"""
AeroVision-GGM 2.0 - Technical Configuration
Focus: Solving air quality monitoring with hyperlocal accuracy
"""
from pathlib import Path
import pandas as pd
import numpy as np

# Base configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TARGET_CITIES = ['Gurugram', 'Gurgaon']

# Model configurations
MODEL_CONFIGS = {
    'ensemble': {
        'rf_estimators': 100,
        'xgb_estimators': 100,
        'weights': [0.4, 0.4, 0.2]
    },
    'lstm': {
        'sequence_length': 24,
        'epochs': 25,
        'batch_size': 32
    }
}

# SECTOR 49 PILOT WARD - Technical Implementation
PILOT_WARD_CONFIG = {
    'ward_name': 'Sector 49, Gurugram',
    'ward_bounds': {
        'lat_min': 28.4380,
        'lat_max': 28.4480,
        'lon_min': 28.0700,
        'lon_max': 77.0800
    },
    'population': 28000,
    'area_sq_km': 3.2,
    'iot_sensors': {
        'total_deployed': 25,
        'sensor_types': ['PMS5003', 'SDS011', 'BME280'],
        'data_frequency_minutes': 15,
        'deployment_pattern': 'grid_strategic'
    }
}

# TECHNICAL DIFFERENTIATION from existing solutions
TECHNICAL_ADVANTAGES = {
    'data_foundation': {
        'google_airview_records': 400000,
        'temporal_coverage_months': 18,
        'spatial_resolution_km': 0.1,
        'update_frequency_minutes': 15
    },
    'ai_capabilities': {
        'ensemble_models': ['RandomForest', 'XGBoost', 'LSTM'],
        'uncertainty_quantification': True,
        'forecast_horizon_hours': 24,
        'mae_performance': 47.06  # Actual from your training
    },
    'hyperlocal_integration': {
        'iot_sensor_fusion': True,
        'spatial_clustering': True,
        'data_quality_weighting': True,
        'real_time_calibration': True
    }
}

# REALISTIC AQI/PM2.5 RANGES for Gurugram
REALISTIC_RANGES = {
    'pm25': {
        'excellent': (5, 12),      # Very rare in Gurugram
        'good': (12, 35),          # Monsoon/early morning
        'moderate': (35, 55),      # Normal conditions
        'poor': (55, 90),          # Common in winter
        'very_poor': (90, 150),    # Winter evenings
        'severe': (150, 250)       # Extreme winter days
    },
    'seasonal_factors': {
        'winter_multiplier': 2.5,   # Dec-Feb
        'monsoon_reduction': 0.4,   # Jul-Sep
        'summer_base': 1.2,         # Mar-Jun
        'post_monsoon': 1.8         # Oct-Nov
    }
}

# IoT SENSOR SPECIFICATIONS - Real devices
IOT_SENSOR_SPECS = {
    'PMS5003': {
        'manufacturer': 'Plantower',
        'parameters': ['PM1.0', 'PM2.5', 'PM10'],
        'accuracy_pm25': '±10 μg/m³ (0-100), ±10% (100-500)',
        'measurement_range': '0-500 μg/m³',
        'resolution': '1 μg/m³',
        'response_time': '10 seconds'
    },
    'SDS011': {
        'manufacturer': 'Nova Fitness',
        'parameters': ['PM2.5', 'PM10'],
        'accuracy_pm25': '±15 μg/m³',
        'measurement_range': '0-999.9 μg/m³',
        'resolution': '0.3 μg/m³',
        'response_time': '1 second'
    },
    'BME280': {
        'manufacturer': 'Bosch',
        'parameters': ['Temperature', 'Humidity', 'Pressure'],
        'temp_accuracy': '±1.0°C',
        'humidity_accuracy': '±3% RH',
        'pressure_accuracy': '±1 hPa'
    }
}
