# Phase 1: Data Exploration and Preprocessing
# File: 01_data_exploration.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# For Google Colab users
# from google.colab import drive
# drive.mount('/content/drive')

print("=== Phase 1: Data Exploration and Preprocessing ===")
print("Libraries imported successfully!")

# Step 1: Load the HAM10000 dataset
def load_ham10000_data(data_path):
    """
    Load HAM10000 dataset metadata and images

    Args:
        data_path (str): Path to the dataset directory

    Returns:
        pandas.DataFrame: Metadata dataframe
    """
    # Load metadata CSV
    metadata_path = os.path.join(data_path, 'HAM10000_metadata.csv')
    metadata = pd.read_csv(metadata_path)

    print(f"Dataset loaded with {len(metadata)} samples")
    print(f"Columns: {list(metadata.columns)}")

    return metadata

# Step 2: Data exploration functions
def explore_data(df):
    """Explore the dataset structure and distributions"""

    print("\n=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"\nData Types:")
    print(df.dtypes)

    print(f"\nMissing Values:")
    print(df.isnull().sum())

    print(f"\nClass Distribution:")
    class_counts = df['dx'].value_counts()
    print(class_counts)

    # Plot class distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    class_counts.plot(kind='bar')
    plt.title('Skin Disease Class Distribution')
    plt.xlabel('Disease Type')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)

    # Plot age distribution
    plt.subplot(1, 2, 2)
    df['age'].hist(bins=30, alpha=0.7)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.sh

    
# Phase 2: Model Training - CNN and Transfer Learning
# File: 02_model_training.ipynb

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=== Phase 2: Model Training ===")

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Step 1: Data preparation and augmentation
def create_data_generators(train_df, val_df, images_path, target_size=(224, 224), batch_size=32):
    """
    Create data generators with augmentation for training and validation
    """

    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    # Validation data (no augmentation, only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=images_path,
        x_col='image_id',
        y_col='dx',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_dataframe(