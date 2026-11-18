from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

def _to_object_array(images):
    """Safely converts a list of differently sized images into a 1D object array."""
    arr = np.empty(len(images), dtype=object)
    for i, img in enumerate(images):
        arr[i] = img
    return arr

def preprocess_images(images, meta, img_size=32, use_roi=False):
    """
    Preprocesses RGB images for the CNN.

    Steps:
      1. (Optional) crop to the ROI from the metadata
      2. Convert RGB -> grayscale
      3. Resize to img_size x img_size
      4. Normalize pixel values to [0, 1]
      5. Add channel dimension for Keras Conv2D

    Returns:
        np.ndarray of shape (N, img_size, img_size, 1), dtype float32, values in [0, 1]
    """
    processed = []
    for img, (_, row) in zip(images, meta.iterrows()):
        if use_roi:
            img = img[row["Roi.Y1"]:row["Roi.Y2"], row["Roi.X1"]:row["Roi.X2"]]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (img_size, img_size))
        processed.append(img)

    X = np.array(processed, dtype=np.float32) / 255.0   # (N, 32, 32)
    return np.expand_dims(X, axis=-1)                   # (N, 32, 32, 1)

def _load_split(data_dir, csv_name):
    """Loads images, labels and metadata of one split based on its CSV file."""
    df = pd.read_csv(data_dir / csv_name)
    images, labels, valid_rows = [], [], []
    n_failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {csv_name}"):
        img = cv2.imread(str(data_dir / row["Path"]))
        if img is None:
            n_failed += 1
            continue
        images.append(img)
        labels.append(row["ClassId"])
        valid_rows.append(idx)

    if n_failed:
        print(f"⚠️ {n_failed} images from {csv_name} could not be loaded.")

    meta = df.loc[valid_rows].reset_index(drop=True)   # incl. ROI, Width, Height
    return _to_object_array(images), np.array(labels), meta


def load_gtsrb_data(data_dir):
    """
    Loads the GTSRB dataset using Train.csv and Test.csv.

    Args:
        data_dir (str or Path): Directory containing 'Train.csv', 'Test.csv'
                                and the 'Train' and 'Test' folders.
    Returns:
        tuple: (X_train, y_train, meta_train, X_test, y_test, meta_test)
               X_*: images (RGB, varying sizes, object array)
               y_*: labels (ClassId)
               meta_*: DataFrame with metadata (incl. ROI) aligned with the images
    """
    data_dir = Path(data_dir)
    # 1. Load
    X_train, y_train, meta_train = _load_split(data_dir, "Train.csv")
    X_test, y_test, meta_test = _load_split(data_dir, "Test.csv")
    # 2. Preprocessing (grayscale, resize 32×32, normalize 0–1, add channel dimension)
    X_train = preprocess_images(X_train, meta_train, use_roi=False)
    X_test  = preprocess_images(X_test,  meta_test,  use_roi=False)
    print("✅ Data loading, resizing, and grayscaling complete.")
    print(f"Training images shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Test images shape: {X_test.shape}, Labels shape: {y_test.shape}")
    return X_train, y_train, meta_train, X_test, y_test, meta_test

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    print(project_dir)
    processed_data_path = project_dir / 'data'
    # --- 1. Load the preprocessed data ---
    X_train, y_train, meta_train, X_test, y_test, meta_test = load_gtsrb_data(processed_data_path)
    # --- 2. Inspect the labels BEFORE encoding ---
    print("--- Labels Before One-Hot Encoding ---")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"First 5 labels in y_train: {y_train[:5]}")
    print("-" * 40)

    # --- 3. Perform One-Hot Encoding ---
    # The number of classes in the GTSRB dataset is 43.
    num_classes = 43

    # The `to_categorical` function from Keras handles the conversion for us.
    # It takes the integer labels and the total number of classes as input.
    y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
    y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

    # --- 4. Inspect the labels AFTER encoding ---
    print("--- Labels After One-Hot Encoding ---")
    print(f"Shape of y_train_one_hot: {y_train_one_hot.shape}")
    print("First 5 one-hot encoded labels in y_train_one_hot:")
    print(y_train_one_hot[:5])
    print("-" * 40)