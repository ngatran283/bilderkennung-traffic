from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def _to_object_array(images):
    """Safely converts a list of differently sized images into a 1D object array."""
    arr = np.empty(len(images), dtype=object)
    for i, img in enumerate(images):
        arr[i] = img
    return arr


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
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
    X_train, y_train, meta_train = _load_split(data_dir, "Train.csv")
    X_test, y_test, meta_test = _load_split(data_dir, "Test.csv")

    print(f"✅ Train: {len(X_train)} images | Test: {len(X_test)} images")
    return X_train, y_train, meta_train, X_test, y_test, meta_test

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    print(project_dir)
    processed_data_path = project_dir / 'data'

    X_train, y_train, meta_train, X_test, y_test, meta_test = load_gtsrb_data(processed_data_path)

    if X_train is not None:
        print(f"\nSuccessfully loaded {len(X_train)} training images and {len(X_test)} test images.")
        print(f"Type of X_train: {type(X_train)}")
        print(X_train[0].shape)  # (26, 27, 3)   ← image
        print(y_train[0])  # 20            ← label
        print(meta_train.iloc[0])  # Width 27, Height 26, Roi..., ClassId 20, Path ...