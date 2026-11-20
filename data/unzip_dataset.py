import zipfile
from pathlib import Path
import shutil
import pandas as pd

def organize_gtsrb_dataset():
    """
    Unzips and organizes the GTSRB dataset from the raw data directory
    into a structured 'processed' directory.
    """
    # --- 1. Define Paths ---
    project_dir = Path(__file__).resolve().parents[1]
    raw_data_dir = project_dir / "data" / "raw"
    processed_data_dir = project_dir / "data" / "processed"

    # Define paths to the specific zip files
    training_zip_path = raw_data_dir / "GTSRB_Final_Training_Images.zip"
    test_zip_path = raw_data_dir / "GTSRB_Final_Test_Images.zip"
    test_classId_zip_path = raw_data_dir / "GTSRB_Final_Test_GT.zip"

    # Define paths for the final structured directories
    train_dir = processed_data_dir / "train"
    test_dir = processed_data_dir / "test"
    meta_dir = processed_data_dir / "meta"

    print("--- Starting Dataset Organization ---")

    # --- 2. Create Processed Directories ---
    print(f"Creating directories at: {processed_data_dir}")
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # --- 3. Unzip and Organize Training Data ---
    print(f"Unzipping training data from {training_zip_path}...")
    with zipfile.ZipFile(training_zip_path, 'r') as zip_ref:
        temp_extract_path = processed_data_dir / "temp_train_extract"
        zip_ref.extractall(temp_extract_path)

    source_train_images = temp_extract_path / "GTSRB" / "Final_Training" / "Images"

    for class_folder in source_train_images.iterdir():
        if class_folder.is_dir():
            shutil.move(str(class_folder), str(train_dir))

    print("\nBuilding Train.csv...")
    train_frames = []
    for class_folder in sorted(train_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        gt_csv = class_folder / f"GT-{class_folder.name}.csv"
        if not gt_csv.exists():
            continue
        df = pd.read_csv(gt_csv, sep=";")
        df["Path"] = df["Filename"].apply(lambda f: f"train/{class_folder.name}/{f}")
        df.drop(columns=["Filename"], inplace=True)
        train_frames.append(df)

    train_df = pd.concat(train_frames, ignore_index=True)
    train_df.to_csv(meta_dir / "Train.csv", index=False)
    print(f"Saved Train.csv ({len(train_df)} rows)")

    shutil.rmtree(temp_extract_path)
    print("Training data successfully organized.")

    # --- 4. Unzip and Organize Test Data ---
    print(f"Unzipping test data from {test_zip_path}...")
    with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
        temp_extract_path = processed_data_dir / "temp_test_extract"
        zip_ref.extractall(temp_extract_path)

    source_test_images = temp_extract_path / "GTSRB" / "Final_Test" / "Images"

    for image_file in source_test_images.glob('*.ppm'):
        shutil.move(str(image_file), str(test_dir))

    shutil.rmtree(temp_extract_path)
    print("Test data successfully organized.")

    # --- 5. Unzip test classId and Move Metadata File ---
    print(f"Unzipping test data from {test_classId_zip_path}...")
    with zipfile.ZipFile(test_classId_zip_path, 'r') as zip_ref:
        temp_extract_path = processed_data_dir / "temp_test_classid_extract"
        zip_ref.extractall(temp_extract_path)
    source_csv_path = temp_extract_path / "GT-final_test.csv"
    test_df = pd.read_csv(source_csv_path, sep=";")
    test_df["Path"] = test_df["Filename"].apply(lambda f: f"test/{f}")
    test_df.drop(columns=["Filename"], inplace=True)
    test_df.to_csv(meta_dir / "Test.csv", index=False)
    print(f"Saved Test.csv ({len(test_df)} rows)")
    shutil.rmtree(temp_extract_path)
    print("Metadata successfully organized.")

    print("\n--- Dataset Organization Complete! ---")
    print(f"Training images are in: {train_dir}")
    print(f"Test images are in: {test_dir}")
    print(f"Metadata is in: {meta_dir}")


if __name__ == '__main__':
    organize_gtsrb_dataset()
