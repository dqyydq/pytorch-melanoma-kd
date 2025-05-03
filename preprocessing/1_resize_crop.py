
import os
import numpy as np
from PIL import Image # Using PIL for loading and saving
import cv2          # Using OpenCV for resizing (as in original)
from pathlib import Path # Using pathlib for cleaner path operations
from tqdm import tqdm    # Adding a progress bar for better feedback

def create_dir(dir_path):
    """Creates a directory if it doesn't exist."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def process_image(input_path, output_path, target_size):
    """Loads, center crops, resizes, and saves a single image."""
    try:
        # 1. Load image using PIL
        img_pil = Image.open(input_path).convert('RGB') # Ensure 3 channels (RGB)
        org_img = np.array(img_pil) # Convert to numpy array for processing

        # 2. Centered square crop (same logic as original)
        h, w, _ = org_img.shape
        remove = min(h, w) // 2
        center_y, center_x = h // 2, w // 2
        cropped_img = org_img[center_y - remove : center_y + remove,
                              center_x - remove : center_x + remove]

        # 3. Resize using OpenCV (same logic as original)
        # cv2.resize expects (width, height)
        resized_img = cv2.resize(cropped_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # 4. Save the processed image using PIL
        output_pil = Image.fromarray(resized_img)
        output_pil.save(output_path)
        return True # Indicate success

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False # Indicate failure

def main():
    """Main function to run the preprocessing."""
    DATA_TYPE = '.jpg'
    TARGET_SIZE = 256

    # --- Define Paths using pathlib ---
    # Adjust base path if your data is located elsewhere
    base_data_path = Path('D:/python_code/pytorch_melanama_kd/data/ISIC2019') 
    
    # Input paths for ISIC 2019
    input_train_path = base_data_path / 'ISIC_2019_Training_Input/' # Corrected folder name based on snippet
    input_test_path = base_data_path / 'ISIC_2019_Test_Input/'     # Corrected folder name based on snippet

    # Output paths (using a distinct name for PyTorch version)
    output_base_path = base_data_path / 'centre_square_cropped_pytorch/' 
    output_train_path = output_base_path / 'train/'
    output_test_path = output_base_path / 'test/'

    # --- Create Output Directories ---
    print(f"Creating output directories at: {output_base_path}")
    create_dir(output_train_path)
    create_dir(output_test_path)

    # --- Process Training Images ---
    print(f"\nProcessing Training Images from: {input_train_path}")
    # Use pathlib's glob to find image files
    train_files = sorted(list(input_train_path.glob(f'*{DATA_TYPE}'))) 
    if not train_files:
        print(f"Warning: No training images found in {input_train_path}")
        
    success_count_train = 0
    fail_count_train = 0
    for img_file_path in tqdm(train_files, desc="Training Images"):
        # Construct output filename based on input filename
        output_file_path = output_train_path / img_file_path.name 
        if process_image(img_file_path, output_file_path, TARGET_SIZE):
            success_count_train += 1
        else:
            fail_count_train += 1
            
    print(f"Training images processed: {success_count_train} succeeded, {fail_count_train} failed.")

    # --- Process Test Images ---
    print(f"\nProcessing Test Images from: {input_test_path}")
    test_files = sorted(list(input_test_path.glob(f'*{DATA_TYPE}')))
    if not test_files:
         print(f"Warning: No test images found in {input_test_path}")

    success_count_test = 0
    fail_count_test = 0
    for img_file_path in tqdm(test_files, desc="Test Images"):
        output_file_path = output_test_path / img_file_path.name
        if process_image(img_file_path, output_file_path, TARGET_SIZE):
             success_count_test += 1
        else:
            fail_count_test += 1
            
    print(f"Test images processed: {success_count_test} succeeded, {fail_count_test} failed.")

    print("\nPreprocessing complete.")
    print(f"Processed images saved in: {output_train_path} and {output_test_path}")

if __name__ == "__main__":
    # Ensure the script runs only when executed directly
    main()