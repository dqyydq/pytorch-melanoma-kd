import numpy as np
import os
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm # For progress bars

# -----------------------------------------------------------------------------
# Configuration (Modify paths and parameters here)
# -----------------------------------------------------------------------------
# Base directory containing ISIC 2019 data folders
BASE_DATA_DIR = 'D:/python_code/pytorch_melanama_kd/data/ISIC2019'

# Input directory name (containing output of 1_crop_resize.py)
# This directory should have 'train' and 'test' subdirectories
INPUT_IMAGE_BASE_DIR_NAME = 'centre_square_cropped_pytorch/'

# Output directory name for saving inpainted images
# This directory will also have 'train' and 'test' subdirectories created
OUTPUT_IMAGE_BASE_DIR_NAME = 'inpainted_pytorch/'

# --- Inpainting Parameters (from original script's removeHair logic) ---
# Kernel size for morphological operations
MORPH_KERNEL_SIZE = (17, 17) # Cross-shaped kernel
# Threshold value for creating the mask from blackhat
INPAINT_THRESHOLD_VALUE = 36
# Radius for the inpainting algorithm (cv2.INPAINT_TELEA)
INPAINT_RADIUS = 1
# -----------------------------------------------------------------------------

# Image file extension
IMAGE_EXTENSION = '.jpg'

def create_dir(dir_path):
    """Creates a directory if it doesn't exist."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def removeHair_inpainting(image_np: np.ndarray) -> np.ndarray:
    """
    Applies hair removal inpainting using OpenCV's morphological
    blackhat operator and Telea inpainting algorithm.

    Args:
        image_np: Input image as a NumPy array (RGB format).

    Returns:
        Inpainted image as a NumPy array (RGB format).
    """
    # Convert to grayscale for morphological operations
    grayScale = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Create the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, MORPH_KERNEL_SIZE)

    # Apply blackhat operation to find dark hair-like structures
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # Apply thresholding to create a binary mask of the structures to remove
    # Pixels above the threshold (likely hair) become white (255) in the mask
    _, threshold_mask = cv2.threshold(blackhat, INPAINT_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    # Inpaint the original RGB image using the mask
    # cv2.INPAINT_TELEA is an algorithm based on Fast Marching Method
    final_image_np = cv2.inpaint(image_np, threshold_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)

    return final_image_np

def process_directory(input_dir: Path, output_dir: Path):
    """
    Processes all images in an input directory using inpainting
    and saves them to the output directory.

    Args:
        input_dir: Path object for the input directory.
        output_dir: Path object for the output directory.

    Returns:
        Tuple[int, int]: Number of successfully processed images, number of failed images.
    """
    print(f"\nProcessing images from: {input_dir}")
    # Find all images with the specified extension in the input directory
    image_files = sorted(list(input_dir.glob(f'*{IMAGE_EXTENSION}')))

    if not image_files:
        print(f"Warning: No images with extension '{IMAGE_EXTENSION}' found in {input_dir}")
        return 0, 0 # Return zero counts if no files found

    success_count = 0
    fail_count = 0

    # Iterate through found image files with a progress bar
    for input_path in tqdm(image_files, desc=f"Inpainting {input_dir.name}"):
        # Define the corresponding output path
        output_path = output_dir / input_path.name
        try:
            # 1. Load image using PIL (consistent with step 1 output)
            img_pil = Image.open(input_path).convert('RGB')
            org_img_np = np.array(img_pil) # Convert to NumPy array for OpenCV

            # 2. Apply the inpainting function
            inpainted_img_np = removeHair_inpainting(org_img_np)

            # 3. Convert back to PIL Image for saving
            finalImg_pil = Image.fromarray(inpainted_img_np)

            # 4. Save the inpainted image
            finalImg_pil.save(output_path)
            success_count += 1

        except Exception as e:
            # Catch any errors during processing of a single file
            print(f"Error processing {input_path}: {e}")
            fail_count += 1

    print(f"{input_dir.name} directory processed: {success_count} succeeded, {fail_count} failed.")
    return success_count, fail_count


def main():
    """Main function to orchestrate the inpainting process."""
    base_path = Path(BASE_DATA_DIR)
    input_base = base_path / INPUT_IMAGE_BASE_DIR_NAME
    output_base = base_path / OUTPUT_IMAGE_BASE_DIR_NAME

    # Define specific input and output directories for train and test sets
    input_train_dir = input_base / 'train/'
    input_test_dir = input_base / 'test/'
    output_train_dir = output_base / 'train/'
    output_test_dir = output_base / 'test/'

    print("Starting Inpainting Preprocessing Step...")
    print(f"Reading images from base: {input_base}")
    print(f"Saving inpainted images to base: {output_base}")

    # Ensure output directories exist
    create_dir(output_train_dir)
    create_dir(output_test_dir)

    # Process the training directory
    s_train, f_train = process_directory(input_train_dir, output_train_dir)

    # Process the testing directory
    s_test, f_test = process_directory(input_test_dir, output_test_dir)

    total_success = s_train + s_test
    total_fail = f_train + f_test

    print("\nInpainting complete.")
    print(f"Total images processed: {total_success + total_fail}")
    print(f"Successful: {total_success}, Failed: {total_fail}")
    print(f"Inpainted images saved in directories: {output_train_dir} and {output_test_dir}")

if __name__ == "__main__":
    main()