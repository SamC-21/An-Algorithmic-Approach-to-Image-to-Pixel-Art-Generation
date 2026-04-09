import os
import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float


# CONFIGURATION


DATASET_PATH = r"C:\Users\Sam\DatasetsDiss\DatasetLandscapes"


# UTILITY FUNCTION DEBUGGING


def save_debug_stage(img_rgb, stage_name, output_dir, filename):
    #Saves each stage of the pipeline for debugging and comparison.
    debug_dir = os.path.join(output_dir, 'DEBUG_STAGES')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Strip extension from original filename and append the stage name
    base = os.path.splitext(filename)[0]
    save_path = os.path.join(debug_dir, f"{base}__{stage_name}.png")
    
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)
    print(f"  [debug] Saved: {stage_name}")



# IMAGE PROCESSING


def enhance_contrast(img_rgb):
    #Uses CLAHE to amplify the L channel for better contrast without blowing out skies
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # The image is split into 8x8 tiles; clipLimit prevents over-amplification
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def prepare_low_res_canvas(img_rgb, target_width):
    #Downscales the image while maintaining aspect ratio and smoothing noise
    h, w = img_rgb.shape[:2]
    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)
    
    # Bilateral filter removes noise while keeping sharp silhouettes
    smoothed = cv2.bilateralFilter(img_rgb, d=7, sigmaColor=50, sigmaSpace=50)
    
    # Downscales using AREA interpolation to prevent artifacts
    small_img = cv2.resize(smoothed, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return small_img, (w, h)


def apply_segmentation_posterize(img_rgb):
    #Groups pixels into semantic blobs and averages the colours using Felzenszwalb segmentation
    img_float = img_as_float(img_rgb)
    
    # Connected regions of similar colour/texture controlled by scale, sigma, and min_size
    segments = felzenszwalb(img_float, scale=100, sigma=0.5, min_size=20)
    final_canvas = np.copy(img_rgb)
    
    for seg_id in np.unique(segments):
        mask = (segments == seg_id)
        avg_color = np.mean(img_rgb[mask], axis=0)
        
        # Blend is 30% original pixel and 70% average colour for an artistic look
        final_canvas[mask] = (img_rgb[mask] * 0.3 + avg_color * 0.7).astype(np.uint8)
        
    return final_canvas



# COLOUR & DITHERING


def get_kmeans_palette(img_rgb, k=16):
    # Uses K-means clustering to extract a custom dominant colour palette
    pixels = img_rgb.reshape((-1, 3)).astype(np.float32)
    
    # Balance speed and quality
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    
    # Runs K-means 10 times and picks the best cluster center setup
    _, _, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    palette = centers.astype(np.uint8).flatten().tolist()
    padding = [0] * (768 - len(palette)) # PIL needs a full 256-colour palette
    return palette + padding


def apply_dither(img_rgb, palette_data, spread=0.08):
    # 2x2 bayer dither using sobel edges to protect borders in the image
    # VALUES that can be tweaked for different results:
    # spread recommended(0.05 to 0.2): Controls 'crunchiness'. Higher = more visible dots; Lower = flatter
    # weight multiplier recommended(2.5): Controls edge thickness in the dither mask
    # Increase to 5.0 for extremely clean edges; decrease to 1.0 for more dither bleed on edges
    h, w, _ = img_rgb.shape
    
    # Edge Protection via Sobel
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize: Strong edges = 1 (no dither), flat areas = 0 (full dither)
    edge_mask = cv2.normalize(edge_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    dither_weight = 1.0 - np.clip(edge_mask * 2.5, 0, 1)
    dither_weight = np.stack([dither_weight] * 3, axis=-1)

    # 2x2 Bayer Matrix (centered between -0.5 and 0.5)
    bayer_matrix = np.array([[0, 2], [3, 1]], dtype=float) / 4.0 - 0.5
    tiled_matrix = np.tile(bayer_matrix, (h // 2 + 1, w // 2 + 1))[:h, :w]
    tiled_matrix = np.stack([tiled_matrix] * 3, axis=-1)
    
    # Apply dither without overflow
    img_float = img_rgb.astype(np.float32) / 255.0
    img_dithered = img_float + (tiled_matrix * spread * dither_weight)
    img_dithered = np.clip(img_dithered * 255, 0, 255).astype(np.uint8)

    # Map dithered pixels to the nearest colour in the k-means palette
    pil_img = Image.fromarray(img_dithered)
    palette_img = Image.new("P", (1, 1))
    palette_img.putpalette(palette_data)
    
    return pil_img.quantize(palette=palette_img, dither=0).convert("RGB")



# CORE PIPELINE 

def generate_pixel_art(img_rgb, target_width, k_colors, debug=False, output_dir=None, filename=None):
    
    def save_steps(img, stage_name):
        if debug and output_dir and filename:
            save_debug_stage(img, stage_name, output_dir, filename)

    # 1. Image Enhancement
    enhanced = enhance_contrast(img_rgb)
    save_steps(enhanced, "1_enhanced_contrast")

    # 2. Downscaling & Smoothing
    small_rgb, original_dims = prepare_low_res_canvas(enhanced, target_width)
    save_steps(small_rgb, "2_downscaled")

    # 3. Structural Segmentation
    segmented_canvas = apply_segmentation_posterize(small_rgb)
    save_steps(segmented_canvas, "3_segmented")

    # 4. Colour Logic & Dithering
    dynamic_palette = get_kmeans_palette(segmented_canvas, k=k_colors)
    dithered_rgb = apply_dither(segmented_canvas, dynamic_palette)
    save_steps(np.array(dithered_rgb), "4_dithered")

    # 5. Final Upscale (Nearest Neighbor)
    res_array = np.array(dithered_rgb)
    final = cv2.resize(res_array, original_dims, interpolation=cv2.INTER_NEAREST)
    save_steps(final, "5_final_upscaled")

    return final

# EXECUTION LOOP

def main():
    target_dir = DATASET_PATH
    OUTPUT_DIR = os.path.join(target_dir, 'FINAL_PIXEL_ART')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print(f"No images found in {target_dir}")
        return

    print(f"\n--- UNIFIED RGB PIXEL GENERATOR ---")
    print(f"Found {len(images)} images in {target_dir}")
    
    img_idx = input("Select image index (or type 'all' to batch process): ")
    target_w = int(input("Enter target pixel width (e.g., 256): ") or 256)
    k_colors = int(input("Enter number of colours (k) (e.g., 16): ") or 16)

    def process_and_save(filename):
        img_path = os.path.join(target_dir, filename)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        print(f"Processing {filename} (k={k_colors}, w={target_w})...")
        result_rgb = generate_pixel_art(
            img_rgb, target_w, k_colors,
            debug=True, # Set to False to disable debug images
            output_dir=OUTPUT_DIR,
            filename=filename
        )

        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        save_name = f"FinalOutput_{k_colors}_{target_w}_{filename}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), result_bgr)

    # Batch Processing Logic
    if img_idx.strip().lower() in ('all', 'a'):
        batch_size = 50
        for i in range(0, len(images), batch_size):
            current_batch = images[i : i + batch_size]
            
            for filename in current_batch:
                process_and_save(filename)
            
            total_done = min(i + batch_size, len(images))
            print(f"\n--- Batch Complete ({total_done}/{len(images)}) ---")
            
            if total_done < len(images):
                cont = input("Press Enter to continue to next batch or 'q' to quit: ")
                if cont.lower() == 'q':
                    print("Exiting process...")
                    break
            else:
                print("All images processed successfully.")
    else:
        # Single Image Logic
        try:
            process_and_save(images[int(img_idx)])
            print("Processing complete.")
        except (IndexError, ValueError):
            print("Invalid input. Please enter a valid index or 'all'.")

if __name__ == "__main__":
    main()