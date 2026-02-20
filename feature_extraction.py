import os
import cv2
import numpy as np
import pandas as pd
from skimage import color, filters, morphology, measure, feature
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy.stats import moment
from scipy import ndimage
from rembg import remove
from pyefd import elliptic_fourier_descriptors
from skimage.feature import canny


SOURCE_DIR = '/home/jona/Schreibtisch/Studium/Semester3/Bioimaging/Practical_week/dog-learning/data/dog_chicken/test/Not_used/dog'
OUTPUT_CSV = '/home/jona/Schreibtisch/Studium/Semester3/Bioimaging/Practical_week/dog-learning/data/dog_chicken/test/Not_used/dog_features.csv'
DISPLAY_HEIGHT = 500

def resize_to_height(image, target_height):
    h, w = image.shape[:2]
    scale = target_height / h
    return cv2.resize(image, (int(w * scale), target_height))

def get_binary_mask_chicken(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = color.rgb2hsv(img_rgb)
    saturation = hsv[:, :, 1]
    
    blurred = filters.gaussian(saturation, sigma=3.0)
    
    try:
        thresh = filters.threshold_otsu(blurred)
        binary = blurred > thresh
    except:
        return np.ones(saturation.shape, dtype=bool)

    corners = [binary[0,0], binary[0,-1], binary[-1,0], binary[-1,-1]]
    if sum(corners) > 2:
        binary = ~binary

    binary = morphology.binary_closing(binary, morphology.disk(10))
    binary = ndimage.binary_fill_holes(binary)
    
    label_img = measure.label(binary)
    if label_img.max() == 0:
        return binary
    
    regions = measure.regionprops(label_img)
    largest_region = max(regions, key=lambda x: x.area)
    mask = np.zeros_like(binary)
    mask[label_img == largest_region.label] = 1
    return mask

def calculate_features(img_bgr, mask, filename, mask_source):
    features = {'filename': filename, 'mask_source': mask_source}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. ROI Cutout
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin = max(0, rmin-10); rmax = min(img_gray.shape[0], rmax+10)
        cmin = max(0, cmin-10); cmax = min(img_gray.shape[1], cmax+10)
        roi_gray = img_gray[rmin:rmax, cmin:cmax]
        roi_mask = mask[rmin:rmax, cmin:cmax]
        roi_rgb = img_rgb[rmin:rmax, cmin:cmax]
    else:
        roi_gray = img_gray
        roi_mask = mask
        roi_rgb = img_rgb

    # A. color
    if np.sum(roi_mask) > 0:
        features['mean_r'] = roi_rgb[:,:,0][roi_mask].mean()
        features['mean_g'] = roi_rgb[:,:,1][roi_mask].mean()
        features['mean_b'] = roi_rgb[:,:,2][roi_mask].mean()
    else:
        features['mean_r'] = roi_rgb[:,:,0].mean()
        features['mean_g'] = roi_rgb[:,:,1].mean()
        features['mean_b'] = roi_rgb[:,:,2].mean()

    # B. textur (GLCM)
    glcm = feature.graycomatrix(roi_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features['glcm_contrast'] = feature.graycoprops(glcm, 'contrast')[0, 0]
    features['glcm_homogeneity'] = feature.graycoprops(glcm, 'homogeneity')[0, 0]
    features['glcm_energy'] = feature.graycoprops(glcm, 'energy')[0, 0]
    features['glcm_correlation'] = feature.graycoprops(glcm, 'correlation')[0, 0]

    # C. entropy
    entr_img = entropy(roi_gray, disk(5)) 
    if np.sum(roi_mask) > 0:
        features['mean_entropy'] = entr_img[roi_mask].mean()
    else:
        features['mean_entropy'] = entr_img.mean()

    # D. edge Density
    edges = canny(roi_gray, sigma=2.0)
    edges_on_object = edges & roi_mask 
    obj_pixels = np.sum(roi_mask)
    if obj_pixels > 0:
        features['internal_edge_density'] = np.sum(edges_on_object) / obj_pixels
    else:
        features['internal_edge_density'] = 0

    # E. regionprops
    label_img = measure.label(mask) 
    if label_img.max() > 0:
        regions = measure.regionprops(label_img)
        props = max(regions, key=lambda x: x.area)
        features['eccentricity'] = props.eccentricity
        features['solidity'] = props.solidity
        
        # F. EFD
        contours = measure.find_contours(mask, 0.5)
        if len(contours) > 0:
            contour = max(contours, key=len)
            try:
                coeffs = elliptic_fourier_descriptors(contour, order=10, normalize=True)
                for i in range(1, 4):
                    harm = coeffs[i-1]
                    amp = np.sqrt(harm[0]**2 + harm[1]**2 + harm[2]**2 + harm[3]**2)
                    features[f'efd_harm_{i}_amp'] = amp
            except:
                 pass
            
    if mask_source == "full_rect":
        features['is_fallback'] = 1
    else:
        features['is_fallback'] = 0

    defaults = ['eccentricity', 'solidity', 'efd_harm_1_amp', 'efd_harm_2_amp', 'efd_harm_3_amp', 'internal_edge_density']
    for key in defaults:
        if key not in features:
            features[key] = 0
            
    return features

# -----------------------------------------------------------------------------
    Main Loop
# -----------------------------------------------------------------------------

images = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
df = pd.DataFrame()

print("--- STEUERUNG ---")
print("Arrow to the left = mask A (Custom/Otsu)")
print("Arrow to the right = mask B (Rembg)")
print("Arrow down = no mask (whole image)")
print("ESC                     = Save and end")
print("-----------------")

cv2.namedWindow('Selector', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Selector', 1200, 600)

for i, img_name in enumerate(images):
    img_path = os.path.join(SOURCE_DIR, img_name)
    img = cv2.imread(img_path)
    
    if img is None: continue

    print(f"[{i+1}/{len(images)}] Verarbeite: {img_name}")

    # --- 1. Calculating masks ---
    mask_a_bool = get_binary_mask_chicken(img)
    
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_rembg = remove(img_rgb)
        mask_b_bool = output_rembg[:, :, 3] > 0
    except Exception as e:
        mask_b_bool = np.zeros(img.shape[:2], dtype=bool)

    # --- 2. prepare visualisation ---
    mask_a_vis = (mask_a_bool * 255).astype('uint8')
    mask_a_vis = cv2.cvtColor(mask_a_vis, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_a_vis, "LINKS: Custom", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    mask_a_vis = resize_to_height(mask_a_vis, DISPLAY_HEIGHT)

    mask_b_vis = (mask_b_bool * 255).astype('uint8')
    mask_b_vis = cv2.cvtColor(mask_b_vis, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_b_vis, "RECHTS: Rembg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    mask_b_vis = resize_to_height(mask_b_vis, DISPLAY_HEIGHT)

    img_display = img.copy()
    cv2.putText(img_display, "MITTE: Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    img_display = resize_to_height(img_display, DISPLAY_HEIGHT)

    # --- 3. Display image + masks ---
    combined = np.hstack((mask_a_vis, img_display, mask_b_vis))
    cv2.imshow('Selector', combined)
    
    # --- 4. Input Handling ---
    key = cv2.waitKey(0)
    
    selected_mask = None
    mask_source = ""
    
    if key in [81, 2, ord('a')]: # Left
        print(f"-> Custom")
        selected_mask = mask_a_bool
        mask_source = "custom_otsu"
    elif key in [83, 3, ord('d')]: # Right
        print(f"-> Rembg")
        selected_mask = mask_b_bool
        mask_source = "rembg_ai"
    elif key in [84, 0, ord('s')]: # Down
        print(f"-> Full Rect")
        selected_mask = np.ones(img.shape[:2], dtype=bool)
        mask_source = "full_rect"
    elif key == 27: # ESC
        print("Abbruch.")
        break
    else:
        print("Taste ignoriert -> Bild übersprungen.")
        continue

    # Calculating features
    feats = calculate_features(img, selected_mask, img_name, mask_source)
    new_row = pd.DataFrame([feats])
    df = pd.concat([df, new_row], ignore_index=True)

cv2.destroyAllWindows()

df.to_csv(OUTPUT_CSV, index=False)
print(f"Gespeichert: {OUTPUT_CSV}")
print(df.head())
