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

# Pfad zu deinen Bildern
SOURCE_DIR = '/home/jona/Schreibtisch/Studium/Semester3/Bioimaging/Practical_week/Data/test_images_feature_extraction_script'

# DataFrame initialisieren
df = pd.DataFrame()

# csv zum Speichern der Ergebnisse
output_csv = '/home/jona/Schreibtisch/Studium/Semester3/Bioimaging/Practical_week/Data/final_features.csv'

# -----------------------------------------------------------------------------
# 1. Hilfsfunktionen (Segmentierung & Features)
# -----------------------------------------------------------------------------

def get_binary_mask_chicken(img_bgr):
    """Deine Custom-Segmentierung (angepasst für OpenCV BGR Input)"""
    # OpenCV lädt BGR, skimage will oft RGB/HSV. Konvertierung ist sicherer.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = color.rgb2hsv(img_rgb)
    saturation = hsv[:, :, 1]
    
    blurred = filters.gaussian(saturation, sigma=3.0)
    
    try:
        thresh = filters.threshold_otsu(blurred)
        binary = blurred > thresh
    except:
        return np.ones(saturation.shape, dtype=bool)

    # Ecken-Check
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
    
    # -----------------------------------------------------------
    # 1. Bounding Box Cutout (WICHTIG für saubere GLCM/Textur)
    # -----------------------------------------------------------
    # Wir schneiden das Rechteck aus, das das Objekt enthält.
    # So rechnen wir nicht auf riesigen schwarzen Flächen.
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if rows.any() and cols.any(): # Sicherheitscheck falls Maske leer
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        rmin = max(0, rmin-10); rmax = min(img_gray.shape[0], rmax+10)
        cmin = max(0, cmin-10); cmax = min(img_gray.shape[1], cmax+10)

        roi_gray = img_gray[rmin:rmax, cmin:cmax]
        roi_mask = mask[rmin:rmax, cmin:cmax]
        roi_rgb = img_rgb[rmin:rmax, cmin:cmax]
    else:
        # Fallback: Nimm ganzes Bild
        roi_gray = img_gray
        roi_mask = mask
        roi_rgb = img_rgb

    # -----------------------------------------------------------
    # A. Farbe (Nur innerhalb der Maske)
    # -----------------------------------------------------------
    if np.sum(roi_mask) > 0:
        features['mean_r'] = roi_rgb[:,:,0][roi_mask].mean()
        features['mean_g'] = roi_rgb[:,:,1][roi_mask].mean()
        features['mean_b'] = roi_rgb[:,:,2][roi_mask].mean()
    else:
        features['mean_r'] = roi_rgb[:,:,0].mean()
        features['mean_g'] = roi_rgb[:,:,1].mean()
        features['mean_b'] = roi_rgb[:,:,2].mean()

    # -----------------------------------------------------------
    # B. Textur (GLCM) - Auf der Bounding Box (ROI)
    # -----------------------------------------------------------
    glcm = feature.graycomatrix(roi_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features['glcm_contrast'] = feature.graycoprops(glcm, 'contrast')[0, 0]
    features['glcm_homogeneity'] = feature.graycoprops(glcm, 'homogeneity')[0, 0]
    features['glcm_energy'] = feature.graycoprops(glcm, 'energy')[0, 0]
    features['glcm_correlation'] = feature.graycoprops(glcm, 'correlation')[0, 0]

    # -----------------------------------------------------------
    # C. Entropy (Maskiert)
    # -----------------------------------------------------------
    entr_img = entropy(roi_gray, disk(5)) 
    if np.sum(roi_mask) > 0:
        features['mean_entropy'] = entr_img[roi_mask].mean()
    else:
        features['mean_entropy'] = entr_img.mean()

    # -----------------------------------------------------------
    # D. "Knusprigkeit" (Ersatz für Euler Number)
    # -----------------------------------------------------------
    # Canny Edge Detector innerhalb der Maske.
    edges = canny(roi_gray, sigma=2.0)
    edges_on_object = edges & roi_mask 
    
    # Feature: Edge Density (Kantenpixel pro Objektpixel)
    obj_pixels = np.sum(roi_mask)
    if obj_pixels > 0:
        features['internal_edge_density'] = np.sum(edges_on_object) / obj_pixels
    else:
        features['internal_edge_density'] = 0


    # -----------------------------------------------------------
    # E. Regionprops (Form)
    # -----------------------------------------------------------
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
            
    # Flag für Maske/Keine Maske
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
# 2. Main Loop
# -----------------------------------------------------------------------------
images = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]

print("--- STEUERUNG ---")
print("Pfeil LINKS (oder 'a')  = Maske A (Custom/Otsu)")
print("Pfeil RECHTS (oder 'd') = Maske B (Rembg)")
print("Pfeil RUNTER (oder 's') = Keine Maske (Ganzes Bild)")
print("ESC                     = Beenden und Speichern")
print("-----------------")

for i, img_name in enumerate(images):
    img_path = os.path.join(SOURCE_DIR, img_name)
    img = cv2.imread(img_path)
    
    if img is None: continue

    print(f"[{i+1}/{len(images)}] Verarbeite: {img_name}")

    # 1. Maske A berechnen (Custom)
    mask_a_bool = get_binary_mask_chicken(img)
    mask_a_vis = (mask_a_bool * 255).astype('uint8')
    mask_a_vis = cv2.cvtColor(mask_a_vis, cv2.COLOR_GRAY2BGR) # BGR machen für Concatenation
    
    # Text Overlay A
    cv2.putText(mask_a_vis, "LINKS: Custom", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 2. Maske B berechnen (Rembg)
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_rembg = remove(img_rgb)
        # Rembg gibt RGBA zurück, Alpha ist Index 3
        mask_b_bool = output_rembg[:, :, 3] > 0
    except Exception as e:
        print(f"Rembg Fehler: {e}")
        mask_b_bool = np.zeros(img.shape[:2], dtype=bool)

    mask_b_vis = (mask_b_bool * 255).astype('uint8')
    mask_b_vis = cv2.cvtColor(mask_b_vis, cv2.COLOR_GRAY2BGR)
    
    # Text Overlay B
    cv2.putText(mask_b_vis, "RECHTS: Rembg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Text Overlay Original
    img_display = img.copy()
    cv2.putText(img_display, "MITTE: Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 3. Bilder nebeneinander packen (Links | Mitte | Rechts)
    combined = np.hstack((mask_a_vis, img_display, mask_b_vis))

    cv2.imshow('Feature Extractor Selector', combined)
    
    # Warten auf Taste
    key = cv2.waitKey(0)
    
    selected_mask = None
    mask_source = ""

    # Key Codes: 81=Links, 83=Rechts, 84=Runter (Linux/GTK oft). 
    # Windows oft anders (2424832, etc). 
    # Daher auch 'a', 'd', 's' als Backup!
    
    if key in [81, 2, ord('a')]: # LINKS
        print(f"-> Nehme Custom Maske")
        selected_mask = mask_a_bool
        mask_source = "custom_otsu"
        
    elif key in [83, 3, ord('d')]: # RECHTS
        print(f"-> Nehme Rembg Maske")
        selected_mask = mask_b_bool
        mask_source = "rembg_ai"
        
    elif key in [84, 0, ord('s')]: # RUNTER
        print(f"-> Nehme ganzes Bild (Rechteck)")
        selected_mask = np.ones(img.shape[:2], dtype=bool)
        mask_source = "full_rect"
        
    elif key == 27: # ESC
        print("Abbruch durch User.")
        break
    
    else:
        print("Taste nicht erkannt, überspringe Bild (Features werden nicht gespeichert).")
        continue

    # 4. Features berechnen und speichern
    if selected_mask is not None:
        feats = calculate_features(img, selected_mask, img_name, mask_source)
        new_row = pd.DataFrame([feats])
        df = pd.concat([df, new_row], ignore_index=True)

cv2.destroyAllWindows()

# 5. Speichern
df.to_csv(output_csv, index=False)
print(f"Fertig! Features gespeichert in {output_csv}")
print(df.head())