import os
import cv2 

source_dir = '/home/jona/Schreibtisch/Studium/Semester3/Bioimaging/Practical_week/Data/dataset_puppy_donuts'
images = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

print("Anleitung: Pfeil LINKS = Behalten | Pfeil RECHTS = Löschen | ESC = Beenden")

for img_name in images:
    img_path = os.path.join(source_dir, img_name)
    img = cv2.imread(img_path)
    
    if img is None: continue

    cv2.imshow('Dataset Cleaner', img)
    key = cv2.waitKey(0)

    if key == 81 or key == 2: # Pfeil Links (Code variiert je nach System)
        print(f"Behalten: {img_name}")
    elif key == 83 or key == 3: # Pfeil Rechts
        print(f"Lösche: {img_name}")
        os.remove(img_path)
    elif key == 27: # ESC
        break

cv2.destroyAllWindows()

