import cv2
import time
from cvzone.HandTrackingModule import HandDetector
from hand_sign_classifier import HandSignClassifier

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "keras_model.keras"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.4
OFFSET = 20
IMG_SIZE = 224

# ==========================================
# INITIALIZATION
# ==========================================
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

try:
    classifier = HandSignClassifier(MODEL_PATH, LABELS_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)
    print("Hand Sign Recognition System Ready.")
except Exception as e:
    print(f"Error: {e}")
    print("Ensure you have trained the model and have 'keras_model.h5' and 'labels.txt' in the same directory.")
    exit()

prev_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img_output = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Crop Hand Region with Offset
        try:
            img_crop = img[y - OFFSET : y + h + OFFSET, x - OFFSET : x + w + OFFSET]
            
            if img_crop.size > 0:
                # Predict
                label, confidence = classifier.predict(img_crop)

                # Draw Smooth UI
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                
                # Bounding Box
                cv2.rectangle(img_output, (x - OFFSET, y - OFFSET), (x + w + OFFSET, y + h + OFFSET), color, 4)
                
                # Label Background
                cv2.rectangle(img_output, (x - OFFSET, y - OFFSET - 40), (x + w + OFFSET, y - OFFSET), color, cv2.FILLED)
                
                # Text
                text = f"{label} ({confidence:.1f}%)"
                cv2.putText(img_output, text, (x - OFFSET + 5, y - OFFSET - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

        except Exception as e:
            # Handle edge cases where crop might be out of bounds
            pass

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS
    cv2.putText(img_output, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    # Show Window
    cv2.imshow("Hand Sign Recognition", img_output)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
