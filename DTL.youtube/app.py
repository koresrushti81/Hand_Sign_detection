import sys
import os

# Add src to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# pyrefly: ignore [missing-import]
from test_model import cap, detector, classifier, OFFSET
import cv2
import time

def main():
    """
    Main entry point for the Hand Sign Recognition System.
    This script runs the real-time webcam inference.
    """
    print("--- ASL Hand Sign Recognition System ---")
    print("Press 'q' to quit.")
    
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

            try:
                # Add padding/offset
                y1, y2 = max(0, y - OFFSET), min(img.shape[0], y + h + OFFSET)
                x1, x2 = max(0, x - OFFSET), min(img.shape[1], x + w + OFFSET)
                
                img_crop = img[y1:y2, x1:x2]
                
                if img_crop.size > 0:
                    label, confidence = classifier.predict(img_crop)

                    # UI Styling
                    color = (45, 255, 45) if label != "Unknown" else (45, 45, 255)
                    
                    # Draw Bounding Box
                    cv2.rectangle(img_output, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw Label Box
                    label_text = f"{label} {confidence:.0f}%"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
                    cv2.rectangle(img_output, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
                    cv2.putText(img_output, label_text, (x1 + 5, y1 - 10), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)

            except Exception as e:
                pass

        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        cv2.putText(img_output, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("ASL Recognition Pro", img_output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
