import os
import cv2
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from hand_sign_classifier import HandSignClassifier

# CONFIG
MODEL_PATH = "keras_model.keras"
LABELS_PATH = "labels.txt"
TEST_DATA_PATH = r"c:\Users\Aris\Downloads\DTL.youtube\asl_alphabet_test\asl_alphabet_test"

def verify():
    print(f"--- Verifying Model on Dataset: {TEST_DATA_PATH} ---")
    
    try:
        classifier = HandSignClassifier(MODEL_PATH, LABELS_PATH, confidence_threshold=0.5)
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return

    files = [f for f in os.listdir(TEST_DATA_PATH) if f.lower().endswith('.jpg')]
    
    correct_count = 0
    total_count = len(files)
    
    print(f"{'Filename':<20} | {'Expected':<10} | {'Predicted':<10} | {'Confidence':<10}")
    print("-" * 60)
    
    for f in files:
        img_path = os.path.join(TEST_DATA_PATH, f)
        expected_label = f.split('_')[0]
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        predicted_label, confidence = classifier.predict(img)
        
        is_correct = predicted_label.lower() == expected_label.lower()
        if is_correct:
            correct_count += 1
            
        result_marker = "[OK]" if is_correct else "[FAIL]"
        print(f"{f:<20} | {expected_label:<10} | {predicted_label:<10} | {confidence:>8.1f}%  {result_marker}")

    print("-" * 60)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"Final Accuracy on Test Dataset: {accuracy:.2f}% ({correct_count}/{total_count})")

if __name__ == "__main__":
    verify()
