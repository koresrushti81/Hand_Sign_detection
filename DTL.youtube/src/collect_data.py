"""
Hand Sign Data Collection Script
CLI-based data collection tool
"""

import cv2
import os
import time
import config
from data_collector import HandSignDataCollector


def main():
    """Main function for data collection"""
    print("\n🖐️ Hand Sign Data Collection Tool")
    print("=" * 60)
    
    collector = HandSignDataCollector()
    
    while True:
        print("\n📋 Options:")
        print("1. View available hand signs")
        print("2. Start new collection")
        print("3. Delete hand sign folder")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            _view_available_signs(collector)
        elif choice == "2":
            _collect_new_sign(collector)
        elif choice == "3":
            _delete_sign(collector)
        elif choice == "4":
            print("✅ Goodbye!")
            break
        else:
            print("❌ Invalid option")


def _view_available_signs(collector):
    signs = collector.get_available_hand_signs()
    print("\n📊 Available Hand Signs:")
    print("-" * 40)
    if not signs:
        print("No hand signs collected yet")
    else:
        for sign in signs:
            count = collector.get_collection_stats(sign)
            print(f"  {sign:20} - {count:3} images")
    print("-" * 40)


def _collect_new_sign(collector):
    hand_sign_name = input("\nEnter hand sign name: ").strip()
    if not hand_sign_name: return
    
    if not collector.start_collection(hand_sign_name):
        print("❌ Failed to start collection")
        return
    
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    print(f"\n✅ Collecting '{hand_sign_name}' | 's': Save, 'SPACE': Auto-save, 'q': Quit")
    
    auto_save = False
    last_save_time = time.time()
    
    try:
        while True:
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            
            result = collector.collect_frame(frame)
            
            if result['detected']:
                x1, y1, x2, y2 = result['bbox']
                color = config.COLOR_GREEN if auto_save else config.COLOR_CYAN
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                if auto_save and time.time() - last_save_time >= 0.5:
                    collector.save_image(result['processed_image'])
                    last_save_time = time.time()
            
            cv2.putText(frame, f"Count: {collector.counter}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Collection Mode", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s') and result['detected']:
                collector.save_image(result['processed_image'])
                print(f"Saved: {collector.counter}")
            elif key == ord(' '):
                auto_save = not auto_save
                print(f"Auto-save: {'ON' if auto_save else 'OFF'}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        collector.stop_collection()


def _delete_sign(collector):
    signs = collector.get_available_hand_signs()
    if not signs: return
    
    for i, sign in enumerate(signs, 1):
        print(f"{i}. {sign}")
    
    try:
        idx = int(input("Select sign to delete: ")) - 1
        if 0 <= idx < len(signs):
            import shutil
            shutil.rmtree(os.path.join(config.DATA_IMG_BASE_PATH, signs[idx]))
            print(f"Deleted {signs[idx]}")
    except:
        print("Invalid input")


if __name__ == "__main__":
    main()
