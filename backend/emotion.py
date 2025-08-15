import cv2
from fer import FER

def capture_emotion():
    # Try different camera indices if needed
    for cam_index in range(3):
        cam = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if cam.isOpened():
            print(f"📷 Using camera index: {cam_index}")
            break
        cam.release()
    else:
        print("❌ Cannot open any webcam.")
        return None

    detector = FER(mtcnn=True)

    print("📸 Capturing image... Press 'c' to capture or 'q' to quit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Failed to grab frame from webcam.")
            break

        # Show live camera feed
        cv2.imshow("Camera - Press 'c' to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Capture on 'c'
            cam.release()
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):
            # Quit without capturing
            cam.release()
            cv2.destroyAllWindows()
            print("❌ Capture canceled by user.")
            return None

    # Detect top emotion
    result = detector.top_emotion(frame)
    if not result or result[0] is None:
        print("😐 No emotion detected.")
        return None

    emotion, score = result
    print(f"😊 Detected Emotion: {emotion} (confidence: {score:.2f})")
    return emotion

if __name__ == "__main__":
    capture_emotion()