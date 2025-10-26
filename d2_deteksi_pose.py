import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba index 1/2.")

print("✅ Kamera berhasil dibuka! Tekan 'q' untuk keluar.")

# Loop untuk membaca frame kamera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ubah warna frame dari BGR ke RGB (dibutuhkan oleh MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses frame dengan MediaPipe
    results = pose.process(rgb_frame)

    # Jika terdeteksi pose, gambar titik dan garis kerangka tubuh
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    # Tampilkan hasil
    cv2.imshow("Deteksi Pose Tubuh - Tekan 'q' untuk keluar", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()