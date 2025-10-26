import cv2
import mediapipe as mp
import warnings

# Sembunyikan warning agar tampilan bersih
warnings.filterwarnings("ignore")

# Inisialisasi modul MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba index 1/2.")

print("✅ Kamera berhasil dibuka! Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame agar mirror
    frame = cv2.flip(frame, 1)

    # Ubah BGR ke RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = pose.process(rgb)

    # Jika pose terdeteksi
    if hasil.pose_landmarks:
        # Gambar kerangka tubuh
        mp_drawing.draw_landmarks(
            frame, hasil.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
        )

        # Ambil landmark tertentu (contoh: hidung)
        lm = hasil.pose_landmarks.landmark
        hidung = lm[0]  # ID 0 = hidung

        # Ambil koordinat 3D
        x = round(hidung.x, 3)
        y = round(hidung.y, 3)
        z = round(hidung.z, 3)

        # Tampilkan nilai posisi 3D di layar
        teks = f"Posisi 3D (x={x}, y={y}, z={z})"
        cv2.putText(frame, teks, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Tampilkan hasil
    cv2.imshow("Estimasi Posisi Tubuh 3D (Tekan 'q' untuk keluar)", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
