import cv2
import mediapipe as mp
import warnings

# Sembunyikan warning agar tidak mengganggu tampilan
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

# Variabel untuk menyimpan posisi sebelumnya
prev_y = None
status_gerakan = "Diam"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip agar mirror
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = pose.process(rgb)

    if hasil.pose_landmarks:
        # Gambar kerangka tubuh
        mp_drawing.draw_landmarks(
            frame, hasil.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
        )

        # Ambil posisi tangan kanan (pergelangan tangan, landmark 16)
        tangan_kanan = hasil.pose_landmarks.landmark[16]
        y = tangan_kanan.y  # posisi vertikal

        # Bandingkan dengan posisi sebelumnya untuk menentukan arah gerakan
        if prev_y is not None:
            if y < prev_y - 0.02:
                status_gerakan = "Tangan Naik ⬆️"
            elif y > prev_y + 0.02:
                status_gerakan = "Tangan Turun ⬇️"
            else:
                status_gerakan = "Diam ⏸️"

        prev_y = y  # simpan posisi sekarang untuk perbandingan selanjutnya

        # Tampilkan informasi gerakan di layar
        cv2.putText(frame, f"Gerakan: {status_gerakan}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    # Tampilkan hasil kamera
    cv2.imshow("Analisis Gerakan Tubuh (Tekan 'q' untuk keluar)", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
