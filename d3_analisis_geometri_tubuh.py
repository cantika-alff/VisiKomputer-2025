import cv2
import mediapipe as mp
import math
import warnings

# Sembunyikan warning dari library
warnings.filterwarnings("ignore")

# Inisialisasi modul MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Fungsi untuk menghitung sudut antar tiga titik (misalnya bahu–siku–pergelangan)
def hitung_sudut(a, b, c):
    """
    a, b, c: tuple (x, y)
    """
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_cb = math.sqrt(cb[0]**2 + cb[1]**2)
    if mag_ab * mag_cb == 0:
        return 0
    cos_theta = dot / (mag_ab * mag_cb)
    return round(math.degrees(math.acos(cos_theta)))

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba index 1/2.")

print("✅ Kamera berhasil dibuka! Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror agar gerakan lebih natural
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = pose.process(rgb)

    if hasil.pose_landmarks:
        # Gambar kerangka tubuh
        mp_drawing.draw_landmarks(
            frame, hasil.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        # Ambil koordinat 3 titik: bahu kanan (12), siku kanan (14), pergelangan tangan kanan (16)
        lm = hasil.pose_landmarks.landmark
        bahu_kanan = (lm[12].x, lm[12].y)
        siku_kanan = (lm[14].x, lm[14].y)
        pergelangan_kanan = (lm[16].x, lm[16].y)

        # Hitung sudut siku kanan
        sudut_siku = hitung_sudut(bahu_kanan, siku_kanan, pergelangan_kanan)

        # Tampilkan nilai sudut di layar
        cv2.putText(
            frame,
            f"Sudut siku kanan: {sudut_siku} derajat",  # ✅ ganti simbol ° jadi teks agar terbaca
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

    # Tampilkan hasil
    cv2.imshow("Analisis Geometri Tubuh (Tekan 'q' untuk keluar)", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
