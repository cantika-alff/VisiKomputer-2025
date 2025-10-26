import cv2
import mediapipe as mp
import math
import warnings

# Hilangkan warning agar tampilan bersih
warnings.filterwarnings("ignore")

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Fungsi bantu menghitung sudut antara 3 titik
def hitung_sudut(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    sudut = abs(radians * 180.0 / math.pi)
    if sudut > 180.0:
        sudut = 360 - sudut
    return sudut

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba index 1/2.")

print("✅ Kamera berhasil dibuka! Tekan 'q' untuk keluar.")

# Variabel penghitung
count = 0
posisi = None  # bisa 'atas' atau 'bawah'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = pose.process(rgb)

    if hasil.pose_landmarks:
        # Gambar skeleton tubuh
        mp_drawing.draw_landmarks(
            frame, hasil.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
        )

        # Ambil titik bahu, siku, dan pergelangan tangan kanan
        bahu = hasil.pose_landmarks.landmark[12]
        siku = hasil.pose_landmarks.landmark[14]
        pergelangan = hasil.pose_landmarks.landmark[16]

        # Hitung sudut siku kanan
        sudut = hitung_sudut(bahu, siku, pergelangan)

        # Tampilkan sudut di layar
        cv2.putText(frame, f"Sudut siku kanan: {int(sudut)}derajat", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        # Logika sederhana untuk push-up counter
        if sudut > 160:
            posisi = "atas"
        if sudut < 70 and posisi == "atas":
            posisi = "bawah"
            count += 1  # naikkan hitungan

        # Tampilkan jumlah hitungan di layar
        cv2.putText(frame, f"Hitungan: {count}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

    # Tampilkan kamera
    cv2.imshow("Fitness Counter (Tekan 'q' untuk keluar)", frame)

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
