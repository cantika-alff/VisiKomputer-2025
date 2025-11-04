import cv2
from cvzone.HandTrackingModule import HandDetector

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi HandDetector
detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.5,
    minTrackCon=0.5
)

while True:
    ok, img = cap.read()
    if not ok:
        break

    # Deteksi tangan
    hands, img = detector.findHands(img, draw=True, flipType=True)  # flipType untuk tampilan mirror

    if hands:
        hand = hands[0]  # dict berisi "lmList", "bbox", dll.
        fingers = detector.fingersUp(hand)  # list panjang 5 berisi nilai 0/1
        count = sum(fingers)

        # Tampilkan jumlah jari yang terangkat
        cv2.putText(img, f"Fingers: {count}  {fingers}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow("Hands + Fingers", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepas kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
