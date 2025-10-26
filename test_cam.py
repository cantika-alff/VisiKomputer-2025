import cv2

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("❌ Kamera tidak bisa diakses. Coba ganti index ke 1 atau 2.")
else:
    print("✅ Kamera berhasil dibuka!")
    print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("⚠️ Gagal membaca frame dari kamera.")
        break

    cv2.imshow("Tes Kamera (real-time)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
