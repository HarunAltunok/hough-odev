# GEÇİCİ test resimleri üretir (gerçek fotoğraflar gelene kadar).
# a) beyaz kâğıt üstünde madeni paralar + cetvel
# b) iç içe çemberler (hedef tahtası) + tek başına duran bir madeni para
import cv2
import numpy as np

np.random.seed(0)  # her çalıştırmada aynı gürültü çıksın

# ---------- (a) paralar + cetvel ----------
resim = np.full((768, 1024, 3), 235, dtype=np.uint8)  # açık gri/beyaz kâğıt

# cetvel: eğik duran sarımsı bir dikdörtgen + çentikler
cetvel = np.array([[120, 600], [860, 420], [880, 500], [140, 680]], dtype=np.int32)
cv2.fillPoly(resim, [cetvel], (90, 190, 220))
for i in range(1, 30):  # cetvel üstündeki çentik çizgileri
    t = i / 30
    x1 = int(120 + t * 740)
    y1 = int(600 - t * 180)
    uzunluk = 30 if i % 5 == 0 else 15
    cv2.line(resim, (x1, y1), (x1 + int(uzunluk * 0.24), y1 + uzunluk), (40, 40, 40), 2)

# madeni paralar: (merkez x, merkez y, yarıçap)
paralar = [(200, 200, 60), (420, 160, 75), (650, 230, 55), (850, 180, 65), (560, 640, 50)]
for x, y, r in paralar:
    cv2.circle(resim, (x, y), r, (120, 140, 150), -1)          # paranın gövdesi
    cv2.circle(resim, (x, y), r - 8, (150, 165, 175), 2)       # kenar kabartması

gurultu = np.random.normal(0, 8, resim.shape)                   # kamera gürültüsü taklidi
resim = np.clip(resim + gurultu, 0, 255).astype(np.uint8)
cv2.imwrite("resimler/paralar_cetvel.jpg", resim)

# ---------- (b) iç içe çemberler + tek para ----------
resim = np.full((768, 1024, 3), 235, dtype=np.uint8)

# hedef tahtası: aynı merkezli 3 halka (renkler sırayla koyu/açık)
merkez = (340, 384)
for r, renk in [(150, (60, 60, 200)), (100, (230, 230, 230)), (50, (60, 60, 200))]:
    cv2.circle(resim, merkez, r, renk, -1)

# tek başına duran madeni para
cv2.circle(resim, (780, 400), 80, (120, 140, 150), -1)

gurultu = np.random.normal(0, 8, resim.shape)
resim = np.clip(resim + gurultu, 0, 255).astype(np.uint8)
cv2.imwrite("resimler/ic_ice_tek.jpg", resim)

print("Sentetik resimler resimler/ klasörüne yazıldı.")
