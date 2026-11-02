# MADDE 3: OpenCV ile çember bulma (cv2.HoughCircles)
import time
import cv2

# ---------------- AYARLAR ----------------
RESIM_YOLU = "resimler/paralar_cetvel.jpg"
CANNY_ALT = 50          # Canny alt eşik
CANNY_UST = 150         # Canny üst eşik
MIN_MESAFE = 40         # iki çember merkezi arasındaki en az uzaklık (piksel)
OY_ESIGI = 15           # param2: merkez için gereken oy; küçülürse sahte çember artar
                        # (median kenarları seyrelttiği için düşük; 10'da sahte çemberler çıktı)
MIN_YARICAP = 30        # aranacak en küçük yarıçap (piksel)
MAX_YARICAP = 100       # aranacak en büyük yarıçap (piksel)
# -----------------------------------------

baslangic = time.perf_counter()

# 1) Resmi oku ve 1024x768'e getir
resim = cv2.imread(RESIM_YOLU)
resim = cv2.resize(resim, (1024, 768))
gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)

# 2) Canny: kenarları bul
kenarlar_canny = cv2.Canny(gri, CANNY_ALT, CANNY_UST)

# 3) Median (3x3) KENAR resmine: tek başına duran gürültü piksellerini siler
#    (ince kenarların bir kısmı da gider, çemberler kesik kesik kalır)
kenarlar = cv2.medianBlur(kenarlar_canny, 3)
print("Kenar pikseli: Canny sonrası", (kenarlar_canny > 0).sum(),
      "-> median sonrası", (kenarlar > 0).sum())
cv2.imwrite("outputs/madde3_kenarlar.png", cv2.hconcat([kenarlar_canny, kenarlar]))

# 4) HoughCircles: median'dan geçmiş kenar resmini veriyoruz. HOUGH_GRADIENT
#    yöntemi her kenar pikselinin gradyan yönünde merkez arar; param1 kendi
#    içindeki Canny'nin üst eşiği.
cemberler = cv2.HoughCircles(kenarlar, cv2.HOUGH_GRADIENT, 1, MIN_MESAFE,
                             param1=CANNY_UST, param2=OY_ESIGI,
                             minRadius=MIN_YARICAP, maxRadius=MAX_YARICAP)

sure = time.perf_counter() - baslangic

# 5) Çiz: her çember farklı renkte, merkezine küçük nokta
renkler = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]
cizim = resim.copy()
sayi = 0
if cemberler is not None:
    for i, cember in enumerate(cemberler.reshape(-1, 3)):  # her satır: [x, y, r]
        x, y, r = int(round(cember[0])), int(round(cember[1])), int(round(cember[2]))
        renk = renkler[i % len(renkler)]
        cv2.circle(cizim, (x, y), r, renk, 3)   # çemberin kendisi
        cv2.circle(cizim, (x, y), 3, renk, -1)  # merkez noktası
        print(f"Çember {i + 1}: merkez=({x}, {y}), yarıçap={r}")
        sayi += 1

# 6) Kaydet
cv2.imwrite("outputs/madde3_cemberler.png", cizim)
print("Bulunan çember sayısı:", sayi)
print(f"Süre: {sure:.4f} saniye")
