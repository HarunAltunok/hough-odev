# MADDE 1: OpenCV ile doğru bulma (cv2.HoughLines ve cv2.HoughLinesP)
import time
import cv2
import numpy as np

# ---------------- AYARLAR ----------------
RESIM_YOLU = "resimler/paralar_cetvel.jpg"
CANNY_ALT = 50          # Canny alt eşik: bunun altındaki gradyanlar kenar değil
CANNY_UST = 150         # Canny üst eşik: bunun üstündekiler kesin kenar
ESIK_LINES = 40         # HoughLines: bir doğru için gereken en az oy sayısı
                        # (median kenarların çoğunu sildiği için düşük; 30'da sahte doğrular başlıyor)
ESIK_LINESP = 20        # HoughLinesP: bir parça için gereken en az oy sayısı
MIN_UZUNLUK = 40        # HoughLinesP: bundan kısa parçaları atla (piksel)
MAX_BOSLUK = 20         # HoughLinesP: aradaki boşluk bundan küçükse parçaları birleştir
                        # (median kenarı kesik kesik bıraktığı için boşluğu büyük tuttuk)
# -----------------------------------------

baslangic = time.perf_counter()  # süre ölçümü başlıyor

# 1) Resmi oku ve 1024x768'e getir
resim = cv2.imread(RESIM_YOLU)
resim = cv2.resize(resim, (1024, 768))
gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)  # Canny gri resim ister

# 2) Canny: Hough sadece kenar pikselleriyle oylama yapar
kenarlar_canny = cv2.Canny(gri, CANNY_ALT, CANNY_UST)

# 3) Median (3x3) KENAR resmine: tek başına duran gürültü kenar piksellerini siler.
#    Yan etkisi: 1 piksel kalınlığındaki ince kenarların çoğunu da siler, çünkü
#    çizgi pikselinin 3x3 komşuluğunda çoğunluk siyahtır (denemede %77'si gitti).
kenarlar = cv2.medianBlur(kenarlar_canny, 3)
print("Kenar pikseli: Canny sonrası", (kenarlar_canny > 0).sum(),
      "-> median sonrası", (kenarlar > 0).sum())
cv2.imwrite("outputs/madde1_kenarlar.png", cv2.hconcat([kenarlar_canny, kenarlar]))

# 4a) HoughLines: her doğruyu (rho, theta) olarak verir, sonsuz uzun doğru
dogrular = cv2.HoughLines(kenarlar, 1, np.pi / 180, ESIK_LINES)

# 4b) HoughLinesP: doğru PARÇALARINI uç noktalarıyla (x1, y1, x2, y2) verir
parcalar = cv2.HoughLinesP(kenarlar, 1, np.pi / 180, ESIK_LINESP,
                           minLineLength=MIN_UZUNLUK, maxLineGap=MAX_BOSLUK)

sure = time.perf_counter() - baslangic

# 5) Çiz: HoughLines KIRMIZI, HoughLinesP YEŞİL
cizim = resim.copy()
if dogrular is not None:
    for dogru in dogrular.reshape(-1, 2):  # her satır: [rho, theta]
        rho, theta = dogru
        # (rho, theta) doğrusunun üstünde iki uzak nokta bulup çizgi çekiyoruz
        x0 = rho * np.cos(theta)  # doğruya orijinden inen dikmenin ayağı
        y0 = rho * np.sin(theta)
        x1 = int(x0 - 2000 * np.sin(theta))
        y1 = int(y0 + 2000 * np.cos(theta))
        x2 = int(x0 + 2000 * np.sin(theta))
        y2 = int(y0 - 2000 * np.cos(theta))
        cv2.line(cizim, (x1, y1), (x2, y2), (0, 0, 255), 2)

if parcalar is not None:
    # reshape: OpenCV 4 (N,1,4), OpenCV 5 (N,4) döndürüyor; ikisini de (N,4) yapar
    for parca in parcalar.reshape(-1, 4):
        x1, y1, x2, y2 = parca
        cv2.line(cizim, (x1, y1), (x2, y2), (0, 255, 0), 3)

# 6) Kaydet ve sonucu yaz
cv2.imwrite("outputs/madde1_dogrular.png", cizim)
print("HoughLines doğru sayısı :", 0 if dogrular is None else len(dogrular))
print("HoughLinesP parça sayısı:", 0 if parcalar is None else len(parcalar))
print(f"Süre: {sure:.4f} saniye")
