# MADDE 4: Kendi Hough çember kodum (cv2.HoughCircles KULLANILMADI)
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- AYARLAR ----------------
RESIM_YOLU = "resimler/paralar_cetvel.jpg"
KUCULTME = 0.5          # 1024x768 resmi bu oranla küçült (saf Python döngüsü yavaş)
CANNY_ALT = 50          # Canny alt eşik
CANNY_UST = 150         # Canny üst eşik
MIN_YARICAP = 22        # aranacak en küçük yarıçap (KÜÇÜLTÜLMÜŞ resimde, piksel)
MAX_YARICAP = 40        # aranacak en büyük yarıçap (KÜÇÜLTÜLMÜŞ resimde, piksel)
ESIK = 140              # bir çember için gereken en az oy (en fazla 360 olabilir)
MIN_MESAFE = 15         # merkezleri bundan yakın çemberler aynı çember sayılır
# -----------------------------------------

# 1) Resmi oku, 1024x768'e getir, sonra hız için küçült
resim = cv2.imread(RESIM_YOLU)
resim = cv2.resize(resim, (1024, 768))
kucuk = cv2.resize(resim, None, fx=KUCULTME, fy=KUCULTME)
gri = cv2.cvtColor(kucuk, cv2.COLOR_BGR2GRAY)
yukseklik, genislik = gri.shape

# 2) Canny: oylamaya sadece kenar pikselleri katılacak (bu maddede median istenmiyor)
kenarlar = cv2.Canny(gri, CANNY_ALT, CANNY_UST)

baslangic = time.perf_counter()

# 4) HOUGH OYLAMASI
# sin/cos tablosu 0..359 derece için döngüden ÖNCE bir kez hesaplanır
cos_tablosu = []
sin_tablosu = []
for aci in range(360):
    radyan = aci * np.pi / 180
    cos_tablosu.append(np.cos(radyan))
    sin_tablosu.append(np.sin(radyan))

# 3 boyutlu oylama matrisi: [a (merkez x), b (merkez y), r (yarıçap)]
# Çemberin 3 bilinmeyeni var (merkez x, merkez y, yarıçap), bu yüzden 3 boyut.
yaricap_sayisi = MAX_YARICAP - MIN_YARICAP + 1
oylama_matrisi = np.zeros((genislik, yukseklik, yaricap_sayisi), dtype=np.int32)

kenar_y, kenar_x = np.nonzero(kenarlar)
print("Kenar piksel sayısı:", len(kenar_x))

# Her kenar pikseli (x, y), kendisinden r uzaklıktaki bütün olası merkezlere oy verir:
# a = x - r*cos, b = y - r*sin. Gerçek merkez, çemberdeki her pikselden oy alır.
for i in range(len(kenar_x)):
    x = kenar_x[i]
    y = kenar_y[i]
    for r_indeks in range(yaricap_sayisi):
        r = MIN_YARICAP + r_indeks
        for aci in range(360):
            a = int(round(x - r * cos_tablosu[aci]))
            b = int(round(y - r * sin_tablosu[aci]))
            if 0 <= a < genislik and 0 <= b < yukseklik:  # merkez resmin içindeyse
                oylama_matrisi[a, b, r_indeks] += 1

# ÇEMBERLERİ SEÇ: eşik üstü hücreler aday; en çok oydan başla,
# daha önce seçilmiş bir merkeze çok yakın olanı atla
adaylar = []
for a in range(genislik):
    for b in range(yukseklik):
        for r_indeks in range(yaricap_sayisi):
            oy = oylama_matrisi[a, b, r_indeks]
            if oy >= ESIK:
                adaylar.append((oy, a, b, MIN_YARICAP + r_indeks))
adaylar.sort(reverse=True)

cemberler = []
for oy, a, b, r in adaylar:
    yakin_var = False
    for _, a2, b2, _ in cemberler:
        if (a - a2) ** 2 + (b - b2) ** 2 < MIN_MESAFE ** 2:
            yakin_var = True
    if not yakin_var:
        cemberler.append((oy, a, b, r))

sure = time.perf_counter() - baslangic

# 5a) En çok oy alan r katmanını resim olarak kaydet
en_iyi_r_indeks = 0
for r_indeks in range(yaricap_sayisi):
    if oylama_matrisi[:, :, r_indeks].max() > oylama_matrisi[:, :, en_iyi_r_indeks].max():
        en_iyi_r_indeks = r_indeks
katman = oylama_matrisi[:, :, en_iyi_r_indeks].T  # .T: [a, b] -> [satır=b, sütun=a]
plt.figure(figsize=(8, 6))
goruntu = plt.imshow(katman, cmap="hot")
plt.colorbar(goruntu, label="oy sayısı")
plt.title(f"Oylama matrisi, r = {MIN_YARICAP + en_iyi_r_indeks} katmanı (küçük resimde)")
plt.xlabel("a (merkez x)")
plt.ylabel("b (merkez y)")
plt.savefig("outputs/madde4_oylama_katmani.png", dpi=100)
plt.close()

# 5b) Bulunan çemberleri ORİJİNAL (1024x768) resme çiz: koordinatları geri büyüt
renkler = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]
cizim = resim.copy()
for i, (oy, a, b, r) in enumerate(cemberler):
    x = int(a / KUCULTME)
    y = int(b / KUCULTME)
    r_buyuk = int(r / KUCULTME)
    renk = renkler[i % len(renkler)]
    cv2.circle(cizim, (x, y), r_buyuk, renk, 3)
    cv2.circle(cizim, (x, y), 3, renk, -1)
    print(f"Çember {i + 1}: merkez=({x}, {y}), yarıçap={r_buyuk}, oy={oy}")

# 6) Kaydet
cv2.imwrite("outputs/madde4_cemberler.png", cizim)
print("Bulunan çember sayısı:", len(cemberler))
print(f"Süre (oylama + seçme): {sure:.2f} saniye")
