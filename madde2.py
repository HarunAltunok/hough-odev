# MADDE 2: Kendi Hough doğru kodum (cv2.HoughLines KULLANILMADI)
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- AYARLAR ----------------
RESIM_YOLU = "resimler/paralar_cetvel.jpg"
CANNY_ALT = 50          # Canny alt eşik
CANNY_UST = 150         # Canny üst eşik
ESIK = 150              # bir doğru sayılmak için gereken en az oy (madde 1 ile aynı)
YAKIN_RHO = 10          # birbirine bu kadar yakın tepeler aynı doğru sayılır (piksel)
YAKIN_THETA = 5         # ... ve bu kadar yakın açılar (derece)
# -----------------------------------------

# 1) Resmi oku ve 1024x768'e getir
resim = cv2.imread(RESIM_YOLU)
resim = cv2.resize(resim, (1024, 768))
gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
yukseklik, genislik = gri.shape

# 2) Canny: oylamaya sadece kenar pikselleri katılacak
#    (ödev metni bu maddede median istemiyor; median ince kenarları da siliyor, bkz. madde1.py)
kenarlar = cv2.Canny(gri, CANNY_ALT, CANNY_UST)

baslangic = time.perf_counter()  # süreyi sadece Hough kısmı için ölçüyoruz

# 4) HOUGH OYLAMASI
# theta: 0..179 derece (180 sütun). Doğruyu tarif etmek için 180 derece yeterli,
# çünkü theta+180 aynı doğruyu -rho ile verir.
theta_sayisi = 180
# sin ve cos tablosu: döngüden ÖNCE bir kez hesaplanır. Döngüde her seferinde
# np.cos çağırmak yerine tablodan okumak çok daha hızlı.
cos_tablosu = []
sin_tablosu = []
for theta in range(theta_sayisi):
    radyan = theta * np.pi / 180
    cos_tablosu.append(np.cos(radyan))
    sin_tablosu.append(np.sin(radyan))

# rho, -kosegen ile +kosegen arasında olabilir. Dizi indeksi negatif olamayacağı
# için rho'ya kosegen ekleyip öyle saklıyoruz (kaydırma).
kosegen = int(np.ceil(np.sqrt(genislik ** 2 + yukseklik ** 2)))
oylama_matrisi = np.zeros((2 * kosegen + 1, theta_sayisi), dtype=np.int32)

# kenar piksellerinin koordinatları (y = satır, x = sütun)
kenar_y, kenar_x = np.nonzero(kenarlar)
print("Kenar piksel sayısı:", len(kenar_x))

# her kenar pikseli, içinden geçebilecek BÜTÜN doğrulara (her theta için bir tane) oy verir
for i in range(len(kenar_x)):
    x = kenar_x[i]
    y = kenar_y[i]
    for theta in range(theta_sayisi):
        rho = x * cos_tablosu[theta] + y * sin_tablosu[theta]
        rho_indeks = int(round(rho)) + kosegen  # negatif olmasın diye kaydır
        oylama_matrisi[rho_indeks, theta] += 1

# TEPELERİ BUL: eşik üstündeki hücreler aday; en çok oy alandan başlayarak seç,
# zaten seçilmiş bir tepeye çok yakın olanı atla (aynı doğrunun komşu hücreleri)
adaylar = []
for rho_indeks in range(oylama_matrisi.shape[0]):
    for theta in range(theta_sayisi):
        oy = oylama_matrisi[rho_indeks, theta]
        if oy >= ESIK:
            adaylar.append((oy, rho_indeks, theta))
adaylar.sort(reverse=True)  # en çok oy alan en başa

tepeler = []
for oy, rho_indeks, theta in adaylar:
    yakin_var = False
    for _, r2, t2 in tepeler:
        if abs(rho_indeks - r2) <= YAKIN_RHO and abs(theta - t2) <= YAKIN_THETA:
            yakin_var = True
    if not yakin_var:
        tepeler.append((oy, rho_indeks, theta))

sure = time.perf_counter() - baslangic

# 5a) Oylama matrisini resim olarak kaydet; tepeleri numaralandır
#     Matris 2561 satır; ekrana sığdırınca küçük tepeler kayboluyor. Sadece GÖRSEL
#     için her hücreyi 9x9 komşuluğunun en büyüğüyle değiştiriyoruz (dilate):
#     tepeler büyük parlak lekeye dönüşüp gözle sayılabiliyor.
gosterim = cv2.dilate(oylama_matrisi.astype(np.float32), np.ones((9, 9)))
plt.figure(figsize=(8, 8))
goruntu = plt.imshow(gosterim, cmap="hot", aspect="auto",
                     extent=(0, theta_sayisi, kosegen, -kosegen))  # eksenler gerçek rho/theta
for numara, (oy, rho_indeks, theta) in enumerate(tepeler, start=1):
    rho = rho_indeks - kosegen
    plt.scatter(theta, rho, s=300, facecolors="none", edgecolors="cyan")  # tepeyi halkala
    plt.text(theta + 4, rho, str(numara), color="cyan", fontsize=16, weight="bold")
plt.xlabel("theta (derece)")
plt.ylabel("rho (piksel)")
plt.title(f"Oylama matrisi - eşik {ESIK} üstü {len(tepeler)} tepe = {len(tepeler)} doğru")
plt.colorbar(goruntu, label="oy sayısı")
plt.savefig("outputs/madde2_oylama_matrisi.png", dpi=100)
plt.close()

# 5b) Bulunan doğruları orijinal resme çiz, aynı numarayı yaz
cizim = resim.copy()
for numara, (oy, rho_indeks, theta) in enumerate(tepeler, start=1):
    rho = rho_indeks - kosegen
    c = cos_tablosu[theta]
    s = sin_tablosu[theta]
    x0 = rho * c  # orijinden doğruya inen dikmenin ayağı
    y0 = rho * s
    # doğru yönü (-s, c); iki uzak nokta alıp çiziyoruz
    x1, y1 = int(x0 - 2000 * s), int(y0 + 2000 * c)
    x2, y2 = int(x0 + 2000 * s), int(y0 - 2000 * c)
    cv2.line(cizim, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # numarayı doğrunun resim içinde kalan kısmının ortasına yaz
    icerdeki_noktalar = []
    for t in range(-2000, 2000, 5):
        px = int(x0 - t * s)
        py = int(y0 + t * c)
        if 0 <= px < genislik and 0 <= py < yukseklik:
            icerdeki_noktalar.append((px, py))
    if len(icerdeki_noktalar) > 0:
        px, py = icerdeki_noktalar[len(icerdeki_noktalar) // 2]
        cv2.putText(cizim, str(numara), (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 0, 0), 3)

# 6) Kaydet ve yaz
cv2.imwrite("outputs/madde2_dogrular.png", cizim)
for numara, (oy, rho_indeks, theta) in enumerate(tepeler, start=1):
    print(f"Doğru {numara}: rho={rho_indeks - kosegen}, theta={theta} derece, oy={oy}")
print("Bulunan doğru sayısı:", len(tepeler))
print(f"Süre (oylama + tepe bulma): {sure:.2f} saniye")
