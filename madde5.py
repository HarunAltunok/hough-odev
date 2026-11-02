# MADDE 5: Sadece iç içe (aynı merkezli) çemberleri bulma
#
# TASARIM: Madde 4'teki 3 boyutlu oylamayı [a, b, r] aynen yapıyoruz. Sonra her
# merkez (a, b) için ikinci bir oylama yapıyoruz: o merkezin PUANI = o merkezde
# yeterli oy alan FARKLI yarıçap sayısı. Tek duran bir çember, merkezinde sadece
# 1 yarıçapa oy toplar, puanı 1 olur ve "en az 2 yarıçap" şartını geçemez.
# Çemberleri hiçbir zaman tek tek bulup listelemiyoruz; karar doğrudan oylama
# matrisinden, merkez başına verilir. Bu yüzden post processing değildir.
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- AYARLAR ----------------
RESIM_YOLU = "resimler/ic_ice_tek.jpg"
KUCULTME = 0.5          # 1024x768 -> 512x384 (saf Python döngüsü yavaş; 0.25'te
                        # kenarlar kalınlaşıp sahte merkezler çıktı, 0.5 temiz)
CANNY_ALT = 50          # Canny alt eşik
CANNY_UST = 150         # Canny üst eşik
MIN_YARICAP = 20        # aranacak en küçük yarıçap (küçük resimde, piksel)
MAX_YARICAP = 80        # aranacak en büyük yarıçap (küçük resimde, piksel)
ORAN = 0.5              # bir yarıçap "yeterli oy aldı" sayılır: oy >= ORAN * çevre (2*pi*r)
YARICAP_FARKI = 4       # birbirine bundan yakın yarıçaplar aynı halka sayılır (r, r+1 ...)
EN_AZ_YARICAP = 2       # bir merkez seçilmek için en az kaç farklı yarıçap gerekli
MIN_MESAFE = 10         # merkezleri bundan yakın gruplar aynı grup sayılır
# -----------------------------------------

# 1) Resmi oku, 1024x768'e getir, hız için küçült
resim = cv2.imread(RESIM_YOLU)
resim = cv2.resize(resim, (1024, 768))
kucuk = cv2.resize(resim, None, fx=KUCULTME, fy=KUCULTME)
gri = cv2.cvtColor(kucuk, cv2.COLOR_BGR2GRAY)
yukseklik, genislik = gri.shape

# 2) Canny: oylamaya sadece kenar pikselleri katılacak (bu maddede median istenmiyor)
kenarlar = cv2.Canny(gri, CANNY_ALT, CANNY_UST)

baslangic = time.perf_counter()

# 4a) MADDE 4 İLE AYNI 3B OYLAMA
cos_tablosu = []
sin_tablosu = []
for aci in range(360):  # sin/cos tablosu döngüden önce bir kez
    radyan = aci * np.pi / 180
    cos_tablosu.append(np.cos(radyan))
    sin_tablosu.append(np.sin(radyan))

yaricap_sayisi = MAX_YARICAP - MIN_YARICAP + 1
oylama_matrisi = np.zeros((genislik, yukseklik, yaricap_sayisi), dtype=np.int32)

kenar_y, kenar_x = np.nonzero(kenarlar)
print("Kenar piksel sayısı:", len(kenar_x))

for i in range(len(kenar_x)):
    x = kenar_x[i]
    y = kenar_y[i]
    for r_indeks in range(yaricap_sayisi):
        r = MIN_YARICAP + r_indeks
        for aci in range(360):
            a = int(round(x - r * cos_tablosu[aci]))
            b = int(round(y - r * sin_tablosu[aci]))
            if 0 <= a < genislik and 0 <= b < yukseklik:
                oylama_matrisi[a, b, r_indeks] += 1

# 4b) MERKEZ OYLAMASI: her merkezin puanı = yeterli oy alan farklı yarıçap sayısı
merkez_puani = np.zeros((genislik, yukseklik), dtype=np.int32)
merkez_yaricaplari = {}  # (a, b) -> o merkezde oy alan yarıçaplar (çizim için)
en_kucuk_gereken = ORAN * 2 * np.pi * MIN_YARICAP  # en küçük yarıçabın eşiği
for a in range(genislik):
    for b in range(yukseklik):
        # hızlandırma: bu merkezin hiçbir yarıçapı en küçük eşiği bile geçemiyorsa atla
        if oylama_matrisi[a, b].max() < en_kucuk_gereken:
            continue
        yaricaplar = []
        for r_indeks in range(yaricap_sayisi):
            r = MIN_YARICAP + r_indeks
            oy = oylama_matrisi[a, b, r_indeks]
            gereken = ORAN * 2 * np.pi * r  # büyük çember daha çok pikselden oy alır
            if oy >= gereken:
                # r, r+1 gibi komşu yarıçaplar aynı halkadır; yalnız bir kez say
                if len(yaricaplar) == 0 or r - yaricaplar[-1] >= YARICAP_FARKI:
                    yaricaplar.append(r)
        merkez_puani[a, b] = len(yaricaplar)
        if len(yaricaplar) >= EN_AZ_YARICAP:
            merkez_yaricaplari[(a, b)] = yaricaplar

# puanı EN_AZ_YARICAP ve üstü olan merkezler; en yüksek puandan başla, yakın olanı atla
adaylar = []
for (a, b), yaricaplar in merkez_yaricaplari.items():
    adaylar.append((len(yaricaplar), a, b))
adaylar.sort(reverse=True)

gruplar = []
for puan, a, b in adaylar:
    yakin_var = False
    for _, a2, b2 in gruplar:
        if (a - a2) ** 2 + (b - b2) ** 2 < MIN_MESAFE ** 2:
            yakin_var = True
    if not yakin_var:
        gruplar.append((puan, a, b))

sure = time.perf_counter() - baslangic

# 5a) Merkez puanı haritasını kaydet (iç içe merkez parlak, tek çember merkezi sönük)
#     (sadece görsel için 9x9 dilate: tek piksellik merkezler gözle görülsün)
gosterim = cv2.dilate(merkez_puani.T.astype(np.float32), np.ones((9, 9)))
plt.figure(figsize=(8, 6))
goruntu = plt.imshow(gosterim, cmap="hot", vmin=0, vmax=3)
for puan, a, b in gruplar:
    plt.text(a + 8, b - 8, f"puan {puan} (seçildi)", color="cyan", fontsize=12)
plt.colorbar(goruntu, label="yeterli oy alan farklı yarıçap sayısı")
plt.title(f"Merkez puanı (seçilmek için en az {EN_AZ_YARICAP})")
plt.xlabel("a (merkez x)")
plt.ylabel("b (merkez y)")
plt.savefig("outputs/madde5_merkez_puani.png", dpi=100)
plt.close()

# 5b) Bulunan grupları orijinal resme çiz: bir grubun bütün halkaları aynı renkte
renkler = [(0, 200, 0), (255, 0, 0), (255, 0, 255)]
cizim = resim.copy()
for i, (puan, a, b) in enumerate(gruplar):
    renk = renkler[i % len(renkler)]
    x = int(a / KUCULTME)
    y = int(b / KUCULTME)
    for r in merkez_yaricaplari[(a, b)]:
        cv2.circle(cizim, (x, y), int(r / KUCULTME), renk, 4)
    cv2.circle(cizim, (x, y), 4, renk, -1)
    yaricap_listesi = [int(r / KUCULTME) for r in merkez_yaricaplari[(a, b)]]
    print(f"Grup {i + 1}: merkez=({x}, {y}), {puan} halka, yarıçaplar={yaricap_listesi}")

# 6) Kaydet
cv2.imwrite("outputs/madde5_ic_ice.png", cizim)
print("Bulunan iç içe grup sayısı:", len(gruplar))
print(f"Süre (oylama + merkez puanı): {sure:.2f} saniye")
