# ADR-010: Hiyerarşik Navigasyon ve Mesafe Curriculum'u

**Durum:** Kabul edildi  
**Tarih:** 11 Eylül 2026  
**İlişki:** ADR-007 sahne curriculum'unu görev mesafesi ve büyük alan desteğiyle
genişletir; yerel algı ADR-011 uyarınca kamera tabanlıdır. Episode ve zaman
bütçesi ayrıntıları ADR-013'te sabitlenmiştir.

## Bağlam

Mevcut baseline `6 x 5 x 2.5 m` çalışma hacmi, `4 m` sabit başlangıç-hedef
mesafesi, `1.5 m/s` yatay hız, `20 s` episode sınırı ve `5 m` rangefinder
menzili için tasarlanmıştır. Observation v1 hedef konumunu ve mesafesini
`6 m` ölçeğiyle normalize edip `[-1, 1]` aralığında kırpar.

Bu sabitleri yalnızca yüzlerce metre veya kilometre düzeyine çıkarmak yeterli
değildir:

- Hedef uzaklığı `6 m` sonrasında observation'da ayırt edilemez hâle gelir.
- `1.5 m/s` hızla `1 km` uçuşun teorik alt sınırı yaklaşık `667 s` ve `333,500`
  fizik adımıdır.
- `0.10 m` çözünürlüklü `1 km x 1 km` occupancy grid yaklaşık `100 milyon`
  hücre üretir; mevcut Python aramasıyla episode başına kullanılması uygun
  değildir.
- Uzak terminal hedef ödülü PPO için gereksiz derecede uzun kredi atama problemi
  oluşturur.
- Daha yüksek hızda sabit `5 m` algı menzili, frenleme ve gecikme payı için
  yetersiz kalabilir.

## Karar

Kilometre ölçekli görev, tek bir dev uçtan uca PPO episode'u olarak değil üç
katmanlı hiyerarşiyle ele alınacaktır:

```text
Global görev/harita ve rota üretici (yaklaşık 1-5 Hz)
                    |
                    v
       Yerel waypoint veya güvenli koridor
                    |
                    v
  PPO yerel navigasyon ve kaçınma politikası (20 Hz)
                    |
                    v
       Klasik uçuş kontrol zinciri (500 Hz)
```

- Global katman kilometre koordinatlarını, görev rotasını, coğrafi sınırları ve
  waypoint sırasını yönetir.
- PPO, aracın çevresindeki sınırlı ve ego-merkezli bir pencere içinde bir
  sonraki waypoint'e giderken engelden kaçar.
- MuJoCo'ya yalnızca yakın çevrede fiziksel etkileşime girebilecek engeller
  yüklenir. Uzak dünya ayrı bir seyrek sahne/harita temsili olarak tutulur.
- Büyük alan rota doğrulaması kaba çözünürlüklü global planlayıcıyla; yakın
  açıklık doğrulaması yüksek çözünürlüklü yerel grid veya sürekli geometriyle
  yapılır.
- Mevcut `EbabilNavigation-v0` ve observation v1, Faz 6 baseline
  tekrarlanabilirliği için dondurulur. Ölçeklenebilir gözlem sözleşmesi yeni
  environment/observation sürümüyle gelir.
- Yerel PPO eğitimi kısa segment episode'ları kullanır; D3-D5 uzun görevleri
  segmentleri fizik durumunu sıfırlamadan zincirleyen mission evaluation ile
  ölçülür.
- Episode zaman sınırı rota uzunluğu ve nominal hızdan türetilir; ayrıntılı
  formül, Gymnasium semantiği ve kalite kapıları ADR-013 içindedir.

Yeni observation sürümünde nihai hedefin kırpılmış Kartezyen farkı yerine en az
şunlar değerlendirilmelidir:

- body-frame birim waypoint yönü,
- logaritmik veya parça-bazlı normalize waypoint mesafesi,
- yerel waypoint/koridor durumu,
- hızla uyumlu yerel engel algısı,
- gerekirse önceki action/observation geçmişi.

Rangefinder veya yerel algı menzili sabit bir görsel tercih olarak değil,
yaklaşık durma mesafesini karşılayacak şekilde seçilmelidir:

```text
gerekli_menzil >= v^2 / (2 * a_fren) + v * toplam_gecikme + güvenlik_payı
```

## Mesafe curriculum'u

Engel zorluğu ve görev mesafesi birbirinden bağımsız eksenler olacaktır.
Başlangıç önerisi:

| Seviye | Görev ölçeği | Eğitim amacı |
|---|---:|---|
| D0-V | `4-6 m` | Legacy geometriyi `vision-v1` ile yeniden öğrenmek |
| D1 | `5-15 m` | Tek blocker ve yerel waypoint davranışı |
| D2 | `15-50 m` | Seyrek engeller ve değişken endpoint'ler |
| D3 | `50-150 m` | Birden fazla waypoint/yerel pencere geçişi |
| D4 | `150-500 m` | Bölgesel rüzgâr, sensör ve rota çeşitliliği |
| D5 | `0.5-2 km` | Yerel politikanın birleşik görev üzerindeki başarısı |

Geçiş yalnızca eğitim reward'una bağlanmaz. Kayan evaluation penceresinde başarı,
çarpışma, minimum açıklık, rota verimi ve timeout oranı birlikte eşiklenir.
Eski seviyeler catastrophic forgetting'i ölçmek ve azaltmak için eğitim
karışımında tutulur.

Mesafe, engel yoğunluğu, geçit genişliği, hız, sensör bozulması ve rüzgâr aynı
anda artırılmaz. Her eksen ayrı loglanır ve curriculum kapalı/açık ablation'ı
yapılır.

## Sonuçlar

- Aynı yerel PPO politikası farklı uzunluktaki görevlerde yeniden kullanılabilir.
- PPO'nun uzun menzilli rota planlamayı ve yakın engel kaçınmayı aynı ağda
  öğrenmesi beklenmez; modüller ayrı test edilebilir.
- Kilometre ölçeği MuJoCo koordinat sınırından çok sahne temsili, algı menzili,
  episode ufku ve eğitim maliyeti problemidir.
- Observation sözleşmesi değişeceği için eski checkpoint'ler yeni ortamda
  doğrudan kullanılamaz; Faz 6 artefact'ları karşılaştırma baseline'ı olarak
  saklanır.

## Uygulama doğrulama işleri

1. Yerel pencere yarıçapı, seyir hızı ve sensör menzili için fiziksel bütçe
   çalışması yapmak.
2. `10-50 m` ölçeğinde waypoint arayüzü ve observation v2 prototipi hazırlamak.
3. Büyük sahne için kaba global planlayıcı ile yerel geometri yükleme maliyetini
   benchmark etmek.
4. D0-V–D2 curriculum smoke eğitiminde öğrenme ve forgetting eğrisini ölçmek.
5. Sonuçlara göre D3-D5 mission-level evaluation tasarımını kesinleştirmek.
