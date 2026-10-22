# ADR-013: Segmentli Episode ve Mesafeye Bağlı Zaman Bütçesi

**Durum:** Kabul edildi  
**Tarih:** 11 Eylül 2026  
**İlişki:** ADR-010 içindeki hiyerarşik mesafe curriculum'unu ve ADR-012
içindeki hedef bölge hover görevini somutlaştırır.

## Bağlam

Legacy `EbabilNavigation-v0`, `20 Hz` politika frekansında `20 s / 400 adım`
sabit episode sınırı kullanır. Bu sınır yaklaşık `4-6 m` görevler için yeterlidir
fakat harita büyüdükçe iki zıt sorun oluşur:

- sabit `20 s`, uzun ve geçerli rotaları yanlış biçimde timeout yapar;
- bütün kilometre görevini tek episode'a çevirmek kredi atama ufkunu, simülasyon
  maliyetini ve başarısız episode başına kaybedilen örneği aşırı büyütür.

PPO rollout uzunluğu ile fiziksel görevin episode uzunluğu aynı kavram değildir.
Harita büyüdüğü için rollout'u ve yerel politika hafızasını kilometre boyunca
doğrusal artırmak gerekmemelidir.

## Karar

İki ayrı çalışma modu kullanılacaktır:

1. **Yerel eğitim episode'u:** PPO, sınırlı uzunluktaki bir rota segmentinde
   mevcut waypoint'e ulaşmayı, kamerayla engelden kaçınmayı ve terminal
   segmentte hedef üzerinde hover'a yerleşmeyi öğrenir.
2. **Mission-level evaluation:** Global rota yöneticisi segmentleri aynı fizik
   durumu içinde kesintisiz zincirler. Ara waypoint'e ulaşmak episode'u
   bitirmez; yalnız waypoint indeksi ve yerel sahne penceresi ilerler. Nihai
   başarı ADR-012'deki hover hold koşuluyla oluşur.

Bu ayrım sayesinde eğitim örnekleri kısa ve çeşitli kalırken uzun görevde
birikimli localization drift'i, waypoint geçişleri ve toplam görev süresi yine
ölçülebilir.

## Zaman bütçesi

Yeni vision ortamında sabit bir global `max_episode_steps` yerine rota temelli
bütçe hesaplanacaktır:

```text
T_segment = clamp(
    T_fixed + k_time * L_plan / max(v_nominal, v_floor) + T_terminal,
    T_min,
    T_max,
)

N_segment = ceil(T_segment * policy_frequency_hz)
```

- `L_plan`: düz çizgi mesafesi değil, global/yerel planlayıcının tahmini rota
  uzunluğudur.
- `k_time > 1`: sapma, kaçınma ve toparlanma payıdır.
- `T_fixed`: kalkış/yerleşme ve waypoint geçiş payıdır.
- `T_terminal`: yalnız nihai hedef segmentinde yaklaşma ve hover hold payıdır.
- `T_min/T_max`: bozuk rota tahmininin çok kısa veya sınırsız episode üretmesini
  engeller.

İlk simülasyon başlangıç değerleri ölçümle ayarlanmak üzere şöyledir:

| Parametre | Başlangıç değeri |
|---|---:|
| Yerel segment plan uzunluğu | `5-30 m` |
| `v_nominal` | `1.5 m/s` |
| `k_time` | `1.75` |
| `T_fixed` | `5 s` |
| Terminal yaklaşma payı | `10 s` |
| Hover hold | `3 s` |
| Yerel `T_min` | `20 s` |
| Yerel `T_max` | `60 s` |

Bu değerler güvenlik sınırı değil başlangıç hipotezidir. Legacy ortamın
`20 s / 400 adım` sözleşmesi değiştirilmez. Aynı `4-6 m` geometriyi kamera
girdisiyle yeniden kuran D0-V ile D1-D2 vision eğitimi dinamik yerel bütçeyi
kullanır. D3-D5 görevlerinde mission bütçesi, segment bütçelerinin
toplamı ile açık takeoff/transition/terminal rezervlerinden hesaplanır; ayrıca
konfigüre edilmiş bir hard cap bulunur.

## Gymnasium ve observation semantiği

- Başarı ve fiziksel failure `terminated=True` üretir.
- Zaman bütçesinin dolması `truncated=True` ve
  `termination_reason="time_limit"` üretir.
- Mission modunda ara waypoint başarısı ne `terminated` ne `truncated` üretir.
- `remaining_budget_fraction`, `segment_index`, `segment_count`, `L_plan`,
  başlangıç bütçesi ve kullanılan süre `info` içinde loglanır.
- Sonlu ufkun politika açısından gizli kalmaması için normalize kalan zaman,
  gerçek araçta da üretilebilen bir actor girdisi olarak `vision-v1`
  sözleşmesine eklenir. Hız/ivme limitleri zaman azaldığında da değişmez.
- PPO `n_steps` ve batch ayarları episode süresinden bağımsız tutulur; yalnız
  harita büyüdüğü için rollout uzunluğu doğrusal artırılmaz.

## Reward ve curriculum sonucu

- Yerel segmentte progress, açıklık, düzgünlük ve waypoint erişimi loglanır.
- Ara waypoint bonusu sınırlıdır ve nihai görev başarısı sayılmaz.
- Terminal segmentte hedef üzerinde güvenli yerleşme/hold reward'u ayrı
  bileşendir; hold dışındaki bekleme başarı üretmez.
- Timeout oranı, timeout anındaki rota ilerlemesi ve bütçe kullanım oranı ayrı
  metriklerdir.
- Curriculum geçişi için başarı ve çarpışmaya ek olarak timeout oranı,
  segment başarı oranı ve mission completion birlikte eşiklenir.

## Doğrulama kapısı

1. Aynı rota/config aynı `N_segment` değerini üretir.
2. Daha uzun geçerli rota monoton biçimde daha kısa bütçe almaz.
3. `T_min/T_max`, geçersiz ve uç rota tahminlerinde uygulanır.
4. Time limit yalnız `truncated=True` üretir; collision/success semantiğini
   değiştirmez.
5. Mission modunda waypoint geçişi fizik durumunu sıfırlamaz.
6. Terminal rezervi, ADR-012 hover hold süresini bütçeye ekler.
7. D0-V–D2 smoke eğitiminde dinamik bütçe ile sabit bütçe timeout/başarı
   ablation'ı raporlanır.
8. D3-D5 yalnız yerel politika kalite kapıları ve GPS-denied localization kapısı
   geçildikten sonra mission-level evaluation olarak açılır.

## Sonuç

Harita büyümesi episode süresinin kontrolsüz büyütülmesiyle çözülmeyecektir.
Yerel PPO eğitimi sınırlı segmentlerde kalacak; uzun menzil başarısı kesintisiz
mission evaluation ve segmentler arası durum sürekliliğiyle ölçülecektir.
