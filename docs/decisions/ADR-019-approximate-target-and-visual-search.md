# ADR-019: Yaklaşık Hedef Konumu ve Görsel Arama

**Durum:** Kabul edildi  
**Tarih:** 12 Eylül 2026  
**İlişki:** ADR-015 `waypoint` kanalının **kaynağını** değiştirir (shape ve
semantik korunur). ADR-017 durum makinesine `search` durumu ekler. ADR-018
merdiveni bir eksen daha kazanır. ADR-011 GPS-denied kapsamını somutlaştırır.

## Bağlam

`vision-v1` içindeki `waypoint` kanalı, hedefin gerçek dünya konumundan ve
aracın gerçek konumundan **sıfır hatayla** hesaplanıyordu:

```python
waypoint_world = scene.goal_world_m - reading.ideal.position_world_m
```

Bu bir oracle'dır ve GPS'in sağladığı şeyin aynısıdır. ADR-015 bunu
"ideal-estimator baseline" olarak dürüstçe etiketlemişti fakat sonucu şudur:
**D0-V'de navigasyon problemi kamera tarafından çözülmedi.** Ölçülen ablation
(`best` checkpoint, 6 episode, deterministik):

| Ablation | Başarı | Terminasyon |
|---|---:|---|
| baseline (oracle) | `6/6` | `hover_success` |
| `waypoint = 0` | `0/6` | `out_of_bounds` |
| `vision_latent = 0` | `0/6` | `time_limit` |

İki kanal da taşıyıcıdır: waypoint kaba navigasyonu, görüntü terminal hover'ı
yapar. Ancak waypoint kusursuz olduğu sürece kamera **yer bulma** işine hiç
katılmaz.

## Karar

### 1. Araç kendi konumunu bilir, hedefin yalnız yaklaşık konumunu bilir

- **Kendi konum:** hatasız kabul edilir. Görsel pozisyonlama (VPS/VIO) tarafından
  sağlandığı varsayılır ve bu katman bu projede modellenmez. Bu bilinçli bir
  kapsam kararıdır, bir doğruluk iddiası değildir.
- **Hedef konumu:** episode başına örneklenmiş bir hata ile bilinir. Araç
  hedefin **gerçek konumunu hiçbir zaman** öğrenmez.

```text
believed_target = true_target + error,   ||error_xy|| <= R
waypoint = f(believed_target - true_position)
```

Hata episode başına sabittir (birikmez) ve scene seed'inden deterministik
üretilir. Bu, VIO drift modellemesi gerektirmediği için tekrar üretilebilir ve
ölçülebilir kalır.

`waypoint` kanalının shape'i (`(4,)`), birimleri ve ölçeklemesi
**değişmez**; yalnız türetildiği hedef konumu değişir. Bu nedenle `vision-v1`
observation sözleşmesi kırılmaz.

### 2. R'nin anlamlı bandı ölçülmüştür

**Alt sınır `0.65 m`** — hedefin güvenli iç bölge yarı-boyutu. `R < 0.65 m`
ise araç doğrudan inanılan konuma gider ve görüntü kullanmadan hover başarısına
ulaşır; değişiklik anlamsız olur.

**Üst sınır `~40 m`** — tek tırmanma ve `360 derece` yaw taramasıyla
bulunabilecek en büyük hata (aşağıda türetilir).

### 3. Arama davranışı zorunludur ve geometrisi ölçülmüştür

Kamera ayak izi `pitch=45 derece`, `vfov=59.84 derece` ile bir halka
sektörüdür. İrtifa `h` için yakın sınır `0.27h`, uzak sınır `3.71h`.

| Arama irtifası | Kapsama `R` | Uzak sınırda hedef | Karar |
|---:|---:|---:|---|
| `2.01 m` | `7.5 m` | `53 px` | güvenilir |
| `4 m` | `14.8 m` | `27.1 px` | güvenilir |
| `8 m` | `29.7 m` | `13.6 px` | güvenilir |
| `10.9 m` | `40.3 m` | `10.0 px` | **optimum** |
| `15 m` | `55.7 m` | `7.2 px` | sınırda |
| `20 m` | `74.2 m` | `5.4 px` | yetersiz |

`10.9 m`'den sonra hedef, kapsama alanı büyüdüğünden daha hızlı küçülür.
Hover irtifasında (`2.01 m`) tek poz `5 m` yarıçaplı diskin yalnız `%46`'sını
görür; `360 derece` yaw taraması `R <= 7.5 m` kapsar.

Davranış sözleşmesi: **tırmanarak ara, alçalarak yerleş.** Gereken arama
irtifası `h_search >= R / 3.71`, hover irtifası alt sınırdır.

### 4. Durum makinesine `search` eklenir

```text
transit -> search -> acquired -> approach -> settle -> hold -> success
   \          \          \          \          \         \
    \--------- stale_frame / target_lost / search_timeout -> abort
```

`search` durumu, araç inanılan hedef konumunun yakınına ulaştığı fakat hedefi
henüz görsel olarak tespit etmediği durumdur. Beklenen davranış: gerekiyorsa
arama irtifasına tırmanmak ve yaw taraması yapmak.

- `transit -> search`: inanılan konuma `search_entry_radius_m` kadar yaklaşıldı
- `search -> acquired`: `acquire_score` eşiği `acquire_seconds` boyunca sağlandı
- `search -> abort` (`search_timeout`): arama bütçesi doldu

`search` durumu ADR-017'nin iki katı kısıtını korur: `SUCCESS` yalnız `HOLD`'dan
girilir, terminal durumlar soğurucudur.

### 5. R merdiveni

| Seviye | Rota | `R` | Gereken arama irtifası | Belirsizlik kanalı |
|---|---:|---:|---:|---|
| D1 | `5-15 m` | `2 m` sabit | `2.0 m` (hover irtifası yeter) | gerekmez |
| D2 | `15-50 m` | `5 m`, episode'lar arası değişken | `2.0 m` | **gerekir** |
| D3 | `50-150 m` | `15 m` değişken | `4.0 m` | gerekir |
| D4 | `150-500 m` | `30 m` değişken | `8.1 m` | gerekir |
| D5 | `0.5-2 km` | `30 m` değişken | `8.1 m` | gerekir |

D1'de `R` sabit olduğu için politika onu örtük öğrenir ve observation değişmez.
D2'den itibaren `R` episode'lar arası değişir; politika waypoint'e ne kadar
güveneceğini bilmek zorundadır. Bu nedenle normalize `waypoint_uncertainty`
skaleri **`vision-v2`** ile eklenir. Gerçek sistemde bu değer görev brifingi
hassasiyetinden bilinir, yani deploy edilebilir.

### 6. Encoder kapsama gereksinimi

Mevcut `vision-dataset-v4c` poz örneklemesi rota etrafında `+-1.25 m` yanal
offset ve `0.80-2.10 m` irtifa kullanır. Yeni senaryoda araç hedefe `2-30 m`
yanal mesafede ve D3'te `4 m`, D4'te `8.1 m` irtifada arama yapacaktır.

- **D1 ve D2 için sorun yoktur:** `R <= 5 m` ve arama irtifası `2.0 m`, mevcut
  dağılıma yakındır.
- **D3'ten itibaren dataset yenilemesi zorunludur:** genişletilmiş yanal offset,
  `10 m`'ye kadar irtifa ve `search_pose_fraction` ile açık bir arama pozu
  ailesi (REV-019'da hold pozu için yapılanın aynısı).

Bu gereksinim REV-021 (risk head, değişken endpoint) ile aynı dataset
yenilemesinde karşılanabilir.

## Doğrulama kapısı

1. `R < 0.65 m` konfigürasyonu reddedilir (değişikliği anlamsız kılar).
2. Aynı scene seed aynı hedef hatasını üretir; hata episode içinde değişmez.
3. Actor observation gerçek hedef konumunu hiçbir kanalda taşımaz; `waypoint`
   yalnız inanılan konumdan türetilir.
4. Başarı doğrulaması **gerçek** hedef konumuna göre yapılır (privileged);
   inanılan konum başarı kararına girmez.
5. `search` durum geçişleri, arama zaman aşımı ve abort testleri.
6. Arama geometrisi: `h_search >= R / 3.71` invariantı ve kapsama testi.
7. Waypoint-tek-başına başarı ablation'ı: `R >= 0.65 m` iken görüntü kanalları
   sıfırlandığında başarı düşmelidir. Aksi halde görev yine oracle'a dayanıyordur.
8. Held-out kapısı ADR-018 seviye eşikleriyle, deterministik ve stokastik ayrı.

## Uygulama notları (12 Eylül 2026)

- İnanılan hedef `believed_target_xy(true_goal, R, scene_seed, safe_half)` ile
  üretilir; disk üzerinde düzgün dağılım için yarıçapın karekökü alınır. Hata
  scene seed'inden deterministiktir ve episode içinde sabittir.
- `waypoint` kanalı `_observation` içinde yalnız inanılan hedeften türetilir;
  `goal_world_m` artık bu fonksiyonda hiç kullanılmaz.
- `search` durumu `believed_horizontal_distance_m <= search_entry_radius_m`
  koşuluyla girilir ve `search_timeout_seconds` aşılınca
  `hover_abort_search_timeout` üretir.
- **Privileged reward şekillendirmesi.** `target_progress` ve hover koşulları
  **gerçek** hedefe göre hesaplanır. Bu ADR-012'nin izin verdiği privileged
  reward'dır ve deploy'da reward bulunmadığı için sızıntı değildir; fakat
  politikanın görüntüye dayanmasını teşvik eden bir eğitim sinyalidir ve
  raporlanırken bu belirtilmelidir.
- `search_entry_radius_m = 4.0 m` D0-V'nin `4 m` sahnesinde episode başından
  itibaren `search` durumu üretir. D1 rotaları `5-15 m` olduğu için bu
  dejenerasyon D1'de yoktur.

## Sonuçlar ve sınırlar

- **Kendi konum hatası modellenmemektedir.** Bu bilinçli kapsam kararıdır.
  Gerçek sistemde VPS/VIO hatası hedef konum hatasına eklenir ve toplam bütçe
  `R + drift` olur. Menzil sınırı `tespit_yarıçapı / drift_oranı` formülüyle
  ayrıca değerlendirilmelidir (REV-024).
- Görsel tespit menzili hedef boyutuna bağlıdır. `2 x 2 m` hedef için güvenilir
  tespit `~42 m`, sınırda `~104 m`, `200 m` üzerinde imkansızdır. Daha büyük
  görev mesafesi daha büyük hedef veya daha uzun odak uzaklığı gerektirir.
- Arama geometrisi noktasal kamera ve düz zemin varsayar; eğim, engel
  occlusion'ı ve rüzgâr hesapta değildir.
- Bu ADR gerçek uçuş güvenliği iddiası taşımaz.
