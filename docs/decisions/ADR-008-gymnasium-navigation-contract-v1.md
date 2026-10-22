# ADR-008: Gymnasium Navigasyon Sözleşmesi v1

**Durum:** Kabul edildi  
**Tarih:** 10 Eylül 2026

> **Donanım güncellemesi (11 Eylül 2026):** Observation v1 içindeki rangefinder
> kanalı gerçek araçta mevcut değildir. Bu sözleşme değiştirilmeden legacy
> baseline olarak korunur; kamera tabanlı görev yeni environment ve observation
> sürümüyle geliştirilecektir. Bkz. ADR-011.

## Bağlam

PPO eğitiminden önce Gymnasium ortamının action/observation sırası, zaman ölçeği, reward bileşenleri ve episode sonlandırma semantiği sabitlenmelidir. Bu alanların sessizce değişmesi checkpoint'leri geçersiz kılar ve deney karşılaştırmalarını bozar.

## Karar

İlk ortam `EbabilNavigation-v0`, sınıfı `EbabilNavigationEnv` ve observation sürümü `v1` olacaktır. Ortam yalnızca headless çalışır; görsel replay ADR-005 kapsamındaki ayrı süreçtir.

### Action space

```text
Box(low=-1, high=1, shape=(4,), dtype=float32)
```

| İndeks | Anlam | Fiziksel sınır |
|---:|---|---:|
| 0 | Yaw ile hizalı gövde/heading `vx` | `±1.5 m/s` |
| 1 | Yaw ile hizalı gövde/heading `vy` | `±1.5 m/s` |
| 2 | Dünya `vz` | `±1.0 m/s` |
| 3 | Yaw-rate | `±1.5 rad/s` |

Bu karar ADR-001'de açık bırakılan `vz_ref` çerçevesini dünya `z` ekseni olarak kesinleştirir. Aralık dışı action kırpılır ve `info["action_clipped"]` ile raporlanır; yanlış shape veya NaN/Inf reddedilir.

Hız komutu her fizik adımında position reference'a entegre edilir. Böylece klasik position/velocity kontrolcüsü RL politikasının hız hedefini izlerken drift'e karşı bir konum referansı da taşır.

### Observation space v1

Tüm observation `float32`, shape `(30,)` ve `[-1,1]` içinde kırpılmıştır.

| İndeks | Bileşen | Boyut | Normalizasyon |
|---|---|---:|---|
| `0:3` | Hedefin body-frame göreli konumu | 3 | `/ 6 m` |
| `3` | Hedef mesafesi | 1 | `/ 6 m` |
| `4:7` | Body-frame doğrusal hız | 3 | Action hız limitleri |
| `7:10` | Body-frame açısal hız | 3 | Controller rate limitleri |
| `10:13` | Body-frame yerçekimi yönü | 3 | Birim vektör |
| `13` | İrtifa | 1 | Workspace yüksekliğinden `[-1,1]` |
| `14:30` | Planar rangefinder | 16 | `2 * distance/max_range - 1` |

`sensor_mode="ideal"` başlangıç PPO baseline'ıdır. `sensor_mode="imperfect"` rangefinder noise, bias, sample, delay ve dropout zincirini kullanır. Observation v1 içindeki konum, orientation, doğrusal hız ve açısal hız MuJoCo'nun ideal sensörlerinden gelir; imperfect mod yalnızca rangefinder'ı değiştirir. Bu ideal state/IMU geçiş baseline'ıdır ve sim-to-real estimator doğrulaması değildir. Reward, termination ve tanılama için kullanılan diğer privileged değerler actor observation'ına eklenmez.

### Zaman ölçeği

- MuJoCo timestep ve alt seviye kontrol: `0.002 s / 500 Hz`
- PPO policy frequency: `20 Hz`
- Bir Gym `step`: `25` fizik/kontrol alt adımı
- Maksimum episode: `20 s / 400 policy step`

### Reward

Toplam reward aşağıdaki ayrı loglanan bileşenlerin toplamıdır:

```text
progress + goal + collision + failure
+ proximity + smoothness + attitude + time
```

- Progress, önceki ve güncel hedef mesafesi farkına dayanır.
- Goal bonus yalnızca hedef yarıçapı ve hız koşulu birlikte sağlanınca verilir.
- Collision ve diğer fiziksel failure cezaları ayrıdır.
- Proximity cezası ideal minimum menzili kullanır; sensör dropout'u güvenlik metriğini değiştirmez.
- Smoothness ardışık normalize action farkını cezalandırır.
- Her bileşen `info["reward_components"]` içinde ayrı raporlanır.

### Episode sonu

Hedef başarısı için araç `0.20 m` hedef yarıçapı ve `0.35 m/s` hız koşulunu
kesintisiz `0.50 s / 10 policy adımı` korumalıdır. Koşullardan biri bozulursa
bekleme sayacı sıfırlanır. Goal bonus yalnız bekleme tamamlandığında verilir.
Sayaç `info["goal_hold_steps"]`, eşik `info["goal_hold_required_steps"]` ve
süre `info["goal_hold_time_s"]` alanlarında raporlanır.

`terminated=True`:

- `goal`
- `obstacle_collision`
- `ground_collision`
- `out_of_bounds`
- `minimum_altitude`
- `unsafe_attitude`
- `invalid_state`

`truncated=True` yalnızca `time_limit` için kullanılır. Kesin neden `info["termination_reason"]`, başarı `info["is_success"]` ile raporlanır.

### Seed sözleşmesi

Gymnasium `reset(seed=...)` episode RNG'sini belirler. Sahne ve sensör için ayrı alt-seed'ler bu RNG'den türetilir; `options["scene_seed"]` ve `options["sensor_seed"]` ile ayrı ayrı sabitlenebilir. Kullanılan değerler `info` içinde raporlanır. Aynı episode/component seed'leri, scene mode, sensor mode ve action dizisi aynı rollout'u üretmelidir. Sabit scene seed ile farklı episode seed, aynı geometride farklı sensör bozulması üretir.

## Doğrulama

- Stable-Baselines3 `check_env` başarılıdır.
- Kayıtlı `EbabilNavigation-v0`, `gym.make` ile oluşturulabilir.
- Aynı seed ve action dizisi ideal ve bozulmuş sensör modlarında tekrarlanabilir.
- Basit hedef yönlendiricisi `clear` sahnede 118 adımda hedefe ulaşır; son 10
  uygun policy adımı kesintisiz hold koşulunu doğrular.
- Aynı yönlendirici `single_blocker` sahnede 28 adımda engelle çarpışır; bu, kaçınmanın gerekli olduğunu doğrular.
- `random_obstacles` modunda 2000 random policy step boyunca observation ve reward sonlu kalmıştır.

## Sonuçlar ve sınırlılıklar

- Observation v1 değişirse ortam ID'si veya observation version artırılmalıdır.
- Basit yönlendirici kaçınma baseline'ı değildir; yalnızca engelsiz kontrol ve blocker etkinliği testidir.
- Sahne her reset'te yeniden derlenir; PPO throughput gerekirse sabit model/mocap obstacle havuzuyla optimize edilecektir.
- İlk ortamda rüzgâr sıfırdır ve yalnızca rangefinder bozulabilir; domain randomization daha sonra eklenecektir.
