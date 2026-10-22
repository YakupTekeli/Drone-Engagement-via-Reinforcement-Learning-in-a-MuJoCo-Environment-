# Faz 7: D0-V Kamera Tabanlı Yerel Politika

**Tarih:** 11 Eylül 2026  
**Durum:** Tamamlandı; görev öğrenildi, iki yeni bulgu açıldı (REV-022, REV-023)

## Amaç ve kapsam

Projede **ilk kez** kamera + telemetri girdisiyle bir navigasyon politikası
eğitmek. Bu dilime kadar vision tarafındaki en uzun koşu `64` PPO adımıydı;
tüm önceki iş altyapı, sözleşme ve perception'dı.

D0-V, ADR-010 merdiveninin en alt basamağıdır: legacy `4-6 m` görev
geometrisinin `vision-v1` ile yeniden öğrenilmesi. **Engel yoktur**
(`scene_mode = "clear"`, engel sayısı `0`) ve sahne sabittir. Engelden kaçınma
curriculum'un bağımsız ekseni olup henüz başlamamıştır.

## Ölçülen görev

```text
scene_mode=clear, endpoint_mode=fixed  ->  engel sayısı 0
start (-2, 0, 1)  ->  hover anchor (2, 0, 2.01)   mesafe 4.0 m
episode bütçesi: 400 adım (20 s min clamp, ADR-013)
```

`clear` + `fixed` sahnesi seed'den bağımsız olarak aynıdır. Bu nedenle D0-V bir
**genelleme ölçümü değildir**; "kamera girdisiyle terminal hover öğrenilebiliyor
mu" sorusunu izole eder. Faz 6 legacy pilotunun `primary_scenario`'su da aynı
sahneydi.

Engel yakınlığı reward bileşeni bu koşuda ölüdür: `minimum_range_m` bütün
episode boyunca `5.0 m` (rangefinder menzil sonu), proximity eşiği `0.75 m`,
bileşen tam olarak `-0.0`.

## Eğitim altyapısı

Eğitim altyapısı legacy ortama sabitlenmişti; bu dilimde vision ortamına
bağlandı:

- `EnvironmentConfig.kind` (`legacy` | `vision`) ve `approval_manifest`
- Vision'da `sensor_mode` ve `max_episode_steps` yasak; episode sınırı ADR-013
  dinamik segment horizon'ından gelir
- Dict observation için `MultiInputPolicy`
- `VISION_MONITOR_INFO_KEYS`: hover durumu, abort sebebi, hold süresi episode
  kayıtlarına giriyor
- `require_legacy_environment()`: vision config legacy evaluation akışına
  sessizce düşmüyor, açık hata veriyor

## Throughput

| Ortam | Hız | 300k tahmini |
|---|---:|---:|
| legacy, 4 subproc | `625.8` ts/s | `8 dk` |
| vision, 1 dummy | `93.4` ts/s | `54 dk` |
| vision, 4 subproc | `216.5` ts/s | `23 dk` |
| vision + PPO optimize (gerçekleşen) | `135` ts/s | `37 dk` |

Vision ham hızı legacy'nin `0.35`'i (kamera render + encoder inference); PPO
optimize yükü `1.45x`.

## Eğitim sonucu

```bash
.venv/bin/python scripts/train_ppo.py \
  --config src/ebabil/configs/training/ppo_vision_d0v.toml \
  --device cpu --run-name d0v-seed20260911
```

`300434` timestep, `852` episode, `37.1 dk`, `135` ts/s. SB3 son raporu:
`success_rate 0.91`, `ep_rew_mean 101`, `explained_variance 0.838`.

| Timestep | Episode | Reward | Başarı | `hold_max` | Baskın termination |
|---|---:|---:|---|---:|---|
| `0-28k` | 106 | `-22.07` | `0/106` | `0.05` | `target_lost` 44, `out_of_bounds` 44 |
| `29-71k` | 106 | `+26.77` | `0/106` | `0.40` | `time_limit` 106 |
| `71-113k` | 106 | `+51.25` | `0/106` | `1.10` | `time_limit` 105 |
| `113-155k` | 106 | `+60.96` | `0/106` | `0.40` | `time_limit` 106 |
| `156-198k` | 106 | `+83.80` | `0/106` | `0.95` | `time_limit` 106 |
| `198-239k` | 106 | `+108.45` | `13/106` | `3.00` | `time_limit` 93, `hover_success` 13 |
| `239-273k` | 106 | `+118.85` | `67/106` | `3.00` | `hover_success` 67, `time_limit` 39 |
| `273-300k` | 110 | `+101.87` | `99/110` | `3.00` | `hover_success` 99, `time_limit` 11 |

Öğrenme sırası beklendiği gibi: hayatta kalma -> yaklaşma -> yerleşme -> hold
tamamlama. İlk başarı `~200k` civarında.

**Abort konusundaki endişe doğrulanmadı.** Koşu öncesinde `target_lost`
abort'unun toparlanmayı öğrenmeyi engelleyebileceği not edilmişti. Gözlem tersi:
`45` `target_lost` ve `44` `out_of_bounds` terminasyonunun **tamamı** ilk `28k`
adımda gerçekleşti; sonrasında pratikten kalktı. Politika hedefi görüşte tutmayı
öğrendi.

## Evaluation

```bash
.venv/bin/python scripts/evaluate_vision_policy.py <checkpoint> \
  runs/vision_audit_v2_20260911/approval_manifest.json <output> \
  --mode {in_distribution|encoder_ood} --episodes 20 --seed 310001 [--deterministic]
```

| Checkpoint | Kip | Policy | Başarı | Timeout | Çarpışma | Abort | Ort. return |
|---|---|---|---:|---:|---:|---:|---:|
| `best` | `in_distribution` | deterministik | **`20/20`** | `0` | `0` | `0` | `84.51` |
| `best` | `in_distribution` | stokastik | `11/20` | `9` | `0` | `0` | `117.15` |
| `final` | `in_distribution` | deterministik | `0/20` | `20` | `0` | `0` | `23.84` |
| `final` | `in_distribution` | stokastik | `20/20` | `0` | `0` | `0` | `94.83` |
| `best` | `encoder_ood` | deterministik | `6/20` | `13` | `0` | `0` | `19.90` |

`best` deterministik koşusu tam deterministiktir: `20/20` episode `193` adım,
`hold_seconds` `3.00`, yerleşme adımı `51`, ilk hold adımı `61`.

Bütün evaluation'larda **sıfır çarpışma ve sıfır abort**. Başarısızlıkların
tamamı timeout.

`encoder_ood` probu (değişken başlangıç, hedef aynı XY'de) `%30` başarı verdi.
Onaylı encoder'ın poz örneklemesi tek rota etrafında `+-1.25 m` yanal offset
kapsadığı için bu kip kısmen eğitim dağılımı dışındadır; bir kalite kapısı değil
tanısal ölçümdür.

## Reward bileşen dağılımı

`best` deterministik başarı episode'u, `193` adım, toplam `+84.51`:

| Bileşen | Toplam | Pay |
|---|---:|---:|
| `hover_hold` | `+41.20` | `48.8%` |
| `success` | `+30.00` | `35.5%` |
| `target_progress` | `+17.36` | `20.5%` |
| `time` | `-1.93` | `-2.3%` |
| `settle_speed` | `-1.83` | `-2.2%` |
| `target_alignment` | `-0.15` | `-0.2%` |
| `smoothness` | `-0.14` | `-0.2%` |
| `proximity`, `collision`, `out_of_bounds`, `abort` | `0.00` | `0.0%` |

Durum dağılımı: `transit` 23, `acquired` 3, `approach` 53, `settle` 10,
`hold` 103, `success` 1.

## Açılan bulgular

**REV-022 — hold reward üst sınırı iddiası yanlıştı.** ADR-017 hold toplamının
`0.40 * 60 = 24.0` değerini aşamayacağını söylüyordu. Gözlenen `41.20`, yani
`1.7` katı ve `success_bonus`'un `1.37` katı. Hold sayacı sıfırlanır fakat durum
yeniden `HOLD` olabilir ve verilmiş reward geri alınmaz; gerçek üst sınır
episode uzunluğudur (`400 * 0.40 = 160.0`). Kod ve ADR düzeltildi.

**REV-023 — uygulanabilir hold bandı `0.108 m`.** ADR-016'nın ölçtüğü nadir kör
bölgesi ile güvenli iç bölge sınırının kesişimi. İki checkpoint hedefin
arkasında neredeyse aynı yere yerleşiyor fakat başarı `3.6 cm` ile belirleniyor:
`best` `0.646 m` (bandın içinde), `final` `0.682 m` (dışında). Bu, deterministik
ve stokastik başarının ters yönde çıkmasını açıklıyor. `hover_height_m`
`2.0 -> 1.5` bandı `0.243 m`'ye çıkarır; görev geometrisi kararı olduğu için ADR
gerektirir.

## Bilinen sınırlılıklar

- **Engel yok.** Bu koşu engelden kaçınma kanıtı değildir; proximity bileşeni
  ölüdür. İlk gerçek kaçınma kanıtı `single_blocker` eğitimi olacaktır.
- **Genelleme yok.** `clear` + `fixed` tek deterministik sahnedir. Tek seed
  (`20260911`) ile koşuldu; ADR-009 en az üç seed ister.
- **Checkpoint seçimi başarıyı belirliyor** (REV-023). `final` deterministik
  kipte `0/20`. Tek kipte raporlanan başarı oranı bu görevde yanıltıcıdır.
- Onaylı encoder sabit geometride eğitilmiştir; `encoder_ood` `%30` bunun
  sınırını gösterir. Rastgele endpoint için REV-021 açıktır.
- `hover_hold`/`success_bonus` oranı ablation'ı yapılmadı (REV-022).
- Sentetik encoder ve privileged başarı doğrulaması gerçek uçuş güvenliği
  değildir.

## Artefact

```text
runs/d0v-seed20260911/
  final_model.zip, best/best_model.zip, vecnormalize.pkl
  checkpoints/, evaluation/evaluations.npz
  logs/{train_monitor,eval_monitor,progress.csv,events.out.tfevents.*}
  policy_evaluation/{best,best_stoch,final_in_distribution,final_stoch,best_ood}.json
```

## Sonraki adım

İki yol var ve ikisi bağımsız:

1. **REV-023 kararı:** `hover_height_m` düşürülüp D0-V aynı seed'le yeniden
   koşulur, deterministik/stokastik açıklığın kapanıp kapanmadığı ölçülür.
2. **Engel ekseni:** `single_blocker` ile ilk gerçek engelden kaçınma eğitimi.
   Proximity ve collision bileşenleri ilk kez canlı olur.

Ayrıca ADR-009 uyarınca D0-V'nin en az üç seed ile tekrarlanması gerekir.
