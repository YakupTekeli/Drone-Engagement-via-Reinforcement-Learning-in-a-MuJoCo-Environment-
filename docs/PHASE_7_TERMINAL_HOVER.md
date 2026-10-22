# Faz 7: Terminal Hover Durum Makinesi ve Reward

**Tarih:** 11 Eylül 2026  
**Durum:** Tamamlandı; durum makinesi, reward bileşenleri, ortam entegrasyonu ve
gerçek renderer PPO smoke kapıları geçti

## Amaç ve kapsam

Bu iş dilimi ADR-012 terminal görevini `EbabilVisionNavigation-v0` ortamına
uygular: hedef bölgenin güvenli iç alanı üzerinde kesintisiz hover, görsel
süreklilik kapıları, abort semantiği ve ayrı loglanan reward bileşenleri.

Kapsam dışı: D0-V–D2 curriculum eğitimi, mission segment zinciri, deploy
edilebilir görsel faz kestirimi, gerçek Pixhawk/VIO adapter'ı.

## Sözleşme

Durum makinesi `src/ebabil/missions/hover_state.py` içindedir. `vision-v1`
observation sözleşmesi **değişmemiştir**; terminal durum yalnız `info` içinde
raporlanır.

```text
transit -> acquired -> approach -> settle -> hold -> success
   \-- stale_frame (her durumdan) / target_lost (tespit sonrası) --> abort
```

| Parametre | Değer | Adım (`20 Hz`) |
|---|---:|---:|
| `acquire_score` | `0.60` | — |
| `loss_score` | `0.40` | — |
| `acquire_seconds` | `0.30 s` | `6` |
| `grace_seconds` | `0.50 s` | `10` |
| `approach_radius_m` | `2.50 m` | — |
| `hold_seconds` (hedef config) | `3.0 s` | `60` |

Hover koşulları privileged ground truth'tan hesaplanır ve birlikte sağlanmalıdır:
güvenli iç bölge (`+-0.65 m`), irtifa hatası `<= 0.20 m`, hız `<= 0.25 m/s`,
tilt `<= 10 derece`, açısal hız `<= 0.35 rad/s`.

### Reward bileşenleri

| Bileşen | Ağırlık | Koşul |
|---|---:|---|
| `target_progress` | `5.0` | Hover anchor'a 3B yaklaşma |
| `target_alignment` | `0.20` | Yalnız görünür hedef + `acquired`/`approach` |
| `settle_speed` | `0.50` | Yalnız güvenli iç bölgede |
| `hover_hold` | `0.40` | Yalnız `HOLD`; toplam `<= 24.0` |
| `proximity` | `0.05` | Privileged minimum açıklık |
| `smoothness` | `0.02` | Action farkı karesi |
| `collision` | `25.0` | Çarpışma |
| `out_of_bounds` | `15.0` | Workspace ihlali |
| `success` | `30.0` | Hover başarısı |
| `abort` | `10.0` | Abort ve diğer fiziksel failure |
| `time` | `0.01` | Adım başına |

## Oluşturulan/değiştirilen dosyalar

- `src/ebabil/missions/{__init__,hover_state}.py` (yeni paket)
- `src/ebabil/rewards/terminal_hover.py` (yeni)
- `src/ebabil/rewards/__init__.py`
- `src/ebabil/envs/navigation.py` — `TerminalDecision`, `StepReward`,
  `_terminal_decision`, `_compute_step_reward` override noktaları
- `src/ebabil/envs/vision.py` — durum makinesi, hover reading, terminal reward
- `src/ebabil/envs/vision_config.py` — `terminal_hover` ve
  `terminal_hover_reward` bölümleri
- `src/ebabil/configs/environment/vision_navigation.toml`
- `tests/test_terminal_hover.py` (yeni, 40 test + 28 subtest)
- `scripts/smoke_vision_navigation_env.py` — hover durumu ve bileşen raporlaması
- `docs/decisions/ADR-017-terminal-hover-state-machine-and-reward.md`

## Doğrulama

### Uygulama sırasında yakalanan ölçek hatası

İlk uygulama `_vision_aux[0]` ham sigmoid skorunu observation ölçeğiyle
karıştırdı ve `target_score = (raw + 1) / 2` hesapladı. Bu, `[0, 1]` aralığını
`[0.5, 1]`'e sıkıştırıyordu:

- `loss_score = 0.40` hiçbir zaman tetiklenemiyordu, yani hedef kaybı
  algılanamıyordu;
- tespit eşiği efektif olarak ham `0.20` skoruna düşüyordu, yani tespit
  neredeyse otomatikti.

Hata gerçek renderer smoke'unda görüldü: random policy `100` adımda `91` adım
`acquired` durumundaydı. Düzeltme sonrası aynı koşuda dağılım
`transit 36 / acquired 62 / approach 2` oldu.

İlk yazılan testler hatayı yakalamıyordu; eğitilmemiş test encoder'ı her görüntü
için `~0.505` skor ürettiği ve bu değer iki eşik arasında kaldığı için görsel
kapılar hiç sınanmıyordu. Bu nedenle `ScriptedVisionEncoder` eklendi ve skoru
test tarafından sürülebilir hâle getirildi. Üç yeni test hatayı doğrudan
yakalıyor; hata geçici olarak geri konularak kanıtlandı:

```text
hatalı kod:     3 failed, 37 passed
düzeltilmiş:   40 passed, 28 subtests passed
```

### Legacy davranışın korunması

`EbabilNavigationEnv.step` iki override noktasına ayrıldı. Refactor'un davranış
değiştirmediği, refactor öncesi alınan uçtan uca rollout parmak iziyle
kanıtlandı (3 sahne modu x 2 sensör modu x 2 seed x 60 adım; observation,
reward, terminated/truncated, termination_reason, reward bileşenleri ve hold
sayaçları dahil):

```text
LEGACY_ROLLOUT_SHA256 = 03f16232adae103e370fe2a5cbcb2d0c62900838f90b51fc7db408e14e3f5e23
refactor öncesi: aynı    refactor sonrası: aynı
LEGACY_SCENE_SHA256 = 75274526eae51f52475ea4bf7ae0d0f44f12d7c2f4cc3f6f84418a93c390dfe6  (değişmedi)
```

### Kalite kapısı

```text
pytest        : 196 passed, 46 subtests passed in 39.30s  (öncesi 156/18)
ruff check    : All checks passed
ruff format   : 90 files already formatted
mypy          : 49 source file temiz
```

### Gerçek renderer smoke

```bash
.venv/bin/python scripts/smoke_vision_navigation_env.py \
  runs/vision_audit_v2_20260911/approval_manifest.json \
  --mode random_obstacles --seed 20260911 \
  --random-steps 100 --ppo-timesteps 64 --evaluation-steps 80 \
  --output-dir runs/vision_hover_smoke_v1_20260911
```

- `check_env` + `100` random step: `1.870 s`;
  hover durum dağılımı `transit 36 / acquired 62 / approach 2`
- `64` PPO step + `80` deterministic evaluation: `2.332 s`
- Toplam: `4.202 s`
- Deterministic evaluation `80` adım cap'ine geldi, hedefe yerleşmedi. Bileşen
  toplamları ayrı loglandı: `target_alignment -0.100`, `target_progress -0.037`,
  `time -0.800`, `hover_hold 0.0`. `64`-step smoke model kalite iddiası
  taşımadığından bu beklenen sonuçtur.
- Artefact: `runs/vision_hover_smoke_v1_20260911/{summary.json,smoke_model.zip}`

Kesintisiz hold'un başarıya gittiği ayrıca ortam entegrasyon testiyle
doğrulanmıştır: araç hover anchor'a yerleştirildiğinde `transit` `5` adım,
`hold` `60` adım, ardından `hover_success` ve `hover_hold_seconds = 3.0`.

## Bilinen sınırlılıklar ve riskler

- Terminal durum actor observation'ında yoktur. Politika fazı `vision_aux`,
  `fc_state` ve `waypoint` üzerinden çıkarmak zorundadır. Bu bilinçli bir
  karardır (privileged localization sızıntısını engeller) fakat öğrenmeyi
  zorlaştırabilir; D0-V curriculum'unda ölçülecektir.
- Reward ağırlıkları ölçülmüş değil başlangıç hipotezidir. `hover_hold` toplam
  üst sınırı (`24.0`) ile `success_bonus` (`30.0`) arasındaki oran ablation
  gerektirir.
- `grace_seconds = 0.50 s` gerçek VIO drift ölçümüne değil ADR-012 ilkesine
  dayanır.
- Kesintisiz hold başarısı henüz yalnız araç hover anchor'a elle yerleştirilerek
  gösterilmiştir; öğrenilmiş politikayla başarı D0-V curriculum kapısıdır.
- Hover koşulları policy adımı çözünürlüğünde (`50 ms`) ölçülür; bu pencere
  içindeki geçici ihlaller görünmez.
- Privileged başarı doğrulaması ve sentetik encoder gerçek uçuş güvenliği
  değildir.

## Sonraki adım

D0-V yerel vision curriculum iş paketi: aynı `4-6 m` görev geometrisini kamera
girdisiyle yeniden eğitmek, hover başarı oranını ve abort/timeout dağılımını
ölçmek. REV-018 (dataset tek rota geometrisi) bu paketin ön koşuludur.
