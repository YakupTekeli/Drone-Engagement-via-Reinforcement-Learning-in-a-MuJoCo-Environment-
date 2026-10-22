# Demo medyası

README'de gösterilen örnek video ve kaynağı. Dosyalar depoya bilinçli olarak
eklenmiştir (`.gitignore` yalnız `runs/` ve `videos/` altındaki çıktıları
dışarıda tutar).

| Dosya | İçerik |
|---|---|
| `demo_hover_bright.mp4` | 1760 x 480, 20 fps, 20.5 s, H.264 (1.9 MB). `runs/demo_20260930/bright_seed12_ep_seed310004.mp4` dosyasının kopyası. |
| `demo_hover_bright.gif` | Aynı videonun README için küçültülmüş hâli: 640 px genişlik, 8 fps, 96 renk (3.7 MB). |

**Üç panel:** solda drone kamerası ve YOLO11n kutuları (kırmızı engel, yeşil
hedef) ile durum metni; ortada dış takip kamerası; sağda üstten harita
(kırmızı engeller, yeşil hedef, sarı x inanılan hedef, mavi rota).

**Kaynak koşu:** politika `runs/detector_cone_d1a_seed20260912_20260929/final_model.zip`
(seed `20260912`, 600k adım, ADR-024 koni kısıtı), dedektör
`runs/detector_audit_v1_20260929/approval_manifest.json`; D1a sahnesi (5
engel, 20 x 20 m), episode seed `310004`, `development_ood` (parlak, sıcak
ışık) görsel alanı, terminal yardımcı açık. Episode 389. adımda
`hover_success` ile bitti (hedefe yatay mesafe `0.29 m`).

**Yeniden üretme komutu** (30 Eylül 2026 koşusuyla aynı):

```bash
.venv/bin/python scripts/render_demo_video.py \
  runs/detector_cone_d1a_seed20260912_20260929/final_model.zip \
  runs/detector_audit_v1_20260929/approval_manifest.json \
  docs/media/demo_hover_bright.mp4 \
  --episode-seed 310004 \
  --scene-config src/ebabil/configs/environment/static_scene_d1a.toml \
  --visual-domain-split development_ood
ffmpeg -i docs/media/demo_hover_bright.mp4 \
  -vf "fps=8,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
  docs/media/demo_hover_bright.gif
```

Simülasyon kaydıdır; gerçek uçuş güvenliği iddiası taşımaz.
