# ADR-005: Headless Eğitim ve Viewer Replay Ayrımı

**Durum:** Kabul edildi  
**Tarih:** 10 Eylül 2026

## Bağlam

MuJoCo viewer gerçek zamanlı görsel inceleme için yararlıdır fakat PPO eğitimi sırasında render maliyeti paralel ortam sayısını ve simülasyon hızını düşürür. Ayrıca otomatik testlerin pencere sistemine bağımlı olması CI ve agent çalışmasını kırılganlaştırır.

## Karar

- PPO eğitimi, smoke training, regression ve toplu deterministic evaluation varsayılan olarak headless çalışacaktır.
- Seçilmiş evaluation episode'ları state izi olarak kaydedilecek; viewer fizik simülasyonunu yeniden çözmek yerine bu izi oynatabilecektir.
- İlk kayıt sözleşmesi her fizik adımı için `time`, MuJoCo `qpos` ve `qvel` değerlerini içerir. Kontrol, rotor ve doygunluk izleri de aynı sonuç nesnesinde tutulur.
- Canlı viewer eğitim API'sinin içine gömülmeyecek; ayrı script/adapter katmanında kalacaktır.
- macOS viewer ana thread gereksinimi nedeniyle `mjpython` kullanılacaktır.

## Uygulama

- `run_hover_simulation` artık `qpos/qvel` dahil tam replay izi üretir.
- `scripts/view_hover_demo.py` önce manevrayı headless hesaplar, sonra kaydı gerçek zamanlı veya seçilen hızda oynatır.
- `scripts/view_static_scene.py` seed tabanlı engel sahnesi, drone ve hedef marker'ını gösterir.
- Proje dizinindeki boşluk, `.venv/bin/mjpython` shebang'ini doğrudan çalıştırmayı engellediği için `scripts/run_mjpython.sh` yorumlayıcıyı güvenli biçimde çağırır.

Viewer komutları:

```bash
scripts/run_mjpython.sh scripts/view_hover_demo.py
scripts/run_mjpython.sh scripts/view_static_scene.py --seed 20260910
```

## Sonuçlar

- Eğitim throughput'u rendering'den bağımsız kalır.
- Aynı değerlendirme izi yavaşlatılarak veya döngüye alınarak incelenebilir.
- Viewer doğrulaması grafik oturumu gerektirdiğinden headless otomatik test kapısının parçası değildir; CLI argümanları ve replay verisi otomatik test edilir.
- PPO aşamasında episode kayıt formatı metadata, reward bileşenleri ve termination nedeni ile genişletilecektir.

## Faz 6 güncellemesi

PPO kayıt formatı ADR-009 ile uygulandı. Replay, policy action/reward verisini
`20 Hz`; `qpos`, `qvel`, rotor thrust, kontrol komutları ve doygunluk izlerini
`500 Hz` zaman ekseninde saklar. `scripts/view_policy_replay.py` bu kaydı
yeniden simüle etmeden oynatır.
