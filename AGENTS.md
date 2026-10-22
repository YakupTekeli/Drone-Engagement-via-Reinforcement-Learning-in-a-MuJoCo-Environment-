# Agent Çalışma Protokolü

Bu dosya, Ebabil MuJoCo projesinde çalışan tüm agentlar için ortak çalışma sözleşmesidir.

## 1. Başlamadan önce

Her agent sırasıyla şunları okumalıdır:

1. `docs/PROGRESS_LOG.md`
2. `docs/IMPLEMENTATION_PLAN.md`
3. İlgili teknik karar için `docs/decisions/`
4. Görev tasarımı değişecekse `docs/PROJECT_DESIGN.md`
5. Araştırma iddiası kullanılacaksa `docs/LITERATURE_REVIEW.md`

## 2. Çalışma ortamı

- Yalnızca proje kökündeki `.venv` kullanılmalıdır.
- Komutlarda belirsiz sistem Python'u yerine `.venv/bin/python` tercih edilmelidir.
- Bağımlılık eklendiğinde hem `pyproject.toml` hem `requirements.lock` güncellenmelidir.
- `.venv`, checkpoint, video ve eğitim çıktıları Git'e eklenmemelidir.
- Makineye global paket kurulumu ve `sudo pip` kullanılmamalıdır.

Kanonik doğrulama komutları:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
.venv/bin/ruff format --check src scripts tests
.venv/bin/mypy
```

`unittest` testleri korunur ve pytest tarafından keşfedilir. Yeni testler mevcut
test stiliyle uyumlu yazılmalıdır.

## 3. Kaynak doğruluk sırası

Çelişki olduğunda aşağıdaki sıra izlenir:

1. Otomatik testlerle doğrulanan mevcut kod davranışı
2. `docs/decisions/` içindeki kabul edilmiş ADR'ler
3. `docs/PROJECT_DESIGN.md`
4. `docs/IMPLEMENTATION_PLAN.md`
5. `docs/LITERATURE_REVIEW.md`

Kod ile belge çelişiyorsa agent bunu sessizce kabul etmemeli; ya kodu karara uygun düzeltmeli ya da karar değişikliğini yeni ADR ile belgelemelidir.

## 4. Değişiklik sınırları

- Kullanıcının veya başka bir agentın ilgisiz değişiklikleri korunmalıdır.
- Aynı dosyada paralel çalışma gerekiyorsa görevler mümkün olduğunca farklı modül veya dosyalara bölünmelidir.
- Koordinat sistemi, observation sırası, action anlamı, reward semantiği veya kontrol frekansı sessizce değiştirilmemelidir.
- Bu sözleşmeleri değiştiren her karar için ADR oluşturulmalı veya mevcut ADR güncellenmelidir.
- Fizik sabitleri kod içinde farklı yerlerde tekrarlanmamalıdır; tek kaynaklı konfigürasyon hedeflenmelidir.

## 5. Her iş diliminin zorunlu kaydı

Her anlamlı iş dilimi sonunda `docs/PROGRESS_LOG.md` içine şu bilgiler eklenmelidir:

- Tarih ve iş dilimi başlığı
- Amaç/kapsam
- Oluşturulan veya değiştirilen dosyalar
- Alınan teknik kararlar ve gerekçeleri
- Çalıştırılan komutlar/testler ve sonuçları
- Bilinen sınırlılıklar veya riskler
- Sonraki önerilen adım

Sadece “tamamlandı” yazmak yeterli değildir. Bir sonraki agentın çalışmaya ek açıklama istemeden devam edebilmesi hedeflenmelidir.

## 6. Test ve doğrulama kuralları

- Fizik ve frame değişiklikleri analitik veya sayısal birim testleri olmadan tamamlanmış sayılmaz.
- RL değişiklikleri kısa smoke training ve ayrı deterministic evaluation olmadan tamamlanmış sayılmaz.
- Reward değişikliklerinde tüm bileşenler ayrı loglanmalıdır.
- Regression toleransları gerekçeli olmalıdır; floating-point fizik için tam eşitlik kullanılmamalıdır.
- Render gerektirmeyen testler headless çalışmalıdır.
- Eğitim ve toplu evaluation headless çalışmalıdır; viewer yalnızca seçilmiş evaluation kayıtlarını incelemek içindir.
- macOS'ta grafik bağlamı gerektiren testler normal sandbox dışında veya uygun viewer süreciyle yürütülebilir; bunun sonucu ilerleme günlüğüne yazılmalıdır.
- Bu projenin boşluk içeren yolu `.venv/bin/mjpython` shebang'ini bozduğu için viewer `scripts/run_mjpython.sh <viewer-script>` ile başlatılmalıdır.

## 7. Koordinat ve isimlendirme sözleşmesi

- Dünya: sağ elli, `+z` yukarı.
- Gövde: FLU, `+x` ileri, `+y` sol, `+z` yukarı.
- Rotor sırası: `front_left`, `front_right`, `rear_right`, `rear_left`.
- Eylem dizilerinin sırası kod, belge ve testlerde açıkça sabitlenmelidir.
- SI birimleri kullanılır: metre, saniye, kilogram, Newton, Newton-metre, radyan.

## 8. Güvenlik

- Simülasyondaki başarı gerçek uçuş güvenliği olarak ifade edilmemelidir.
- Pervaneli gerçek test, SITL/HITL, failsafe, manuel override ve fiziksel test prosedürü tamamlanmadan önerilmemelidir.
- Collision penalty bir güvenlik garantisi değildir.

## 9. Güncel çalışma hedefi

Güncel faz ve sıradaki işler için yalnızca `docs/PROGRESS_LOG.md` içindeki “Güncel durum” bölümünü esas alın. Tamamlanan işten sonra bu bölüm güncellenmelidir.
