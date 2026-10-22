# ADR-012: Güvenli Terminal Görev Kapsamı

**Durum:** Kabul edildi  
**Tarih:** 11 Eylül 2026  
**Güncelleme:** Terminal görev, hedef bölge üzerinde hover olarak kesinleştirildi.

## Bağlam

Hedef, dikdörtgen veya kare biçiminde tanımlanan bir alandır. Kamera hedef alanı
gördüğünde terminal davranışa geçilmesi istenmektedir. Otonom bir hava aracının
bir hedefe çarpma, zarar verme veya silah etkisi oluşturma amacıyla görsel
güdümlenmesi bu projenin güvenli araştırma kapsamı dışındadır.

## Karar

Aktif terminal görev, dikdörtgen/kare hedef bölgenin üzerinde güvenli ve kararlı
hover'dır. Kontrollü iniş ve temassız yanal gözlem olası gelecek görevler olarak
ertelenmiştir; mevcut environment başarı koşuluna dahil değildir.

Terminal durum makinesi şöyledir:

```text
transit -> acquired -> approach -> settle -> hold -> success
                 \-> target_lost / unsafe / stale_state -> abort
```

Desteklenen davranışlar:

- hedef alanı görsel olarak tespit etmek,
- güvenli yaklaşma koridoruna girmek,
- hedef alanın güvenli iç bölgesi üzerinde tanımlı irtifaya yerleşmek,
- düşük hız ve tutum hatasıyla kesintisiz hover hold tamamlamak,
- hedef kaybında hover/geri çekilme/abort davranışı uygulamak.

Desteklenmeyen davranışlar:

- hedefe çarpma veya çarpışmayı başarı sayma,
- kişi, araç, yapı veya başka bir nesneye saldırı/angajman,
- terminal hızlanma veya çarpışma noktasını optimize etme,
- patlayıcı, silah veya zarar verici payload entegrasyonu,
- insan denetimini kaldıran saldırı seçimi.

## Environment ve reward sonucu

- Hedefe veya zemine kontrolsüz çarpma her zaman failure/collision olacaktır.
- Başarı, aracın yatay izdüşümünün güvenlik payıyla küçültülmüş hedef dikdörtgeni
  içinde kalması ve hover koşullarını kesintisiz sağlamasıdır.
- Görsel hedef tespiti tek başına başarı değildir; confidence, süreklilik ve
  güvenli yaklaşma koşulları gerekir.
- Simülasyon başarı/failure kararı privileged ground truth ile hesaplanabilir;
  bu bilgi actor observation'ına verilmez. Görsel kestirim hatası ayrıca ölçülür.
- Terminal mod abort edilebilir olmalı; manuel override ve kill switch gerçek
  test öncesi zorunludur.
- Eğitim ve evaluation metrikleri terminal çarpışma hızı, minimum açıklık,
  target-loss, abort oranı, hover yerleşme süresi ve hold ihlallerini ayrıca
  raporlar.

İlk simülasyon varsayılanları config üzerinden verilecek ve fiziksel hedef/araç
ölçümü sonrası yeniden kalibre edilecektir:

| Koşul | İlk değer |
|---|---:|
| Hedef üstü hover irtifası | `2.0 m` |
| İrtifa hata toleransı | `0.20 m` |
| Maksimum doğrusal hız | `0.25 m/s` |
| Maksimum roll/pitch | `10 derece` |
| Maksimum açısal hız | `0.35 rad/s` |
| Kesintisiz hold süresi | `3.0 s` |

Yatay tolerans sabit bir merkez yarıçapı değildir. Hedef dikdörtgeni, drone
izdüşümü ve konum belirsizliğini içeren yapılandırılabilir safety margin ile
içe doğru küçültülür; araç merkezi bu güvenli iç bölgede kalmalıdır. İç bölge
boş kalıyorsa hedef o araç için geçersizdir.

Kamera montaj açısı hedefin yaklaşma ve hover sırasında görüşte kalmasına izin
vermelidir. Sabit ileri bakan kamera yer hedefinin tam üzerinde hedefi görüşten
çıkarabilir; bu nedenle IMX477 extrinsic/FOV doğrulaması, hedef üstü hover
environment'ının kabul kapısıdır. Kısa süreli görsel kayıp yalnız ölçülmüş VIO
belirsizliği ve sınırlı grace period ile tolere edilebilir.

Reward bileşenleri en az `target_progress`, `target_alignment`, `settle_speed`,
`hover_hold`, `smoothness`, `collision` ve `out_of_bounds` olarak ayrı
loglanacaktır. Hold reward'u sınırlı olacaktır; hedef dışında bekleyerek reward
biriktirmek başarı sayılmaz.

## Sonuç

Kamera tabanlı perception, GPS-denied localization, alan tespiti ve hedef bölge
üzerinde güvenli hover araştırması ana geliştirme yoludur. Zarara yönelik
terminal güdüm bu repository'de tasarlanmayacak veya uygulanmayacaktır. Uzun
görevde hover için ayrılan episode rezervi ADR-013'e göre hesaplanır.
