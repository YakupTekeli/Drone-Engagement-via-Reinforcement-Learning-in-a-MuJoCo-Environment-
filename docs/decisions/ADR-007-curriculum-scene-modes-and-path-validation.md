# ADR-007: Curriculum Sahne Modları ve Alternatif Rota Doğrulaması

**Durum:** Kabul edildi  
**Tarih:** 10 Eylül 2026  
**İlişki:** ADR-006 içindeki “garantili açık düz koridor” kararını genişletir ve engelli seviyeler için değiştirir.

## Bağlam

ADR-006 ile oluşturulan ilk statik sahneler başlangıç ile hedef arasındaki doğrudan koridoru açık tutuyordu. Bu yaklaşım sensör ve hedefe ulaşma smoke testleri için yararlı olsa da engelden kaçınmayı zorunlu kılmaz. Böyle bir dağılım tek başına PPO eğitiminde kullanılırsa politika rangefinder gözlemlerini yok sayarak düz uçmayı öğrenebilir.

## Karar

Sahne üreticisi üç açık mod sunacaktır:

| Mod | Engel | Eğitim amacı |
|---|---:|---|
| `clear` | 0 | Uçuş kontrolü ve engelsiz hedefe ulaşma |
| `single_blocker` | 1 | Doğrudan rotayı kapatarak ilk kaçınma davranışı |
| `random_obstacles` | 8 | Zorunlu blocker ve yedi seed'li rastgele engel |

`single_blocker` ve `random_obstacles` modlarında ilk engel başlangıç-hedef segmentinin orta noktasına yerleştirilir. Engel yüksekliği nominal uçuş irtifasını araç açıklığıyla birlikte kapatacak şekilde seçilir; politika yalnızca düz ilerleyerek hedefe ulaşamaz.

Her engelli sahne kabul edilmeden önce:

1. Başlangıç ve hedef açıklığı kontrol edilir.
2. Engeller arası minimum açıklık kontrol edilir.
3. Doğrudan 3B segmentin araç açıklığıyla genişletilmiş engel AABB'siyle kesiştiği doğrulanır.
4. Engel kutuları `0.25 m` araç açıklığı kadar genişletilir.
5. `0.10 m` çözünürlüklü XY occupancy grid üzerinde 8-komşulu rota aranır.
6. Her grid kenarı sürekli segment–AABB kesişimiyle doğrulanır; diyagonal hareketlerde dolu hücre köşelerinden geçişe ayrıca izin verilmez.

Rota bulunamayan rastgele yerleşim reddedilir ve aynı episode RNG akışıyla yeni sahne örneklenir.

Viewer varsayılanı `single_blocker` olacaktır. Diğer seviyeler `--mode` ile seçilebilir.

### Endpoint dağılımları — Faz 6 eki

Engel modundan bağımsız dört endpoint modu vardır:

| Mod | Başlangıç | Hedef |
|---|---|---|
| `fixed` | Sabit | Sabit |
| `random_start` | Seed ile örneklenir | Sabit |
| `random_goal` | Sabit | Seed ile örneklenir |
| `random_both` | Seed ile örneklenir | Seed ile örneklenir |

Rastgele endpoint'ler güvenli workspace içindeki yapılandırılmış 3B kutudan
örneklenir ve aralarında en az `2.0 m` bulunur. Endpoint örneklemesi scene RNG
akışının parçasıdır; aynı scene seed + scene mode + endpoint mode aynı sahneyi
üretir. Replay metadata endpoint modunu taşır ve viewer sahneyi aynı sözleşmeyle
yeniden kurar.

## Gerekçe

- Engelsiz ve engelli öğrenme hedeflerini ayrı dağılımlara bölmek curriculum ilerlemesini ölçülebilir yapar.
- Zorunlu blocker, kaçınma davranışının tesadüfi karşılaşmalara bağlı kalmasını önler.
- Alternatif rota kontrolü, politikanın çözülemeyen episode'larla cezalandırılmasını engeller.
- Aracı nokta olarak değil açıklık yarıçapıyla modellemek, geometrik yol bulunabilirliği fiziksel olarak daha anlamlı kılar.

## Doğrulama

- `single_blocker` başlangıçtaki body `+x` rangefinder ışınında `1.855 m` mesafede görülmektedir.
- Seed `0–24` için üretilen 25 `random_obstacles` sahnesinin tümünde doğrudan rota kapalı ve alternatif grid rotası açıktır.
- Aynı seed/mod aynı sahneyi, farklı seed farklı sahneyi üretir.
- Aynı seed/endpoint modu aynı başlangıç-hedef çiftini, farklı seed farklı çifti
  üretir; örnekler yapılandırılmış güvenli sınırlar ve minimum ayrım içindedir.

## Sonuçlar ve sınırlılıklar

- Alternatif grid rota doğrulaması 2B ve konservatiftir; engelin üzerinden uçuşu rota olarak saymaz. Doğrudan rota engelleme kontrolü ise 3B'dir.
- Şimdilik yalnızca axis-aligned kutular vardır.
- Rota varlığı kontrol edilir ancak rota PPO gözlemine veya ödülüne verilmez; politika kaçınmayı rangefinder ve ödülden öğrenmelidir.
- Daha zor geçit/koridor seviyeleri ve başarı tabanlı curriculum durum makinesi Faz 7 kapsamındadır.
