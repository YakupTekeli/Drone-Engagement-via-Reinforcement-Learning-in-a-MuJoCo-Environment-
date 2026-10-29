"""Eğitilmiş politikayı simülatör dışına (ör. Unreal Engine + ROS) taşıma çekirdeği.

Bu paket, PPO politikasının Unreal Engine gibi harici bir simülasyonda
çalıştırılması için çatı-bağımsız (ROS'tan bağımsız) bir köprü sağlar.
Gözlem kurulumu ve hareket kısıtları `ebabil` içindeki eğitim sözleşmesinin
AYNI fonksiyonlarını kullanır; böylece politika eğitimdekiyle aynı girdiyi
görür ve davranış sürüklenmesi olmaz.

Kapsam (kullanıcı kararı, 29 Ekim 2026): drone ağaçlardan (engel) kaçınarak
tanka (hedef) yönelir ve güvenli bir **temassız** yakın mesafede durur. Hedefe
fiziksel temas/çarpma bu köprünün kapsamı dışındadır; `DeploymentConfig`
pozitif bir `standoff_distance_m` tabanı uygular.
"""

from ebabil.deploy.policy_runner import (
    BodyVelocityCommand,
    DeploymentConfig,
    DronePose,
    PolicyRunner,
    camera_geometry_from_intrinsics,
    load_policy_predict_fn,
    quaternion_wxyz_to_matrix,
)

__all__ = [
    "BodyVelocityCommand",
    "DeploymentConfig",
    "DronePose",
    "PolicyRunner",
    "camera_geometry_from_intrinsics",
    "load_policy_predict_fn",
    "quaternion_wxyz_to_matrix",
]
