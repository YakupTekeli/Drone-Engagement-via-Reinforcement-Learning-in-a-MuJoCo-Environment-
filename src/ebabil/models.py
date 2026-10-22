"""MuJoCo model yükleme yardımcıları."""

from pathlib import Path

import mujoco

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_QUADROTOR_XML = PACKAGE_ROOT / "assets" / "mjcf" / "quadrotor.xml"


def load_quadrotor(path: str | Path = DEFAULT_QUADROTOR_XML) -> mujoco.MjModel:
    """Belirtilen quadrotor MJCF dosyasını derleyerek yükle."""

    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Quadrotor MJCF bulunamadı: {model_path}")
    return mujoco.MjModel.from_xml_path(str(model_path))


def load_quadrotor_timestep_s(path: str | Path = DEFAULT_QUADROTOR_XML) -> float:
    """MJCF'i derlemeden modelin açık fizik timestep değerini oku."""

    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Quadrotor MJCF bulunamadı: {model_path}")
    spec = mujoco.MjSpec.from_file(str(model_path))
    return float(spec.option.timestep)
