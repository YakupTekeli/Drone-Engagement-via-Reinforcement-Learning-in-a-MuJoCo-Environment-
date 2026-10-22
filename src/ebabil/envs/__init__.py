"""Gymnasium ortamları ve kayıt yardımcıları."""

from gymnasium.envs.registration import register, registry

from ebabil.envs.config import (
    DEFAULT_NAVIGATION_CONFIG,
    NavigationEnvConfig,
    load_navigation_config,
)
from ebabil.envs.detection_vision import (
    DEFAULT_DETECTION_VISION_CONFIG,
    DETECTION_VISION_ENVIRONMENT_ID,
    DETECTION_VISION_OBSERVATION_KEYS,
    DETECTION_VISION_OBSERVATION_VERSION,
    EbabilDetectionVisionNavigationEnv,
)
from ebabil.envs.latent_vision import (
    DEFAULT_LATENT_VISION_CONFIG,
    LATENT_VISION_ENVIRONMENT_ID,
    LATENT_VISION_OBSERVATION_KEYS,
    LATENT_VISION_OBSERVATION_VERSION,
    EbabilLatentVisionNavigationEnv,
)
from ebabil.envs.navigation import (
    OBSERVATION_COMPONENTS,
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    EbabilNavigationEnv,
    NavigationPhysicsTrace,
)
from ebabil.envs.vision import (
    VISION_ACTION_VERSION,
    VISION_ENVIRONMENT_ID,
    VISION_OBSERVATION_KEYS,
    VISION_OBSERVATION_VERSION,
    EbabilVisionNavigationEnv,
    SegmentHorizon,
    SimulationDeadlineScheduler,
    compute_segment_horizon,
)
from ebabil.envs.vision_config import (
    DEFAULT_VISION_NAVIGATION_CONFIG,
    SegmentBudgetConfig,
    VisionNavigationEnvConfig,
    VisionObservationConfig,
    load_vision_navigation_config,
)

ENVIRONMENT_ID = "EbabilNavigation-v0"


def register_environments() -> None:
    if ENVIRONMENT_ID not in registry:
        register(
            id=ENVIRONMENT_ID,
            entry_point="ebabil.envs.navigation:EbabilNavigationEnv",
        )
    if VISION_ENVIRONMENT_ID not in registry:
        register(
            id=VISION_ENVIRONMENT_ID,
            entry_point="ebabil.envs.vision:EbabilVisionNavigationEnv",
        )
    if LATENT_VISION_ENVIRONMENT_ID not in registry:
        register(
            id=LATENT_VISION_ENVIRONMENT_ID,
            entry_point="ebabil.envs.latent_vision:EbabilLatentVisionNavigationEnv",
        )
    if DETECTION_VISION_ENVIRONMENT_ID not in registry:
        register(
            id=DETECTION_VISION_ENVIRONMENT_ID,
            entry_point=(
                "ebabil.envs.detection_vision:EbabilDetectionVisionNavigationEnv"
            ),
        )


register_environments()

__all__ = [
    "DEFAULT_DETECTION_VISION_CONFIG",
    "DETECTION_VISION_ENVIRONMENT_ID",
    "DETECTION_VISION_OBSERVATION_KEYS",
    "DETECTION_VISION_OBSERVATION_VERSION",
    "EbabilDetectionVisionNavigationEnv",
    "DEFAULT_NAVIGATION_CONFIG",
    "DEFAULT_LATENT_VISION_CONFIG",
    "DEFAULT_VISION_NAVIGATION_CONFIG",
    "ENVIRONMENT_ID",
    "LATENT_VISION_ENVIRONMENT_ID",
    "LATENT_VISION_OBSERVATION_KEYS",
    "LATENT_VISION_OBSERVATION_VERSION",
    "EbabilLatentVisionNavigationEnv",
    "EbabilNavigationEnv",
    "EbabilVisionNavigationEnv",
    "NavigationPhysicsTrace",
    "NavigationEnvConfig",
    "OBSERVATION_COMPONENTS",
    "OBSERVATION_SIZE",
    "OBSERVATION_VERSION",
    "SegmentBudgetConfig",
    "SegmentHorizon",
    "SimulationDeadlineScheduler",
    "VISION_ENVIRONMENT_ID",
    "VISION_OBSERVATION_KEYS",
    "VISION_ACTION_VERSION",
    "VISION_OBSERVATION_VERSION",
    "VisionNavigationEnvConfig",
    "VisionObservationConfig",
    "compute_segment_horizon",
    "load_navigation_config",
    "load_vision_navigation_config",
    "register_environments",
]
