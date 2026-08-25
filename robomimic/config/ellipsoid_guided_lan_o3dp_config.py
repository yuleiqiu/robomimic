"""Config registration for LAN-O3DP swept-ellipsoid guidance."""

from robomimic.config.lan_o3dp_config import LanO3DPConfig


class EllipsoidGuidedLanO3DPConfig(LanO3DPConfig):
    ALGO_NAME = "ellipsoid_guided_lan_o3dp"
