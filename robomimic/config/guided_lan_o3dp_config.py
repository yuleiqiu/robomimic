"""Config registration for deployment-time guided LAN-O3DP inference."""

from robomimic.config.lan_o3dp_config import LanO3DPConfig


class GuidedLanO3DPConfig(LanO3DPConfig):
    ALGO_NAME = "guided_lan_o3dp"
