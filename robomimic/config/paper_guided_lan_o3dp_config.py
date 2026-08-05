"""Configuration registration for paper-equation LAN guidance inference."""

from robomimic.config.lan_o3dp_config import LanO3DPConfig


class PaperGuidedLanO3DPConfig(LanO3DPConfig):
    ALGO_NAME = "paper_guided_lan_o3dp"
