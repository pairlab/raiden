from raiden.control.base import TeleopInterface
from raiden.control.oculus import OculusInterface
from raiden.control.spacemouse import SpaceMouseInterface
from raiden.control.yam import YAMInterface

__all__ = [
    "TeleopInterface",
    "YAMInterface",
    "SpaceMouseInterface",
    "OculusInterface",
    "build_interface",
]


def build_interface(
    control: str,
    spacemouse_path_r: str = "/dev/hidraw7",
    spacemouse_path_l: str = "/dev/hidraw6",
    vel_scale: float = 0.07,
    rot_scale: float = 0.8,
    invert_rotation: bool = False,
    oculus_hand: str = "l",
    oculus_pos_scale: float = 0.7,
    oculus_rot_scale: float = 0.5,
    oculus_ip: str = "",
) -> TeleopInterface:
    """Construct the right TeleopInterface from CLI-style arguments."""
    if control == "oculus":
        return OculusInterface(
            ip_address=oculus_ip or None,
            hand_for_left_arm=oculus_hand,
            pos_scale=oculus_pos_scale,
            rot_scale=oculus_rot_scale,
        )
    if control == "spacemouse":
        return SpaceMouseInterface(
            path_r=spacemouse_path_r,
            path_l=spacemouse_path_l,
            vel_scale=vel_scale,
            rot_scale=rot_scale,
            invert_rotation=invert_rotation,
        )
    return YAMInterface()
