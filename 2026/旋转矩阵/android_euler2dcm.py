# Euler angles to DCM - Android coordinate convention
#
# Coordinate system (right-hand):
#   X = lateral  (positive = right)   -> Pitch axis
#   Y = forward  (positive = forward) -> Roll  axis
#   Z = up       (positive = up)      -> Yaw   axis
#
# Input:
#   in[0] = roll  (rotation about Y axis, forward)
#   in[1] = pitch (rotation about X axis, lateral)
#   in[2] = yaw   (rotation about Z axis, up)
#   unit  : degrees
#
# Rotation order: R = Rz(yaw) * Ry(roll) * Rx(pitch)
#
# Full matrix (cr=cos(roll), sr=sin(roll), cp=cos(pitch), sp=sin(pitch), cy=cos(yaw), sy=sin(yaw)):
#
#   R = | cy*cr,   cy*sr*sp - sy*cp,   cy*sr*cp + sy*sp |
#       | sy*cr,   sy*sr*sp + cy*cp,   sy*sr*cp - cy*sp |
#       | -sr,     cr*sp,              cr*cp             |

import numpy as np


def android_euler2dcm(euler_deg):
    """
    Convert Euler angles to 3x3 rotation matrix (row-major).

    Android right-hand coordinate system:
        X = lateral (right),  Pitch axis
        Y = forward (front),  Roll  axis
        Z = up,               Yaw   axis

    Parameters
    ----------
    euler_deg : array-like, length 3
        euler_deg[0] = roll  (deg), rotation about Y (forward axis)
        euler_deg[1] = pitch (deg), rotation about X (lateral axis)
        euler_deg[2] = yaw   (deg), rotation about Z (up axis)

    Returns
    -------
    dcm : numpy array, shape (3, 3), row-major
        R = Rz(yaw) * Ry(roll) * Rx(pitch)
    """
    roll  = np.radians(euler_deg[0])
    pitch = np.radians(euler_deg[1])
    yaw   = np.radians(euler_deg[2])

    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    dcm = np.array([
        [ cy*cr,   cy*sr*sp - sy*cp,   cy*sr*cp + sy*sp ],
        [ sy*cr,   sy*sr*sp + cy*cp,   sy*sr*cp - cy*sp ],
        [ -sr,     cr*sp,              cr*cp             ]
    ])
    return dcm


# ── Test ──────────────────────────────────────────────────────────
if __name__ == "__main__":

    test_cases = [
        # [roll, pitch, yaw],  description
        ([0.0,   0.0,  0.0],  "identity"),
        ([0.0,   0.0, 90.0],  "yaw 90 deg (turn left)"),
        ([19.3,  0.0, 90.0],  "roll 19.3 + yaw 90"),
        ([0.0,  19.3, 180.0], "pitch 19.3 + yaw 180"),
    ]

    for euler, desc in test_cases:
        dcm = android_euler2dcm(euler)
        print(f"\n--- {desc} ---")
        print(f"Input: roll={euler[0]} deg  pitch={euler[1]} deg  yaw={euler[2]} deg")
        print("DCM (3x3 row-major):")
        print(np.array2string(dcm, formatter={'float_kind': lambda x: f"{x:8.4f}"}))
        flat = dcm.flatten()
        print("C array dcm[9]:")
        print("{ " + ", ".join(f"{v:.4f}f" for v in flat) + " }")
