# src/domain/params/rmp.py
def calculate_rmp_lid(bt: float, raw_flag: int) -> float:
    return raw_flag / abs(bt) if bt else 0.0