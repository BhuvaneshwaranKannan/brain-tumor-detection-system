def estimate_stage(tumor_volume_cm3):
    """
    Estimate tumor stage using 3D tumor volume in cm³.
    """
    if tumor_volume_cm3 < 5.0:
        return "Stage 1"
    elif tumor_volume_cm3 < 15.0:
        return "Stage 2"
    elif tumor_volume_cm3 < 30.0:
        return "Stage 3"
    else:
        return "Stage 4"
