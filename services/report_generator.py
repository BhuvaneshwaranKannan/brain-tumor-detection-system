import datetime
import numpy as np

def _calculate_tumor_characteristics(mask, image_shape):
    if mask is None or np.sum(mask) == 0:
        return {
            "location": "N/A",
            "region": "N/A",
            "bbox": "N/A",
            "area_pixels": 0
        }

    # Bounding Box
    coords = np.argwhere(mask > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    height = y_max - y_min
    width = x_max - x_min
    bbox = f"{width}px \u00d7 {height}px"
    
    # Left / Right mapping based on center of mass vs image center width
    cy = np.mean([y_min, y_max])
    cx = np.mean([x_min, x_max])
    
    mid_x = image_shape[1] / 2
    # In medical imaging, left on image is usually right side of patient, but we keep it simple for now based on screen coords
    location = "Right Hemisphere" if cx > mid_x else "Left Hemisphere"

    # Minimal simulated region logic based on y axis (top vs bottom)
    mid_y = image_shape[0] / 2
    if cy < mid_y * 0.75:
        region = "Frontal"
    elif cy > mid_y * 1.25:
        region = "Occipital"
    else:
        region = "Temporal/Parietal"

    return {
        "location": location,
        "region": region,
        "bbox": bbox,
        "area_pixels": len(coords)
    }

def _get_risk_level(stage):
    if stage == "Stage 4": return "High"
    if stage == "Stage 3": return "High"
    if stage == "Stage 2": return "Moderate"
    if stage == "Stage 1": return "Low"
    return "Unknown"

def _get_findings_text(tumor_type, location, region, size, risk):
    if risk in ["High", "Moderate"]:
        return f"A distinct lesion is observed in the {location} ({region} region). Morphology is consistent with {tumor_type.title()} and presents significant mass effect measuring {size} cm\u00b3."
    return f"A well-defined lesion is noted in the {location} ({region} region), suggestive of {tumor_type.title()} with an estimated volume of {size} cm\u00b3."

def _get_recommendations(risk):
    if risk == "High":
        return [
            "Immediate oncology/neurology consultation",
            "Urgent biopsy and surgical evaluation",
            "Advanced imaging (e.g., contrast-enhanced MRI or PET)"
        ]
    elif risk == "Moderate":
        return [
            "Referral to neurology for specialist review",
            "Follow-up scan in 3 months",
            "Functional MRI for pre-operative planning"
        ]
    else:
        return [
            "Routine monitoring",
            "Follow-up scan in 6 months to monitor progression"
        ]

def _generate_symptoms(tumor_type, risk):
    symptoms = {
        "glioma": ["Frequent headaches", "Mild to severe cognitive issues", "Seizures", "Weakness on one side"],
        "meningioma": ["Vision changes", "Mild headaches", "Focal seizures", "Memory loss"],
        "pituitary": ["Hormonal imbalances", "Visual field defects", "Fatigue", "Weight changes"]
    }
    
    base_symptoms = symptoms.get(tumor_type.lower(), ["Unspecified neurological symptoms"])
    
    if risk == "High":
        base_symptoms = [s.replace("Mild", "Severe").replace("Frequent", "Intense") for s in base_symptoms]
        base_symptoms.append("Significant neurological deficits")
    elif risk == "Moderate":
        if "Significant neurological deficits" not in base_symptoms:
            base_symptoms.append("Progressive neurological symptoms")

    return ", ".join(base_symptoms)

def generate_clinical_report(data):
    """
    data dictionary requires:
    - patient_name
    - date_str
    - scan_type
    - organ
    - detection (str)
    - tumor_type (str)
    - confidence (float)
    - mask (numpy array)
    - image_shape (tuple)
    - stage (str)
    - volume (float)
    """
    
    report = {
        "patient_info": {
            "name": data.get("patient_name", "Unknown"),
            "scan_date": data.get("date_str", datetime.datetime.now().strftime("%Y-%m-%d")),
            "scan_type": data.get("scan_type", "MRI"),
            "organ": data.get("organ", "Brain")
        }
    }

    if "No Tumor Detect" in data.get("detection", "") or float(data.get("volume", 0)) == 0.0:
        report["detection_summary"] = {
            "tumor_detected": "Not Detected",
            "tumor_type": "None",
            "confidence": 0.0
        }
        report["tumor_characteristics"] = {
            "location": "N/A",
            "region": "N/A",
            "volume_cm3": "0",
            "bbox": "N/A"
        }
        report["severity_assessment"] = {
            "stage": "None",
            "risk_level": "None"
        }
        report["clinical_findings"] = "Scan appears normal. No anomalous growths, lesions, or abnormal tissue densities detected in the brain."
        report["symptoms"] = "None"
        report["recommendations"] = ["Routine health monitoring", "No immediate medical action required"]
        
    else:
        chars = _calculate_tumor_characteristics(data.get("mask"), data.get("image_shape", (128, 128)))
        risk = _get_risk_level(data.get("stage", "Unknown"))
        v_cm3 = round(data.get("volume", 0), 2)
        
        report["detection_summary"] = {
            "tumor_detected": "Detected",
            "tumor_type": data.get("tumor_type", "Unknown").title(),
            "confidence": f"{data.get('confidence', 0)}%"
        }
        report["tumor_characteristics"] = {
            "location": chars["location"],
            "region": chars["region"],
            "volume_cm3": str(v_cm3),
            "bbox": chars["bbox"]
        }
        report["severity_assessment"] = {
            "stage": data.get("stage", "Unknown"),
            "risk_level": risk
        }
        
        report["clinical_findings"] = _get_findings_text(
            report["detection_summary"]["tumor_type"], 
            chars["location"], 
            chars["region"], 
            v_cm3, 
            risk
        )
        report["symptoms"] = _generate_symptoms(data.get("tumor_type", ""), risk)
        report["recommendations"] = _get_recommendations(risk)

    return report
