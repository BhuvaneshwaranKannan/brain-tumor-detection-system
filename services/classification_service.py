import cv2
import numpy as np

def classify(tumor_crop, classifier_model, classes):
    """
    Prepare tumor crop for classifier input and run tumor_type_model.h5.
    """
    target_size = (128, 128)
    
    # Check if crop is valid
    if tumor_crop is None or tumor_crop.size == 0:
        return "Unknown", 0.0
        
    slice_img = cv2.normalize(tumor_crop, None, 0, 255, cv2.NORM_MINMAX)
    slice_img = cv2.cvtColor(slice_img.astype("float32"), cv2.COLOR_GRAY2RGB)
    slice_img = cv2.resize(slice_img, target_size)
    slice_img = slice_img / 255.0
    slice_img = np.expand_dims(slice_img, axis=0)

    pred = classifier_model.predict(slice_img)[0]
    class_index = np.argmax(pred)
    tumor_type = classes[class_index]
    confidence = float(np.max(pred)) * 100

    return tumor_type, confidence
