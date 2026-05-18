import cv2
import numpy as np

def postprocess_mask(mask):
    """Postprocess the mask to keep largest component and fill holes."""
    mask = mask.astype(np.uint8)

    # keep largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    # fill holes safely by padding
    mask_padded = np.pad(mask, 1, mode='constant', constant_values=0)
    h, w = mask_padded.shape
    flood_mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(mask_padded, flood_mask, (0,0), 2)
    mask_filled = (mask_padded != 2).astype(np.uint8)
    mask = mask_filled[1:-1, 1:-1]

    return mask

def segment(input_img, unet_model=None, resunet_model=None, attention_model=None):
    """
    Run the segmentation ensemble using models: x1.h5 (UNet), y1.h5 (ResUNet), z1.h5 (Attention UNet).
    Average predictions and return tumor mask.
    """
    preds = []
    if unet_model:
        preds.append(unet_model.predict(input_img)[0,:,:,0])
    if resunet_model:
        preds.append(resunet_model.predict(input_img)[0,:,:,0])
    if attention_model:
        preds.append(attention_model.predict(input_img)[0,:,:,0])

    if not preds:
        return np.zeros(input_img.shape[1:3], dtype=np.uint8)

    final_pred = np.mean(preds, axis=0)
    mask = (final_pred > 0.5).astype(np.uint8)
    mask = postprocess_mask(mask)
    return mask
