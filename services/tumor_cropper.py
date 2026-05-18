import cv2

def crop_tumor(slice_img, mask):
    """
    Use cv2.boundingRect on the mask to crop the tumor region from the MRI slice.
    """
    x, y, w, h = cv2.boundingRect(mask)
    
    # If no tumor found in mask, skip classification
    if w == 0 or h == 0:
        return None
        
    return slice_img[y:y+h, x:x+w]
