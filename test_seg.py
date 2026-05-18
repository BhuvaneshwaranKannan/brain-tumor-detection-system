import numpy as np
import cv2
from services.segmentation_service import postprocess_mask

mask = np.zeros((128, 128), dtype=np.uint8)
# Add a hollow square (tumor with a hole)
cv2.rectangle(mask, (30, 30), (80, 80), 1, -1)
cv2.rectangle(mask, (40, 40), (70, 70), 0, -1) # Hole

print("Original sum:", np.sum(mask))
out = postprocess_mask(mask)
print("Processed sum:", np.sum(out))
print("Processed shape:", out.shape)
print("Finished without error.")
