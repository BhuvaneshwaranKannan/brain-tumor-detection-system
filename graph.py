import matplotlib.pyplot as plt
import numpy as np

models = ["U-Net", "ResUNet", "Attention U-Net"]

# Existing model values (from literature approximation)
existing_dice = [0.89, 0.91, 0.92]
existing_iou = [0.81, 0.84, 0.86]

# Proposed system values (your results)
proposed_dice = [0.9279, 0.9479, 0.9279]
proposed_iou = [0.8655, 0.9010, 0.8655]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10,6))

plt.bar(x - width/2, existing_dice, width, label="Existing Dice")
plt.bar(x + width/2, proposed_dice, width, label="Proposed Dice")

plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Existing vs Proposed Model Performance (Dice Score)")
plt.legend()

plt.tight_layout()
plt.show()