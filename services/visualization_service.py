import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_overlay(image, mask, title, output_path):
    """
    Generate overlay visualization of tumor mask on MRI slice using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(image, cmap="gray")
    ax.imshow(mask, cmap="jet", alpha=0.4)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)
    return output_path
