import os
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from pathlib import Path

def get_sorted_png_files(directory):
    png_files = [file for file in Path(directory).glob('*.png')]
    png_files.sort(key=lambda x: int(re.search(r'\d+', x.stem).group()))  # Sort by the number in the filenames
    return png_files

def update_plot(frame):
    ax.clear()
    image_path = sorted_png_files[frame]
    image = Image.open(image_path)
    ax.imshow(image)
    ax.set_title(f"Frame {frame + 1}/{len(sorted_png_files)}")
    # plt.pause(0.1)  # Pause to allow time for rendering

if __name__ == "__main__":
    directory_path = "C:\\Users\\Oliver\\OneDrive - UGent\\Studiejaar 2023-2024\\Machine Design\\Computational assignment\\Project_MD\\Figures"  # Change this to the directory containing your PNG files

    sorted_png_files = get_sorted_png_files(directory_path)

    if not sorted_png_files:
        print("No PNG files found in the specified directory.")
    else:
        # Create a figure and axis for plotting
        fig, ax = plt.subplots()

        # Create an animation to update the plot
        animation = FuncAnimation(fig, update_plot, frames=len(sorted_png_files), repeat=False)

        animation.save('output_animation.mp4', writer='ffmpeg', fps=30)  # Adjust fps as needed
        

