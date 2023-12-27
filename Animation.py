import os
import re
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def get_sorted_png_files(directory):
    png_files = [file for file in Path(directory).glob('*.png')]
    png_files.sort(key=lambda x: int(re.search(r'\d+', x.stem).group()))  # Sort by the number in the filenames
    return png_files

def display_sequence_of_figures(directory, pause_duration=1):
    sorted_png_files = get_sorted_png_files(directory)

    if not sorted_png_files:
        print("No PNG files found in the specified directory.")
    else:
        fig, ax = plt.subplots()  # Create a single figure
        for i, image_path in enumerate(sorted_png_files):
            image = Image.open(image_path)
            ax.clear()
            ax.imshow(image)
            ax.set_title(f"Frame {i + 1}/{len(sorted_png_files)}")

            # Display each figure with a pause
            plt.pause(pause_duration)

if __name__ == "__main__":
    directory_path = "C:\\Users\\Oliver\\OneDrive - UGent\\Studiejaar 2023-2024\\Machine Design\\Computational assignment\\Project_MD\\Figures"  # Change this to the directory containing your PNG files

    display_sequence_of_figures(directory_path, pause_duration=0.01)

        

