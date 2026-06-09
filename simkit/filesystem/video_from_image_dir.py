# import os

# import cv2


# def video_from_image_dir(
#     image_folder, video_name=None, fps=60, width=None, height=None, mogrify=False
# ):
#     """Make a video from a folder of images."""
#     if video_name is None:
#         video_name = os.path.join(image_folder, "video.mp4")

#     if os.path.exists(image_folder):
#         images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

#         if not images:
#             print("No PNG images found in the specified folder.")
#             return

#         # Sort the images based on their filenames (assuming filenames are integers)
#         images.sort(key=lambda x: int(x.split(".")[0]))

#         frame = cv2.imread(os.path.join(image_folder, images[0]))
#         h, w, layers = frame.shape

#         if height is None or width is None:
#             height = h
#             width = w

#         video = cv2.VideoWriter(
#             video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
#         )

#         for image in images:
#             img_path = os.path.join(image_folder, image)

#             # Read the image using cv2.IMREAD_UNCHANGED to handle images with an alpha channel
#             img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

#             if mogrify:
#                 if img.shape[2] == 4:  # If image has alpha channel
#                     alpha_channel = img[:, :, 3]
#                     img[alpha_channel < 10] = [
#                         255,
#                         255,
#                         255,
#                         255,
#                     ]  # Set transparent pixels to white

#                     # also set all black to white
#                     img[
#                         (img[:, :, 0] < 10) & (img[:, :, 1] < 10) & (img[:, :, 2] < 10)
#                     ] = [255, 255, 255, 255]

#             if img.shape[2] == 4:  # If image has alpha channel
#                 img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

#             # Resize the image
#             img = cv2.resize(img, (width, height))

#             # If the image has an alpha channel, remove it to avoid potential issues
#             if img.shape[-1] == 4:
#                 img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

#             video.write(img)

#         cv2.destroyAllWindows()
#         video.release()
#     else:
#         print(f"The specified image folder '{image_folder}' does not exist.")

# import cv2
# import os
# import numpy as np

# def apply_gamma_correction(image, gamma=2.2):
#     """Apply gamma correction to the image to adjust brightness."""
#     inv_gamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(image, table)

# def video_from_image_dir(
#     image_folder, video_name=None, fps=60, width=None, height=None, mogrify=False, gamma_correction=False
# ):
#     """Make a video from a folder of images."""
#     if video_name is None:
#         video_name = os.path.join(image_folder, "video.mp4")

#     if os.path.exists(image_folder):
#         images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

#         if not images:
#             print("No PNG images found in the specified folder.")
#             return

#         images.sort(key=lambda x: int(x.split(".")[0]))

#         frame = cv2.imread(os.path.join(image_folder, images[0]))
#         h, w, layers = frame.shape

#         if height is None or width is None:
#             height = h
#             width = w

#         video = cv2.VideoWriter(
#             video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
#         )

#         for image in images:
#             img_path = os.path.join(image_folder, image)
#             img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

#             if mogrify:
#                 if img.shape[2] == 4:  # If image has an alpha channel
#                     alpha_channel = img[:, :, 3]
#                     img[alpha_channel < 10] = [255, 255, 255, 255]
#                     img[
#                         (img[:, :, 0] < 10) & (img[:, :, 1] < 10) & (img[:, :, 2] < 10)
#                     ] = [255, 255, 255, 255]

#             if img.shape[2] == 4:  # If image has alpha channel
#                 img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

#             img = cv2.resize(img, (width, height))

#             if gamma_correction:
#                 img = apply_gamma_correction(img, gamma=2.2)

#             video.write(img)

#         video.release()
#         cv2.destroyAllWindows()
#     else:
#         print(f"The specified image folder '{image_folder}' does not exist.")



import os
import numpy as np
import subprocess
from PIL import Image

def apply_gamma_correction(image, gamma=2.2):
    """Apply gamma correction to the image to adjust brightness."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return table[image]

def preprocess_images(image_folder, output_folder, mogrify=False, gamma_correction=False):
    """Preprocess images in the folder (mogrify, gamma correction) and save them to output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    if not images:
        print("No PNG images found in the specified folder.")
        return

    images.sort(key=lambda x: int(x.split(".")[0]))

    for i, image in enumerate(images):
        img_path = os.path.join(image_folder, image)
        img = np.array(Image.open(img_path))

        if mogrify:
            if img.shape[2] == 4:  # If image has an alpha channel
                alpha_channel = img[:, :, 3]
                img[alpha_channel < 10] = [255, 255, 255, 255]
                img[(img[:, :, 0] < 10) & (img[:, :, 1] < 10) & (img[:, :, 2] < 10)] = [255, 255, 255, 255]

        if img.shape[2] == 4:  # If image has alpha channel
            img = img[:, :, :3]  # drop alpha (RGBA -> RGB)

        if gamma_correction:
            img = apply_gamma_correction(img, gamma=2.2)

        # Save preprocessed image to the output folder
        output_img_path = os.path.join(output_folder, f"{i:04d}.png")
        Image.fromarray(img).save(output_img_path)

def create_video_with_ffmpeg(image_folder, video_name, fps=60):
    """Use ffmpeg to create a video from the preprocessed images."""
    # Construct the ffmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-r", str(fps),  # Frame rate
        "-f", "image2",  # Image sequence input
        "-i", os.path.join(image_folder, "%04d.png"),  # Input images as numbered sequence
        "-vcodec", "libx264",  # Use the H.264 codec
        "-crf", "25",  # Quality setting
        "-pix_fmt", "yuv420p",  # Pixel format
        video_name  # Output video name
    ]
    
    # Run the ffmpeg command
    subprocess.run(ffmpeg_command)

# Main function
def video_from_image_dir(
    image_folder, video_name=None, fps=60, width=None, height=None, mogrify=False, gamma_correction=False
):
    """Preprocess images, then use ffmpeg to create the video."""
    if video_name is None:
        video_name = os.path.join(image_folder, "video.mp4")

    output_folder = os.path.join(image_folder, "preprocessed")

    # Preprocess the images first
    preprocess_images(image_folder, output_folder, mogrify=mogrify, gamma_correction=gamma_correction)

    # Create the video using ffmpeg
    create_video_with_ffmpeg(output_folder, video_name, fps)

# # Example usage
# video_from_image_dir(
#     image_folder="path_to_your_image_folder",
#     video_name="output_video.mp4",
#     fps=30,
#     gamma_correction=True
# )
