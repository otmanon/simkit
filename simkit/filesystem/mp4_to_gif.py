
from PIL import Image
import cv2
import subprocess
import os

def mp4_to_gif(input_path, output_path, fps=30, scale=None, bitrate='8192k', crf=15):
    """
      Convert an MP4 file to a very high-resolution GIF using ffmpeg with the same resolution as the input video.

      Parameters:
      - input_path: Path to the input MP4 file.
      - output_path: Path to save the output GIF file.
      - fps: Frames per second for the GIF (default is 30).
      - bitrate: Bitrate for the output GIF (default is '8192k').
      - crf: Constant Rate Factor for the output GIF (default is 15).
      """
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return
    else:
        print("FIle exists")
    try:
        # Get video resolution
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of',
             'csv=s=x:p=0', input_path],
            stdout=subprocess.PIPE, text=True
        )
        width, height = map(int, result.stdout.strip().split('x'))

        # Convert to GIF with the same resolution
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file without asking
            '-i', input_path,
            '-vf', f'fps={fps},scale={width}:{height}:flags=lanczos',
            '-c:v', 'gif',
            '-b:v', bitrate,
            '-crf', str(crf),
            output_path
        ]

        subprocess.run(command, check=True)

        print(f"Conversion successful. GIF saved at {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")



# def mp4_to_gif(input_path, output_path):
#     """
#     Convert an MP4 file to a GIF using PIL by extracting frames with OpenCV.
    
#     Parameters:
#     - input_path: Path to the input MP4 file.
#     - output_path: Path to save the output GIF file.
#     - fps: Frames per second for the GIF (default is 30).
#     """
#     try:
#         # Create a list to hold frames
#         frames = []

#         # Load the video using OpenCV
#         cap = cv2.VideoCapture(input_path)
#         video_fps = cap.get(cv2.CAP_PROP_FPS)
#         # frame_skip = int(video_fps / fps)

#         frame_count = 0
#         while True:
#             # Read frame-by-frame
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Only capture frames at the desired FPS rate
#             # if frame_count % frame_skip == 0:
#                 # Convert frame to RGB and then to PIL Image
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             pil_image = Image.fromarray(frame_rgb)
#             frames.append(pil_image)

#             frame_count += 1

#         cap.release()

#         # Save frames as a GIF
#         frames[0].save(
#             output_path,
#             save_all=True,
#             append_images=frames[1:],
#             optimize=False,
#             # duration=int(1000 / fps),
#             disposal=2,
#             loop=0
#         )
#         print(f"Conversion successful. GIF saved at {output_path}")
#     except Exception as e:
#         print(f"Error during conversion: {e}")

# from PIL import Image
# import cv2

# def mp4_to_gif(input_path, output_path, dither=True):
#     """
#     Convert an MP4 file to a GIF with automatic FPS detection and enhanced color preservation.

#     Parameters:
#     - input_path: Path to the input MP4 file.
#     - output_path: Path to save the output GIF file.
#     - dither: Apply dithering to improve color quality (default is True).
#     """
#     try:
#         frames = []
#         cap = cv2.VideoCapture(input_path)
#         video_fps = cap.get(cv2.CAP_PROP_FPS)
#         duration_per_frame = int(1000 / video_fps)  # Calculate frame duration for the GIF in ms

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Convert frame to RGB and then to a PIL Image
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             pil_image = Image.fromarray(frame_rgb)

#             # Convert to P mode with dithering for better color quality
#             if dither:
#                 pil_image = pil_image.convert("P", palette=Image.ADAPTIVE, dither=Image.FLOYDSTEINBERG)
#             else:
#                 pil_image = pil_image.convert("P", palette=Image.ADAPTIVE)

#             frames.append(pil_image)

#         cap.release()

#         # Save frames as a GIF with the video’s FPS-based duration
#         frames[0].save(
#             output_path,
#             save_all=True,
#             append_images=frames[1:],
#             optimize=False,
#             duration=duration_per_frame,
#             loop=0
#         )
#         print(f"Conversion successful. GIF saved at {output_path}")

#     except Exception as e:
#         print(f"Error during conversion: {e}")

# Example usage:
# mp4_to_gif("input_video.mp4", "output_animation.gif")



# def mp4_to_gif(input_path, output_path, fps=30):
#     """
#     Convert an MP4 file to a GIF using OpenCV and imageio.

#     Parameters:
#     - input_path: Path to the input MP4 file.
#     - output_path: Path to save the output GIF file.
#     - fps: Frames per second for the GIF (default is 10).
#     """
#     # Open the video file
#     video_capture = cv2.VideoCapture(input_path)

#     # Get video properties
#     width = int(video_capture.get(3))
#     height = int(video_capture.get(4))

#     # Create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     # Read and write frames until the video is over
#     while video_capture.isOpened():
#         ret, frame = video_capture.read()

#         if not ret:
#             break

#         # Write the frame to the output video
#         video_writer.write(frame)

#     # Release resources
#     video_capture.release()
#     video_writer.release()