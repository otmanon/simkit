import os
import subprocess
import glob

def video_from_image_dir(
    image_folder, video_name=None, fps=60, width=None, height=None, mogrify=False
):
    """Make a video from a folder of images using ffmpeg."""
    if video_name is None:
        video_name = os.path.join(image_folder, "video.mp4")
    else:
        # Ensure video_name has .mp4 extension
        if not video_name.endswith('.mp4'):
            video_name = os.path.splitext(video_name)[0] + '.mp4'

    if os.path.exists(image_folder):
        # Get all PNG images and sort them
        images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

        if not images:
            print("No PNG images found in the specified folder.")
            return

        # Get dimensions from first image if not specified
        if width is None or height is None:
            # Use ffprobe to get image dimensions
            first_image = os.path.join(image_folder, images[0])
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                   '-show_entries', 'stream=width,height', '-of', 'csv=p=0', first_image]
            result = subprocess.run(cmd, capture_output=True, text=True)
            width, height = map(int, result.stdout.strip().split(','))

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(video_name)), exist_ok=True)

        # Create ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-framerate', str(fps),
            '-i', os.path.join(image_folder, '%04d.png'),  # Input pattern
            '-filter_complex', f'[0]format=rgba[fg];color=white:s={width}x{height}[bg];[bg][fg]overlay=shortest=1',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            os.path.abspath(video_name)  # Use absolute path for output
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Video created successfully as {video_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating video: {e.stderr.decode()}")
    else:
        print(f"The specified image folder '{image_folder}' does not exist.")
