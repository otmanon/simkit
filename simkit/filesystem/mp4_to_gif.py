
import subprocess

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