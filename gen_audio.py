'''
Run this after you change content in vids.

Pre-extract the audio files

(Because the audio is played in a seperate thread...)
'''


import os
from moviepy.editor import VideoFileClip

ignore = ('start', 'face_to_fire')
# ignore = ()

# Path to the folder containing video files
video_folder = "vids"  # Change this to your folder path

# Create an output folder for audio files if it doesn't exist
audio_folder = "auds"
os.makedirs(audio_folder, exist_ok=True)

# Iterate over all files in the video folder
for filename in os.listdir(video_folder):
    if filename.startswith(ignore):
        continue
    if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):  # Supported video formats
        video_path = os.path.join(video_folder, filename)
        audio_path = os.path.join(audio_folder, f"{os.path.splitext(filename)[0]}.wav")  # Save as WAV

        try:
            # Load the video file
            video_clip = VideoFileClip(video_path)

            # Extract audio and save it to the audio folder
            video_clip.audio.write_audiofile(audio_path)

            print(f"Extracted audio from {filename} and saved as {audio_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        finally:
            video_clip.close()  # Ensure the video clip is properly closed

print("Audio extraction completed.")
