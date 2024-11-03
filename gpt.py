'''
Yeah, entirely by me.
'''

import cv2
import speech_recognition as sr
import threading
import numpy as np
import time
from pydub import AudioSegment
from pydub.playback import play  # Import play function to play audio

# Global variables to control mask state
apply_mask = False
darkening_start_time = None
video_playing = False
video_cap = None
looping_video_cap = None
third_video_cap = None
video_frame = None
fade_in_start_time = None  # Track when the fade-in starts
command_listening_enabled = True  # Track if command detection should still be active
display_third_video = False  # Track if the third video should be displayed

def listen_for_command():
    global apply_mask, darkening_start_time, command_listening_enabled, display_third_video
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening for command...")
        while True:
            audio = recognizer.listen(source)
            try:
                command = recognizer.recognize_google(audio).lower()
                print(f"You said: {command}")
                
                # Check if command detection is enabled
                if command_listening_enabled and "mirror" in command:
                    apply_mask = True
                    darkening_start_time = time.time()  # Start the darkening timer
                    print("Starting to darken the video feed.")
                    command_listening_enabled = False  # Disable further command detection
                elif not command_listening_enabled:
                    # Trigger the third video display
                    print("Displaying third video temporarily.")
                    print(f'video_playing: {video_playing}')
                    print(f'command_listening_enabled: {command_listening_enabled}')
                    print(f'display_third_video: {command_listening_enabled}')
                    display_third_video = True  # Set flag to display third video
            except sr.UnknownValueError:
                print("Sorry, I did not understand that.")
            except sr.RequestError:
                print("Could not request results from Google Speech Recognition service.")

def play_audio(audio_path):
    # Load and play the audio file using pydub
    audio = AudioSegment.from_file(audio_path)
    play(audio)  # Play the audio

def main(video_path, looping_video_path, third_video_path, custom_audio_path):
    global apply_mask, darkening_start_time, video_playing, video_cap, looping_video_cap, third_video_cap
    global fade_in_start_time, display_third_video

    # Start the voice command listener in a separate thread
    threading.Thread(target=listen_for_command, daemon=True).start()

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Set the video frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Open the main video file
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print("Error: Could not open main video file.")
        cap.release()
        return

    # Open the looping video file
    looping_video_cap = cv2.VideoCapture(looping_video_path)
    if not looping_video_cap.isOpened():
        print("Error: Could not open looping video file.")
        cap.release()
        video_cap.release()
        return

    # Open the third video file
    third_video_cap = cv2.VideoCapture(third_video_path)
    if not third_video_cap.isOpened():
        print("Error: Could not open third video file.")
        cap.release()
        video_cap.release()
        looping_video_cap.release()
        return

    while True:
        # Capture frame-by-frame from webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Apply darkening effect if apply_mask is True
        if apply_mask:
            elapsed_time = time.time() - darkening_start_time
            if elapsed_time <= 2:  # Darkening duration is now 2 seconds
                darkening_factor = 1 - (elapsed_time / 2)  # Gradually decrease from 1 to 0
                darkened_frame = (frame.astype(np.float32) * darkening_factor).clip(0, 255).astype(np.uint8)
                frame = darkened_frame
            else:
                apply_mask = False  # Stop applying the mask after 2 seconds
                video_playing = True  # Start playing the video
                fade_in_start_time = time.time()  # Start the fade-in effect
                threading.Thread(target=play_audio, args=(custom_audio_path,), daemon=True).start()  # Start the audio playback thread

        # If the third video display is triggered
        if display_third_video:
            ret_third, third_frame = third_video_cap.read()
            if not ret_third:
                third_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the third video for next trigger
                display_third_video = False  # Reset flag to stop showing third video
                continue

            # Resize third video frame to match webcam frame size
            third_frame = cv2.resize(third_frame, (640, 480))
            frame = third_frame

        # If the main or looping video is playing
        elif video_playing:
            if video_playing == "looping":
                # If in looping mode, read from looping video
                ret_looping, loop_frame = looping_video_cap.read()
                if not ret_looping:
                    looping_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart looping video from the beginning
                    continue

                # Resize looping video frame to match webcam frame size
                loop_frame = cv2.resize(loop_frame, (640, 480))
                frame = loop_frame

            elif video_playing == True:
                ret_video, video_frame = video_cap.read()
                if not ret_video:
                    # Main video playback finished, switch to the looping video
                    video_playing = "looping"
                    video_cap.release()  # Release the main video capture
                    video_cap = None  # Reset the main video capture
                    continue  # Skip the frame for webcam feed

                # Resize video frame to match webcam frame size
                video_frame = cv2.resize(video_frame, (640, 480))

                # Apply fade-in effect by overlaying a black tint
                elapsed_fade_time = time.time() - fade_in_start_time
                if elapsed_fade_time <= 2:  # Fade-in duration is 2 seconds
                    fade_factor = 1 - (elapsed_fade_time / 2)  # Fade factor decreases from 1 to 0
                    black_overlay = np.zeros_like(video_frame, dtype=np.uint8)
                    faded_frame = cv2.addWeighted(black_overlay, fade_factor, video_frame, 1 - fade_factor, 0)
                    frame = faded_frame
                else:
                    # No overlay once fade-in is complete
                    frame = video_frame

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # If the 'q' key is pressed, exit the loop
        if cv2.waitKey(1) == ord('q'):
            print("Exiting the camera feed.")
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    if video_cap is not None:
        video_cap.release()
    if looping_video_cap is not None:
        looping_video_cap.release()
    if third_video_cap is not None:
        third_video_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_video_path = "start.mp4"  # Replace with the path to your main video file
    looping_video_path = "loop.mp4"  # Replace with the path to your looping video file
    third_video_path = "test2.mov"  # Replace with the path to your third video file
    custom_audio_path = "hello_world.mp3"  # Replace with the path to your custom audio file
    main(main_video_path, looping_video_path, third_video_path, custom_audio_path)
