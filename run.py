'''
I really did write this.



TODO
1. search for gen_ans, implement the generate answer logic in this function
takes in a prompt and (dirtily) modify the global variables ans_vid (video) and ans_aud (audio)

- note: the audio representation is a path string, change this and play_audio call that consumes ans_aud too

2. replace the images with appropriate ones.
'''

import cv2
import speech_recognition as sr
import threading
import numpy as np
import time
from pydub import AudioSegment
from pydub.playback import play
from moviepy.editor import VideoFileClip

# use global when you wanna set the value in the func
# vids use mp4, auds use wav

MIRROR = 0
DARKEN = 1
REVEAL = 2
STATIC = 3
TOFIRE = 4
FLAMES = 5
TOFACE = 6
ANSWER = 7

WINDOW_NAME = 'Mirror'
MAGIC_WORD = "magic"

REVEAL_VID = 'start'
STATIC_VID = 'loop'
TOFIRE_VID = 'hello_world_video'
FLAMES_VID = 'ddy'
TOFACE_VID = 'hello_world_video'

state = MIRROR

darken_start_time = None
reveal_start_time = None
answer_end_time = 0

reveal_cap = None
static_cap = None
to_fire_cap = None
flames_cap = None
to_face_cap = None
ans_cap = None
ans_aud = None

def vid(v):
    return f'vids/{v}.mp4'

def aud(a):
    return f'auds/{a}.wav'

def init_vids():
    global reveal_cap, static_cap, to_fire_cap, flames_cap, to_face_cap
    reveal_cap = cv2.VideoCapture(vid(REVEAL_VID))
    static_cap = cv2.VideoCapture(vid(STATIC_VID))
    to_fire_cap = cv2.VideoCapture(vid(TOFIRE_VID))
    flames_cap = cv2.VideoCapture(vid(FLAMES_VID))
    to_face_cap = cv2.VideoCapture(vid(TOFACE_VID))

def release_vids():
    if reveal_cap is not None: reveal_cap.release()
    if static_cap is not None: static_cap.release()
    if to_fire_cap is not None: to_fire_cap.release()
    if flames_cap is not None: flames_cap.release()
    if to_face_cap is not None: to_face_cap.release()

def listen_for_command():
    global state, darken_start_time

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300  # minimum audio energy to consider for recording
    recognizer.dynamic_energy_threshold = True
    recognizer.dynamic_energy_adjustment_damping = 0.15
    recognizer.dynamic_energy_ratio = 1.5
    recognizer.pause_threshold = 0.8  # seconds of non-speaking audio before a phrase is considered complete
    recognizer.phrase_threshold = 0.3  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
    recognizer.non_speaking_duration = 0.5  # seconds of non-speaking audio to keep on both sides of the recording

    mic = sr.Microphone()

    with mic as source:
        print("Listening for command...")
        while True:
            audio = recognizer.listen(source)
            try:
                command = recognizer.recognize_google(audio).lower()
                print(f"You said: {command}")

                if state == MIRROR and MAGIC_WORD in command:
                    state = DARKEN
                    darken_start_time = time.time()
                    print("Starting to darken the video feed.")
                    
                elif state == STATIC and time.time() > answer_end_time+50:
                    print(f'state: {state}')
                    state = TOFIRE
                    print(f"Heard: {command}, Displaying tofire and thinking")
                    play_audio(TOFIRE_VID)
                    threading.Thread(target=gen_ans, args={command,}).start()

            except sr.UnknownValueError:
                print(f"--")
            except sr.RequestError:
                print(f"Could not request results from recognizer service")

def play_audio_worker(audio_path):
    audio = AudioSegment.from_file(audio_path)
    play(audio)

def play_audio(a):
    return threading.Thread(target=play_audio_worker, args={aud(a),}).start()

def resized(frame):
    _, _, w, h = cv2.getWindowImageRect(WINDOW_NAME)
    return cv2.resize(frame, (w, h))

# TODO
def gen_ans(prompt):
    global ans_cap, ans_aud
    print(f'prompt: {prompt}')

    time.sleep(3)
    ans_cap = cv2.VideoCapture(vid(FLAMES_VID))
    ans_aud = FLAMES_VID

def main():
    global state, ans_cap, ans_aud, answer_end_time

    threading.Thread(target=listen_for_command, daemon=True).start()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)

    init_vids()

    while True:
        cam_ret, cam_frame = cap.read()

        if not cam_ret:
            print("Error: Could not read frame from webcam")
            break
        
        frame = cam_frame
        
        if state == DARKEN:
            elapsed_darken_time = time.time() - darken_start_time
            if elapsed_darken_time <= 2:
                darkening_factor = 1 - (elapsed_darken_time / 2)
                darkened_frame = (frame.astype(np.float32) * darkening_factor).clip(0, 255).astype(np.uint8)
                frame = darkened_frame
            else:
                state = REVEAL
                reveal_start_time = time.time()
                play_audio(REVEAL_VID)

        if state == REVEAL:
            ret, frame = reveal_cap.read()
            if not ret: # video ends
                state = STATIC
                reveal_cap.release()
                continue

            # fade if within two seconds
            elapsed_reveal_time = time.time() - reveal_start_time
            if elapsed_reveal_time <= 2:
                fade_factor = elapsed_reveal_time / 2
                fade_frame = (frame.astype(np.float32) * fade_factor).clip(0, 255).astype(np.uint8)
                frame = fade_frame

        if state == STATIC:
            ret, frame = static_cap.read()
            if not ret:
                static_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        if state == TOFIRE:
            ret, frame = to_fire_cap.read()
            if not ret:
                to_fire_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state = FLAMES
                play_audio('elevator')
                continue
        
        if state == FLAMES:
            ret, frame = flames_cap.read()
            if not ret:
                flames_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if ans_cap is not None:
                state = TOFACE
                play_audio(TOFACE_VID)

        if state == TOFACE:
            ret, frame = to_face_cap.read()
            if not ret:
                to_face_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state = ANSWER
                play_audio(ans_aud)
                continue
    
        if state == ANSWER:
            ret, frame = ans_cap.read()
            if not ret: # video ends
                state = STATIC
                ans_cap.release()
                ans_cap = None
                ans_aud = None
                answer_end_time = time.time()
                continue

        if state != MIRROR:
            frame = resized(frame)
            
        cv2.imshow(WINDOW_NAME, frame)

        # necessary for some reason
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    release_vids()

if __name__ == '__main__':
    main()


