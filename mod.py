'''
Run this to get the 
'''


from TTS.api import TTS
import time
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to('mps')

def gen_audio(text, ref):
    return tts.tts_to_file(
        text=text,
        # file_path="output.wav",
        speaker_wav=[ref],
        language="en",
        split_sentences=True,        
    )


while True:
    prompt = input()

    print(time.time())
    res = gen_audio(
        "Hello Princess.",
        "scary.wav"
    )
    print(time.time())
