'''
Run this to get the 
'''


from TTS.api import TTS
import time
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

def gen_audio(text, ref):
    tts.tts_to_file(
        text=text,
        file_path="output.wav",
        speaker_wav=[ref],
        language="en",
        split_sentences=True
    )

print(time.time())
gen_audio(
    "What wouldst thou know? My Queen? Famed is thy beauty majesty, but hold, a lovely maid I see. Rags cannot hide her gentle grace.",
    "scary.wav"
)
print(time.time())
