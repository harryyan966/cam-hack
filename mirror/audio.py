from TTS.api import TTS
import sys
import io
import warnings
import subprocess

warnings.filterwarnings("ignore")

class MirrorAudio():
    def __init__(self, speaker_wav):
        print("Initializing Text to Speech Model...")

        sys.stdout = io.StringIO()
        with warnings.catch_warnings():
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            # self.tts = TTS("tts_models/en/ljspeech/glow-tts")
            # self.tts = TTS('tts_models/en/ljspeech/vits--neon')
        sys.stdout = sys.__stdout__

        print("Text to Speech Model Initialized ✅")
        self.speaker_wav = speaker_wav

    def text_to_audio(self, text, aud_at='materialized/audio.wav'):
        self.gen_audio(text, aud_at)
        
    def gen_audio(self, text, aud_at):
        print("Generating Audio...")
        self.tts.tts_to_file(
            text=text,
            file_path=f'mirror/{aud_at}',
            speaker_wav=self.speaker_wav,
            language="en",
            split_sentences=True,
        )
        print(f"Audio generated at mirror/{aud_at} ✅")
