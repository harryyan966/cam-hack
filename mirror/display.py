from TTS.api import TTS
import sys
import io
import warnings
import subprocess

warnings.filterwarnings("ignore")

class MirrorDisplay():
    wd = 'mirror'

    def __init__(self, speaker_wav):
        print("Initializing Text to Speech Model...")

        sys.stdout = io.StringIO()
        with warnings.catch_warnings():
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        sys.stdout = sys.__stdout__

        print("Text to Speech Model Initialized ✅")
        self.speaker_wav = speaker_wav

    def materialize(self, text, vid_at='materialized/vid.mp4', aud_at='materialized/aud.wav'):
        self.gen_audio(text, aud_at)
        
        print("Generating Video...")
        face = f"../face.png"
        cmd = " ".join([
            "python", "inference.py",
            "--checkpoint_path", "checkpoints/wav2lip.pth",
            "--face", face,
            "--audio", f'../{aud_at}',
            "--outfile", f'../{vid_at}',
            "--pads", "0 10 0 0",
            "--resize_factor", "1",
            "--nosmooth",
            "--fps", "10"
        ])
        subprocess.run(cmd, cwd=f"{self.wd}/Wav2Lip", shell=True)
        print(f"Video generated at {vid_at} ✅")
    
    def gen_audio(self, text, aud_at):
        print("Generating Audio...")
        self.tts.tts_to_file(
            text=text,
            file_path=f'{self.wd}/{aud_at}',
            speaker_wav=self.speaker_wav,
            language="en",
            split_sentences=True,
        )
        print(f"Audio generated at {aud_at} ✅")
        

