import torch
from TTS.api import TTS
import time

# tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
tts.tts_with_vc_to_file(
    '''No face is ugly, just a dream.  
Heart's essence shines, let kindness reign.  
In love's mirror, none are in vain.''',
    speaker_wav="scary.wav",
    # language='en',
    file_path="output.wav"
)

['tts_models/multilingual/multi-dataset/xtts_v2',
 'tts_models/multilingual/multi-dataset/xtts_v1.1',
 'tts_models/multilingual/multi-dataset/your_tts',
 'tts_models/multilingual/multi-dataset/bark',
 'tts_models/bg/cv/vits',
 'tts_models/cs/cv/vits',
 'tts_models/da/cv/vits',
 'tts_models/et/cv/vits',
 'tts_models/ga/cv/vits',
 'tts_models/en/ek1/tacotron2',
 'tts_models/en/ljspeech/tacotron2-DDC',
 'tts_models/en/ljspeech/tacotron2-DDC_ph',
 'tts_models/en/ljspeech/glow-tts', 
 'tts_models/en/ljspeech/speedy-speech', 
 'tts_models/en/ljspeech/tacotron2-DCA', 
 'tts_models/en/ljspeech/vits', 
 'tts_models/en/ljspeech/vits--neon', 
 'tts_models/en/ljspeech/fast_pitch', 
 'tts_models/en/ljspeech/overflow', 
 'tts_models/en/ljspeech/neural_hmm', 
 'tts_models/en/vctk/vits', 
 'tts_models/en/vctk/fast_pitch', 
 'tts_models/en/sam/tacotron-DDC', 
 'tts_models/en/blizzard2013/capacitron-t2-c50', 
 'tts_models/en/blizzard2013/capacitron-t2-c150_v2', 
 'tts_models/en/multi-dataset/tortoise-v2', 
 'tts_models/en/jenny/jenny', 
 'tts_models/es/mai/tacotron2-DDC', 
 'tts_models/es/css10/vits', 
 'tts_models/fr/mai/tacotron2-DDC', 'tts_models/fr/css10/vits', 'tts_models/uk/mai/glow-tts', 'tts_models/uk/mai/vits', 'tts_models/zh-CN/baker/tacotron2-DDC-GST', 'tts_models/nl/mai/tacotron2-DDC', 'tts_models/nl/css10/vits', 'tts_models/de/thorsten/tacotron2-DCA', 'tts_models/de/thorsten/vits', 'tts_models/de/thorsten/tacotron2-DDC', 'tts_models/de/css10/vits-neon', 'tts_models/ja/kokoro/tacotron2-DDC',
 'tts_models/tr/common-voice/glow-tts', 
 'tts_models/it/mai_female/glow-tts', 
 'tts_models/it/mai_female/vits', 
 'tts_models/it/mai_male/glow-tts', 'tts_models/it/mai_male/vits', 'tts_models/ewe/openbible/vits', 'tts_models/hau/openbible/vits', 'tts_models/lin/openbible/vits', 'tts_models/tw_akuapem/openbible/vits', 'tts_models/tw_asante/openbible/vits', 'tts_models/yor/openbible/vits', 'tts_models/hu/css10/vits', 'tts_models/el/cv/vits', 'tts_models/fi/css10/vits', 'tts_models/hr/cv/vits', 'tts_models/lt/cv/vits', 'tts_models/lv/cv/vits', 'tts_models/mt/cv/vits', 'tts_models/pl/mai_female/vits', 'tts_models/pt/cv/vits', 'tts_models/ro/cv/vits', 'tts_models/sk/cv/vits', 'tts_models/sl/cv/vits', 'tts_models/sv/cv/vits', 'tts_models/ca/custom/vits', 'tts_models/fa/custom/glow-tts', 'tts_models/bn/custom/vits-male', 'tts_models/bn/custom/vits-female', 'tts_models/be/common-voice/glow-tts', 'vocoder_models/universal/libri-tts/wavegrad', 'vocoder_models/universal/libri-tts/fullband-melgan', 'vocoder_models/en/ek1/wavegrad', 'vocoder_models/en/ljspeech/multiband-melgan', 'vocoder_models/en/ljspeech/hifigan_v2', 'vocoder_models/en/ljspeech/univnet', 'vocoder_models/en/blizzard2013/hifigan_v2', 'vocoder_models/en/vctk/hifigan_v2', 'vocoder_models/en/sam/hifigan_v2', 'vocoder_models/nl/mai/parallel-wavegan', 'vocoder_models/de/thorsten/wavegrad', 'vocoder_models/de/thorsten/fullband-melgan', 'vocoder_models/de/thorsten/hifigan_v1', 'vocoder_models/ja/kokoro/hifigan_v1', 'vocoder_models/uk/mai/multiband-melgan', 'vocoder_models/tr/common-voice/hifigan', 'vocoder_models/be/common-voice/hifigan', 'voice_conversion_models/multilingual/vctk/freevc24']
