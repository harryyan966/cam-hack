import librosa
import librosa.filters
import numpy as np
# import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import math
from .Wav2Lip.models import Wav2Lip
import os
import cv2
from time import time
from tqdm import tqdm
import torch
from batch_face import RetinaFace
import subprocess


class HParams:
	def __init__(self, **kwargs):
		self.data = {}

		for key, value in kwargs.items():
			self.data[key] = value

	def __getattr__(self, key):
		if key not in self.data:
			raise AttributeError("'HParams' object has no attribute %s" % key)
		return self.data[key]

	def set_hparam(self, key, value):
		self.data[key] = value

class Args:
    def __init__(self):
        self.checkpoint_path = "mirror/Wav2Lip/checkpoints/wav2lip.pth"
        self.face = 'mirror/face.jpg'
        self.audio = 'mirror/materialized/audio.wav'
        self.outfile = 'mirror/materialized/video.mp4'
        self.static = True
        self.fps = 20
        self.pads = [0, 10, 0, 0]
        self.wav2lip_batch_size = 128
        self.resize_factor = 1
        self.out_height = 480
        self.crop = [0, -1, 0, -1]
        self.box = [-1, -1, -1, -1]
        self.rotate = False
        self.nosmooth = True

args = Args()
args.img_size = 96


class MirrorVideo():
    face_batch_size = 64 * 8
    _mel_basis = None

    hp = HParams(
        num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
        #  network
        rescale=True,  # Whether to rescale audio prior to preprocessing
        rescaling_max=0.9,  # Rescaling value

        # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
        # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
        # Does not work if n_ffit is not multiple of hop_size!!
        use_lws=False,

        n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
        hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
        win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
        sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)

        frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)

        # Mel and Linear spectrograms normalization/scaling and clipping
        signal_normalization=True,
        # Whether to normalize mel spectrograms to some predefined range (following below parameters)
        allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
        symmetric_mels=True,
        # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2,
        # faster and cleaner convergence)
        max_abs_value=4.,
        # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not
        # be too big to avoid gradient explosion,
        # not too small for fast convergence)
        # Contribution by @begeekmyfriend
        # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude
        # levels. Also allows for better G&L phase reconstruction)
        preemphasize=True,  # whether to apply filter
        preemphasis=0.97,  # filter coefficient.

        # Limits
        min_level_db=-100,
        ref_level_db=20,
        fmin=55,
        # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
        # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
        fmax=7600,  # To be increased/reduced depending on data.

        ###################### Our training parameters #################################
        img_size=96,
        fps=25,

        batch_size=16,
        initial_learning_rate=1e-4,
        nepochs=200000000000000000,  ### ctrl + c, stop whenever eval loss is consistently greater than train loss for ~10 epochs
        num_workers=16,
        checkpoint_interval=3000,
        eval_interval=3000,
        save_optimizer_state=True,

        syncnet_wt=0.0, # is initially zero, will be set automatically to 0.03 later. Leads to faster convergence.
        syncnet_batch_size=64,
        syncnet_lr=1e-4,
        syncnet_eval_interval=10000,
        syncnet_checkpoint_interval=10000,

        disc_wt=0.07,
        disc_initial_learning_rate=1e-4,
    )


    def __init__(self):
        self.device = 'cpu'

        # load model
        print("Loading Wav2Lip Model...")

        model = Wav2Lip()
        checkpoint = torch.load(args.checkpoint_path,
                                map_location=lambda storage, _: storage)

        s = checkpoint['state_dict']
        s = {k.replace('module.', ''): v for k, v in s.items()}

        model.load_state_dict(s)
        model = model.to(self.device)

        self.model = model.eval()   ######

        self.detector = RetinaFace(model_path="mirror/Wav2Lip/checkpoints/mobilenet.pth", network="mobilenet")

        self.detector_model = self.detector.model    ######

        print("Models Loaded ✅")

    
    # IDK
    def get_image_list(self, data_root, split):
        filelist = []
        with open(f"filelist/{split}") as f:
            for line in f:
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                filelist.append(os.path.join(data_root, line))
    
        return filelist


    def load_wav(self, path, sr):
        return librosa.core.load(path, sr=sr)[0]

    def save_wav(self, wav, path, sr):
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, sr, wav.astype(np.int16))
    
    def save_wavenet_wav(self, wav, path, sr):
        librosa.output.write_wav(path, wav, sr=sr)
    
    def preemphasis(self, wav, k, preemphasize=True):
        if preemphasize:
            return signal.lfilter([1, -k], [1], wav)
        return wav

    def inv_preemphasis(self, wav, k, inv_preemphasize=True):
        if inv_preemphasize:
            return signal.lfilter([1], [1, -k], wav)
        return wav

    def get_hop_size(self):
        hop_size = self.hp.hop_size
        if hop_size is None:
            assert self.hp.frame_shift_ms is not None
            hop_size = int(self.hp.frame_shift_ms / 1000 * self.hp.sample_rate)
        return hop_size
    
    def linearspectrogram(self, wav):
        D = self._stft(self.preemphasis(wav, self.hp.preemphasis, self.hp.preemphasize))
        S = self._amp_to_db(np.abs(D)) - self.hp.ref_level_db
    
        if self.hp.signal_normalization:
            return self._normalize(S)
        return S
    
    def melspectrogram(self, wav):
        D = self._stft(self.preemphasis(wav, self.hp.preemphasis, self.hp.preemphasize))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.hp.ref_level_db

        if self.hp.signal_normalization:
            return self._normalize(S)
        return S

    def _lws_processor(self):
        import lws
        return lws.lws(self.hp.n_fft, self.get_hop_size(), fftsize=self.hp.win_size, mode="speech")

    def _stft(self, y):
        if self.hp.use_lws:
            return self._lws_processor(self.hp).stft(y).T
        else:
            return librosa.stft(y=y, n_fft=self.hp.n_fft, hop_length=self.get_hop_size(), win_length=self.hp.win_size)


    def audio_to_video(self):
        mel_step_size = 16

        if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            args.static = True
        
        if not os.path.isfile(args.face):
            raise ValueError("face not passed in ❌")
        
        elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(args.face)]
            fps = args.fps
        
        else:
            video_stream = cv2.VideoCapture(args.face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)

            print("Reading video frames...")

            full_frames = []

            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break

                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(frame, (int(args.out_height * aspect_ratio), args.out_height))
                # if args.resize_factor > 1:
                #     frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

                if args.rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = args.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

                full_frames.append(frame)
        
        print(f"Number of frames available for inference: {len(full_frames)}")

        
        if not args.audio.endswith('.wav'):
            print('Extracting raw audio...')
            # command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')
            # subprocess.call(command, shell=True)
            subprocess.check_call([
                "ffmpeg", "-y",
                "-i", args.audio,
                "mirror/Wav2Lip/temp/temp.wav",
            ])
            args.audio = 'mirror/Wav2Lip/temp/temp.wav'

        wav = self.load_wav(args.audio, 16000)
        mel = self.melspectrogram(wav)
        print(mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80./fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[:len(mel_chunks)]

        batch_size = args.wav2lip_batch_size
        gen = self.datagen(full_frames.copy(), mel_chunks)

        s = time()

        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter('mirror/Wav2Lip/temp/result.avi',
                                        cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)
                

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()

        print("wav2lip prediction time:", time() - s)

        subprocess.check_call([
            "ffmpeg", "-y",
            # "-vsync", "0", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-i", "mirror/Wav2Lip/temp/result.avi",
            "-i", args.audio,
            # "-c:v", "h264_nvenc",
            args.outfile,
        ])

    def _amp_to_db(self, x):
        min_level = np.exp(self.hp.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))
    
    def _linear_to_mel(self, spectogram):
        if self._mel_basis is None:
            self._mel_basis = self._build_mel_basis()
        return np.dot(self._mel_basis, spectogram)

    def _build_mel_basis(self):
        assert self.hp.fmax <= self.hp.sample_rate // 2
        return librosa.filters.mel(sr=self.hp.sample_rate, n_fft=self.hp.n_fft, n_mels=self.hp.num_mels,
                               fmin=self.hp.fmin, fmax=self.hp.fmax)
    
    def _normalize(self, S):
        if self.hp.allow_clipping_in_normalization:
            if self.hp.symmetric_mels:
                return np.clip((2 * self.hp.max_abs_value) * ((S - self.hp.min_level_db) / (-self.hp.min_level_db)) - self.hp.max_abs_value,
                            -self.hp.max_abs_value, self.hp.max_abs_value)
            else:
                return np.clip(self.hp.max_abs_value * ((S - self.hp.min_level_db) / (-self.hp.min_level_db)), 0, self.hp.max_abs_value)

        assert S.max() <= 0 and S.min() - self.hp.min_level_db >= 0
        if self.hp.symmetric_mels:
            return (2 * self.hp.max_abs_value) * ((S - self.hp.min_level_db) / (-self.hp.min_level_db)) - self.hp.max_abs_value
        else:
            return self.hp.max_abs_value * ((S - self.hp.min_level_db) / (-self.hp.min_level_db))

    

    def datagen(self, frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if args.box[0] == -1:
            if not args.static:
                face_det_results = self.face_detect(frames) # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = args.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if args.static else i%len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (args.img_size, args.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= args.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, args.img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch


    def face_detect(self, images):
        results = []
        pady1, pady2, padx1, padx2 = args.pads

        s = time()

        for image, rect in zip(images, self.face_rect(images)):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

        print('face detect time:', time() - s)

        boxes = np.array(results)
        if not args.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        return results

    def face_rect(self, images):  # TODO: CONSTANT FACE
        num_batches = math.ceil(len(images) / self.face_batch_size)
        prev_ret = None
        for i in range(num_batches):
            batch = images[i * self.face_batch_size: (i + 1) * self.face_batch_size]
            all_faces = self.detector(batch)  # return faces list of all images
            for faces in all_faces:
                if faces:
                    box, landmarks, score = faces[0]
                    prev_ret = tuple(map(int, box))
                yield prev_ret

        
    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes