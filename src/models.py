import json
import os
import subprocess
import threading

# Everything must run locally: forbid Hugging Face hub traffic.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path

import numpy as np
import pyaudio
import torch
from PyQt6.QtCore import QObject, pyqtSignal
from ultralytics import YOLO

from silero_vad import load_silero_vad, VADIterator

import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor

_HERE = Path(__file__).resolve().parent

# onnxruntime-gpu needs the CUDA 13 libraries shipped by the venv's nvidia-* wheels.
# Torch loads most of them, but e.g. libcurand is only pulled in by this call.
if "CUDAExecutionProvider" in ort.get_available_providers():
    ort.preload_dlls()

KEV_WORKER = _HERE / "kev_worker.py"


class Detector:
    """Loads an ONNX model. Two separate entry points: confidence & annotated frame."""

    def __init__(self, onnx_path=None, conf=0.25):
        self.model = YOLO(str(onnx_path or _HERE / "yolo26n.onnx"))
        self.conf = conf
        self._last_result = None       # cache so we don't run inference twice per frame

    def detect(self, frame):
        """Runs inference once and caches it for both getters."""
        self._last_result = self.model.predict(frame, conf=self.conf, verbose=False)[0]

    def get_phone_confidence(self):
        """Extracts highest confidence for class 67 only, or None."""
        if self._last_result is None:
            return None
        phone_conf = None
        for cls_id, conf in zip(self._last_result.boxes.cls.tolist(),
                                self._last_result.boxes.conf.tolist()):
            if int(cls_id) == 67 and (phone_conf is None or conf > phone_conf):
                phone_conf = conf
        return phone_conf

    def get_annotated_frame(self):
        """Returns the BGR frame with YOLO boxes drawn."""
        return self._last_result.plot() if self._last_result is not None else None


SAMPLE_RATE = 16000
CHUNK = 512           # samples per frame; silero requires exactly 512 @ 16 kHz
MAX_AUDIO_SECONDS = 30          # Whisper consumes 30 s windows
TARGET_PEAK = 0.95              # per-utterance normalization target


def _pick_provider(preferred=None):
    available = ort.get_available_providers()
    candidates = [preferred] if preferred else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    for provider in candidates:
        if provider in available:
            return provider
    return available[0]


class AudioPipeline(QObject):
    text_ready = pyqtSignal(str, str)     # (translation, kev label), emitted from worker thread

    def __init__(self, model_dir=None, provider=None):
        super().__init__()
        model_dir = Path(model_dir) if model_dir else _HERE / "whisper-base"

        self._vad_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vad_model = load_silero_vad().to(self._vad_device)

        # Whisper runs through ONNX Runtime instead of faster-whisper/ctranslate2.
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3      # errors only; hide EP fallback noise
        self._whisper = ORTModelForSpeechSeq2Seq.from_pretrained(
            str(model_dir),
            subfolder="onnx",
            export=False,
            local_files_only=True,                  # model is shipped in ./whisper-base
            provider=_pick_provider(provider),
            session_options=session_options,
            use_io_binding=False,
        )
        # Optimum normally keeps outputs on the provider's device (e.g. cuda), but
        # transformers builds Whisper's logits processors on the input (CPU) device.
        # Keeping the Optimum tensor boundary on CPU avoids that device mismatch;
        # the ONNX sessions still execute on CUDA.
        for part in self._whisper.parts:
            part._device = torch.device("cpu")
        self._processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)

        # kev-0.5b runs in its own interpreter (src/kev_worker.py over pipes, no server).
        self._kev_proc = None
        self._running = False
        self._thread = None

    def _silero(self):
        vad = VADIterator(
            self._vad_model,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=100,   # snappier end-of-speech
        )
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        buffer, in_speech = [], False
        try:
            while self._running:
                raw = stream.read(CHUNK, exception_on_overflow=False)
                int16 = np.frombuffer(raw, dtype=np.int16)
                float32 = int16.astype(np.float32) / 32768.0
                evt = vad(torch.from_numpy(float32).to(self._vad_device), return_seconds=True)

                if evt:
                    if "start" in evt:
                        in_speech = True
                    elif "end" in evt:
                        in_speech = False
                        if buffer:
                            yield np.concatenate(buffer)
                            buffer = []

                if in_speech:
                    buffer.append(int16)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        """Auto-gain + peak limiter: scale the utterance to TARGET_PEAK, never clip."""
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < 1e-4:      # effectively silence
            return audio
        gain = TARGET_PEAK / peak
        return np.clip(audio * gain, -1.0, 1.0)

    def _detect_language(self, features):
        """Whisper language detection: one decoder pass, argmax over lang tokens."""
        lang_id = int(self._whisper.detect_language(features)[0])
        return self._processor.tokenizer.decode([lang_id], skip_special_tokens=False)

    def _whisper_transcribe(self, audio_int16):
        """Detect the source language, then translate the utterance to English."""
        audio = self._normalize(audio_int16.astype(np.float32) / 32768.0)
        audio = audio[: SAMPLE_RATE * MAX_AUDIO_SECONDS]

        features = self._processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features
        language = self._detect_language(features)          # e.g. '<|uk|>'

        generated = self._whisper.generate(
            features,
            language=language,
            task="translate",      # whisper translates back to en tokens
            num_beams=1,
            max_new_tokens=224,
        )
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return language, text

    def _classify(self, text):
        """Ask the local kev-0.5b model whether the utterance is productive, unproductive or neutral."""
        if self._kev_proc is None or self._kev_proc.poll() is not None:
            # kev needs its own interpreter (Python 3.13, torch 2.8, transformers 5.x);
            # importing kev_use from anything else execv's the process away.
            kev_python = _HERE / "kev-source" / ".venv" / "bin" / "python"
            self._kev_proc = subprocess.Popen(
                [str(kev_python), str(KEV_WORKER)],
                cwd=_HERE,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,             # line-buffered both ways
            )
        try:
            self._kev_proc.stdin.write(json.dumps({"content": text}) + "\n")
            self._kev_proc.stdin.flush()
            reply = json.loads(self._kev_proc.stdout.readline())
        except Exception:
            self._kev_proc.kill()      # worker wedged or died; respawn next time
            self._kev_proc = None
            return ""                  # audio/transcription still shown, just unlabelled
        return str(reply.get("label", ""))

    def _run(self):
        for speech in self._silero():
            if not self._running:
                break
            _language, translation = self._whisper_transcribe(speech)
            if not translation:
                continue
            label = self._classify(translation)
            self.text_ready.emit(translation, label)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._kev_proc is not None:
            try:
                self._kev_proc.terminate()
            except Exception:
                pass
            self._kev_proc = None
