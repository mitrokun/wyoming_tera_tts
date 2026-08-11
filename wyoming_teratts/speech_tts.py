"""TeraTTSv2 ONNX Engine wrapper for streaming speech synthesis."""

import asyncio
import logging
import re
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np
from transformers import AutoConfig, AutoModel

log = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_SAMPLE_WIDTH = 2  # 16-bit PCM
DEFAULT_CHANNELS = 1

MODEL_ID = "TeraSpace/TeraTTSv2"


def trim_eos_tail(
    audio_float32: np.ndarray,
    sample_rate: int = 44100,
    trim_ms: float = 30.0,
    fade_ms: float = 5.0,
) -> np.ndarray:
    """Trim vocoder end-of-sentence tail artifact and apply micro fade-out."""
    trim_samples = int(sample_rate * (trim_ms / 1000.0))
    if len(audio_float32) <= trim_samples:
        return audio_float32

    trimmed = audio_float32[:-trim_samples].copy()

    fade_samples = int(sample_rate * (fade_ms / 1000.0))
    if len(trimmed) >= fade_samples:
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        trimmed[-fade_samples:] *= fade_out

    return trimmed


class SpeechTTS:
    """Wrapper around TeraTTSv2 ONNX model supporting native streaming synthesis,

    volume normalization (dBFS), and vocoder artifact trimming.
    """

    def __init__(
        self,
        threads: Optional[int] = None,
        ruaccent_mode: str = "dictionary",
        diffusion_model: str = "distilled",
        chunk_frames: int = 16,
        target_db: Optional[float] = None,
        data_dir: str = ".",
    ) -> None:
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.sample_width = DEFAULT_SAMPLE_WIDTH
        self.channels = DEFAULT_CHANNELS
        self.chunk_frames = chunk_frames
        self.target_db = target_db
        self._lock = asyncio.Lock()

        log.info("Loading TeraTTSv2 model (%s)...", MODEL_ID)

        try:
            self.model = AutoModel.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                provider="CPUExecutionProvider",
                threads=threads,
                ruaccent_mode=ruaccent_mode,
                diffusion_model=diffusion_model,
                cache_dir=data_dir if data_dir != "." else None,
            )
            
            # Динамически считываем список голосов из config.json загруженной модели
            self.voices: List[str] = getattr(self.model.config, "voices", [])
            log.info("Loaded %d voices from model config: %s", len(self.voices), self.voices)

            log.info("TeraTTSv2 ONNX model loaded successfully.")
            if self.target_db is not None:
                log.info("Volume normalization enabled: Target = %.1f dBFS", self.target_db)
            else:
                log.info("Volume normalization disabled.")
        except Exception as e:
            log.critical("Failed to load TeraTTSv2 model: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to initialize TeraTTSv2: {e}") from e

    @classmethod
    def get_available_voices(cls, data_dir: str = ".") -> List[str]:
        """Быстрое чтение списка голосов из config.json без загрузки тяжелых весов модели."""
        try:
            config = AutoConfig.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                cache_dir=data_dir if data_dir != "." else None,
            )
            return getattr(config, "voices", [])
        except Exception as e:
            log.warning("Failed to fetch voices from config: %s", e)
            return []

    @staticmethod
    def _split_digits_letters(text: str) -> str:
        """Insert spaces between letters and digits to help TTS normalizer."""
        text = re.sub(r'([A-Za-zА-Яа-я])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([A-Za-zА-Яа-я])', r'\1 \2', text)
        text = re.sub(r'(\d)([^\w\s])', r'\1 \2', text)
        text = re.sub(r'([^\w\s])(\d)', r'\1 \2', text)

        return text

    @staticmethod
    def _clean_and_normalize_language_tags(text: str) -> str:
        """
        Remove all XML/HTML tags except <ru>, </ru>, <en>, </en>.
        Then remove any stray '<' and '>' characters.
        Auto-wrap with language tags if none present.
        """
        def clean_tags(match):
            tag = match.group(0)
            if tag.lower() in ('<ru>', '</ru>', '<en>', '</en>'):
                return tag
            return ''

        text = re.sub(r'<[^>]+>', clean_tags, text)
        text = text.replace('<', '').replace('>', '')

        has_ru = bool(re.search(r'<ru>.*?</ru>', text, flags=re.DOTALL | re.IGNORECASE))
        has_en = bool(re.search(r'<en>.*?</en>', text, flags=re.DOTALL | re.IGNORECASE))

        if has_ru or has_en:
            return text

        has_cyrillic = bool(re.search(r'[а-яА-ЯёЁ]', text))
        has_latin = bool(re.search(r'[a-zA-Z]', text))

        if not has_cyrillic and has_latin:
            return f"<en>{text}</en>"
        else:
            return f"<ru>{text}</ru>"

    def normalize_sentence_gain(
        self,
        audio_float32: np.ndarray,
        session_scale: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Normalize entire sentence volume to target_db smoothly."""
        if self.target_db is None or len(audio_float32) == 0:
            return audio_float32, session_scale or 1.0

        abs_audio = np.abs(audio_float32)
        perc_peak = float(np.percentile(abs_audio, 99.95))

        # Do not amplify silent sections (< -40 dB)
        if perc_peak < 0.01:
            scale_to_apply = session_scale if session_scale is not None else 1.0
            return np.clip(audio_float32 * scale_to_apply, -1.0, 1.0), scale_to_apply

        effective_peak = max(perc_peak, 1e-5)
        target_linear = 10.0 ** (self.target_db / 20.0)
        raw_target_scale = target_linear / effective_peak

        if session_scale is None:
            final_scale = raw_target_scale
        else:
            # Плавная подстройка между предложениями без скачков внутри речи
            max_dev = 1.414
            clamped_target = np.clip(
                raw_target_scale,
                session_scale / max_dev,
                session_scale * max_dev,
            )
            final_scale = 0.7 * session_scale + 0.3 * clamped_target

        normalized_audio = np.clip(audio_float32 * final_scale, -1.0, 1.0)
        return normalized_audio, final_scale

    def _float32_to_pcm16(self, audio_float32: np.ndarray) -> bytes:
        """Convert float32 numpy array to 16-bit PCM little-endian bytes."""
        pcm16 = np.rint(np.clip(audio_float32, -1.0, 1.0) * 32767.0).astype("<i2")
        return pcm16.tobytes()

    async def synthesize_stream(
        self,
        text: str,
        speaker_name: str,
        duration_scale: float = 1.0,
        session_scale: Optional[float] = None,
    ) -> AsyncGenerator[Tuple[bytes, float], None]:
        """Asynchronously stream speech chunks for the given text."""
        if not text.strip():
            return

        text = text.replace("—", "-").replace("–", "-")
        text = self._split_digits_letters(text)
        tagged_text = self._clean_and_normalize_language_tags(text)

        log.debug("[Final] %s", tagged_text)

        if speaker_name.startswith("eng_") and duration_scale == 1.0:
            duration_scale = 0.8

        def _get_stream_generator():
            return self.model.generate_speech_stream(
                tagged_text,
                voice=speaker_name,
                duration_scale=duration_scale,
                chunk_frames=self.chunk_frames,
            )

        async with self._lock:
            try:
                stream_iter = await asyncio.to_thread(_get_stream_generator)

                chunks = []
                while True:
                    chunk_float32 = await asyncio.to_thread(next, stream_iter, None)
                    if chunk_float32 is None:
                        break
                    chunks.append(chunk_float32)

                if not chunks:
                    return

                sentence_audio = np.concatenate(chunks)
                clean_audio = trim_eos_tail(sentence_audio, sample_rate=self.sample_rate)

                # Нормализация ВСЕГО предложения целиком
                norm_audio, current_scale = self.normalize_sentence_gain(
                    clean_audio, session_scale
                )

                chunk_samples = self.chunk_frames * 3072

                for i in range(0, len(norm_audio), chunk_samples):
                    chunk = norm_audio[i : i + chunk_samples]
                    pcm_bytes = self._float32_to_pcm16(chunk)
                    if pcm_bytes:
                        yield pcm_bytes, current_scale

            except Exception as e:
                log.error("TeraTTSv2 streaming failed: %s", e, exc_info=True)