import logging
import asyncio
import re
import numpy as np
from typing import AsyncGenerator, Optional, Tuple
from transformers import AutoModel

log = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_SAMPLE_WIDTH = 2  # 16-bit PCM
DEFAULT_CHANNELS = 1

MODEL_ID = "TeraSpace/TeraTTSv2"


def trim_eos_tail(
    audio_float32: np.ndarray, 
    sample_rate: int = 44100, 
    trim_ms: float = 30.0, 
    fade_ms: float = 5.0
) -> np.ndarray:
    """
    Безусловно отрезает последние 40 мс аудио от TeraTTS (где гарантированно 
    сидит краевой артефакт вокодера) и делает 5 мс затухание на новом краю.
    """
    trim_samples = int(sample_rate * (trim_ms / 1000.0))
    if len(audio_float32) <= trim_samples:
        return audio_float32

    # Отрезаем грязные последние 40 мс
    trimmed = audio_float32[:-trim_samples].copy()

    # Применяем микро-затухание 5 мс к новому краю, чтобы переход был идеальным
    fade_samples = int(sample_rate * (fade_ms / 1000.0))
    if len(trimmed) >= fade_samples:
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        trimmed[-fade_samples:] *= fade_out

    return trimmed


class SpeechTTS:
    """
    Обертка над TeraTTSv2 ONNX с поддержкой нативного потокового синтеза,
    нормализации громкости (dBFS) и очистки от артефактов.
    """

    AVAILABLE_VOICES = [
        "ru_f1", "ru_m5", "ru_f2", "ru_m1",
        "eng_f3", "eng_f4_whisper", "eng_f5", 
        "eng_m2_whisper", "eng_m3", "eng_m4"
    ]

    def __init__(
        self,
        threads: Optional[int] = None,
        ruaccent_mode: str = "dictionary",
        diffusion_model: str = "distilled",
        chunk_frames: int = 16,
        target_db: Optional[float] = None,
        data_dir: str = "."
    ) -> None:
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.sample_width = DEFAULT_SAMPLE_WIDTH
        self.channels = DEFAULT_CHANNELS
        self.chunk_frames = chunk_frames
        self.target_db = target_db
        self._lock = asyncio.Lock()

        log.info(f"Loading TeraTTSv2 model ({MODEL_ID}) ...")
        
        try:
            self.model = AutoModel.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                provider="CPUExecutionProvider",
                threads=threads,
                ruaccent_mode=ruaccent_mode,
                diffusion_model=diffusion_model,
                cache_dir=data_dir if data_dir != "." else None
            )
            log.info("TeraTTSv2 ONNX model loaded successfully.")
            if self.target_db is not None:
                log.info(f"Volume normalization enabled: Target = {self.target_db} dBFS")
            else:
                log.info("Volume normalization is disabled.")
        except Exception as e:
            log.critical(f"Failed to load TeraTTSv2 model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize TeraTTSv2: {e}") from e

    def _ensure_language_tags(self, text: str) -> str:
        """
        Умное авто-тегирование:
        - Если есть явные теги <ru> или <en> -> оставляем.
        - Если нет кириллицы, но есть латиница -> <en>...</en>.
        - Иначе (кириллица или смеси) -> <ru>...</ru>.
        """
        text = text.strip()
        if re.search(r'<(ru|en)>.*?</\1>', text, flags=re.DOTALL | re.IGNORECASE):
            return text

        has_cyrillic = bool(re.search(r'[а-яА-ЯёЁ]', text))
        has_latin = bool(re.search(r'[a-zA-Z]', text))

        if not has_cyrillic and has_latin:
            return f"<en>{text}</en>"

        return f"<ru>{text}</ru>"

    def process_chunk_gain(
        self, 
        chunk: np.ndarray, 
        session_scale: Optional[float]
    ) -> Tuple[np.ndarray, float]:
        """
        Потоковая нормализация громкости чанка с памятью сессии (EMA)
        и защитой от усиления пауз.
        """
        if self.target_db is None or len(chunk) == 0:
            return chunk, session_scale or 1.0

        abs_chunk = np.abs(chunk)
        perc_peak = float(np.percentile(abs_chunk, 99.95))

        # Защита от тишины (< -40 dB): не пересчитываем гейн для тихих пауз
        if perc_peak < 0.01:
            scale_to_apply = session_scale if session_scale is not None else 1.0
            normalized_chunk = np.clip(chunk * scale_to_apply, -1.0, 1.0)
            return normalized_chunk, scale_to_apply

        effective_peak = max(perc_peak, 1e-5)
        target_linear = 10.0 ** (self.target_db / 20.0)  # e.g. -3.0 dBFS -> ~0.707
        raw_target_scale = target_linear / effective_peak

        if session_scale is None:
            final_scale = raw_target_scale
        else:
            # Ограничиваем максимальное отклонение (+/- 3 дБ)
            max_dev = 1.414
            clamped_target = np.clip(
                raw_target_scale,
                session_scale / max_dev,
                session_scale * max_dev
            )
            # Экспоненциальное сглаживание (EMA): 70% истории + 30% нового значения
            final_scale = 0.7 * session_scale + 0.3 * clamped_target

        normalized_chunk = np.clip(chunk * final_scale, -1.0, 1.0)
        return normalized_chunk, final_scale

    def _float32_to_pcm16(self, audio_float32: np.ndarray) -> bytes:
        """Преобразует Float32 NumPy массив в Int16 PCM Little-Endian байты."""
        pcm16 = np.rint(np.clip(audio_float32, -1.0, 1.0) * 32767.0).astype("<i2")
        return pcm16.tobytes()

    async def synthesize_stream(
        self, 
        text: str, 
        speaker_name: str, 
        duration_scale: float = 1.0,
        session_scale: Optional[float] = None
    ) -> AsyncGenerator[Tuple[bytes, float], None]:
        if not text.strip():
            return

        # Замена юникодных тире на обычный дефис во избежание RuntimeWarning
        text = text.replace('—', '-').replace('–', '-')
        tagged_text = self._ensure_language_tags(text)
        
        if speaker_name.startswith("eng_") and duration_scale == 1.0:
            duration_scale = 0.8

        log.debug(f"Streaming TTS. Voice: {speaker_name}, Scale: {duration_scale}, Text: [{tagged_text[:60]}...]")

        def _get_stream_generator():
            return self.model.generate_speech_stream(
                tagged_text,
                voice=speaker_name,
                duration_scale=duration_scale,
                chunk_frames=self.chunk_frames
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

                # 1. Отрезаем грязные 40 мс в конце фразы (убираем всплеск вокодера)
                clean_audio = trim_eos_tail(sentence_audio, sample_rate=self.sample_rate)

                # 2. Нарезаем обратно на чанки, привязанные к chunk_frames TeraTTS (49 152 сэмпла)
                chunk_samples = self.chunk_frames * 3072
                current_scale = session_scale

                for i in range(0, len(clean_audio), chunk_samples):
                    chunk = clean_audio[i : i + chunk_samples]
                    
                    # 3. Выравниваем громкость
                    norm_chunk, current_scale = self.process_chunk_gain(
                        chunk, current_scale
                    )
                    
                    pcm_bytes = self._float32_to_pcm16(norm_chunk)
                    if pcm_bytes:
                        yield pcm_bytes, current_scale

            except Exception as e:
                log.error(f"TeraTTSv2 streaming failed: {e}", exc_info=True)