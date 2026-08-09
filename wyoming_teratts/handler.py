import logging
import asyncio
from typing import Optional

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .speech_tts import SpeechTTS
from .sentence_boundary import SentenceBoundaryDetector

log = logging.getLogger(__name__)


class SpeechEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args,
        speech_tts: SpeechTTS,
        voice_map: dict,
        default_speaker_name: str,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.speech_tts = speech_tts
        self.voice_map = voice_map
        self.default_speaker_name = default_speaker_name

        self.is_streaming: bool = False
        self.audio_started: bool = False
        self.sbd: Optional[SentenceBoundaryDetector] = None
        self._synthesize: Optional[Synthesize] = None
        self._session_scale: Optional[float] = None

    def _get_silence_bytes(self, duration_sec: float) -> bytes:
        """Генерирует тишину в формате 16-bit PCM Little-Endian."""
        if duration_sec <= 0:
            return b""
        num_samples = int(self.speech_tts.sample_rate * duration_sec)
        return b"\x00\x00" * num_samples

    async def _send_pcm_bytes(self, pcm_bytes: bytes) -> None:
        """Нарезает байты на AudioChunk размера samples_per_chunk."""
        if not pcm_bytes:
            return

        rate = self.speech_tts.sample_rate
        width = self.speech_tts.sample_width
        channels = self.speech_tts.channels
        
        bytes_per_chunk = width * channels * self.cli_args.samples_per_chunk
        
        for i in range(0, len(pcm_bytes), bytes_per_chunk):
            chunk = pcm_bytes[i : i + bytes_per_chunk]
            await self.write_event(
                AudioChunk(
                    audio=chunk, 
                    rate=rate, 
                    width=width, 
                    channels=channels
                ).event()
            )

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        try:
            # 1. СИНТЕЗ ЦЕЛОГО ТЕКСТА (Legacy)
            if Synthesize.is_type(event.type):
                if self.is_streaming:
                    log.debug("Streaming is active, skipping legacy Synthesize event.")
                    return True
                
                self._session_scale = None
                synthesize = Synthesize.from_event(event)
                return await self._handle_legacy_synthesize(synthesize)

            # 2. НАЧАЛО ПОТОКА (Streaming Start)
            if SynthesizeStart.is_type(event.type):
                if not self.cli_args.streaming:
                    log.debug("Streaming is disabled by CLI.")
                    return True

                self.is_streaming = True
                self.audio_started = False
                self._session_scale = None  # Сброс громкости в начале потока
                self.sbd = SentenceBoundaryDetector(emit_break_markers=True)
                stream_start = SynthesizeStart.from_event(event)
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                log.debug(f"Stream started. Requested voice: {stream_start.voice}")
                return True

            # 3. ПОРЦИЯ ТЕКСТА (Streaming Chunk)
            if SynthesizeChunk.is_type(event.type):
                if not self.is_streaming:
                    return True

                assert self._synthesize is not None
                assert self.sbd is not None

                stream_chunk = SynthesizeChunk.from_event(event)
                for token in self.sbd.add_chunk(stream_chunk.text):
                    if token in ("<PARAGRAPH_BREAK>", "<DIALOGUE_BREAK>"):
                        await self._send_pause(self.cli_args.silence_paragraph)
                    else:
                        await self._stream_sentence(token)
                return True

            # 4. ЗАВЕРШЕНИЕ ПОТОКА (Streaming Stop)
            if SynthesizeStop.is_type(event.type):
                if not self.is_streaming:
                    return True

                assert self.sbd is not None

                final_text = self.sbd.finish()
                if final_text:
                    await self._stream_sentence(final_text)

                if self.audio_started:
                    await self.write_event(AudioStop().event())

                await self.write_event(SynthesizeStopped().event())
                
                self.is_streaming = False
                self.audio_started = False
                self._session_scale = None  # Сброс гейна в конце потока
                self.sbd = None
                self._synthesize = None
                log.debug("Stream finished.")
                return True

            return True

        except Exception as e:
            log.error(f"Error handling event: {e}", exc_info=True)
            await self.write_event(Error(text=str(e), code=e.__class__.__name__).event())
            self.is_streaming = False
            self.audio_started = False
            self._session_scale = None
            return False

    async def _ensure_audio_start(self):
        """Отправляет событие AudioStart и стартовую паузу (Piper-style)."""
        if not self.audio_started:
            await self.write_event(
                AudioStart(
                    rate=self.speech_tts.sample_rate,
                    width=self.speech_tts.sample_width,
                    channels=self.speech_tts.channels
                ).event()
            )
            self.audio_started = True

            if self.cli_args.silence_before > 0:
                silence = self._get_silence_bytes(self.cli_args.silence_before)
                await self._send_pcm_bytes(silence)

    async def _send_pause(self, duration_sec: float):
        """Отправляет байты тишины."""
        if self.audio_started and duration_sec > 0:
            silence = self._get_silence_bytes(duration_sec)
            await self._send_pcm_bytes(silence)

    async def _stream_sentence(self, sentence: str):
        """Синтез предложения и отправка чанков с сохранением гейна сессии."""
        if not sentence:
            return

        speaker_name = self._resolve_speaker()
        await self._ensure_audio_start()

        async for pcm_chunk, updated_scale in self.speech_tts.synthesize_stream(
            text=sentence,
            speaker_name=speaker_name,
            duration_scale=self.cli_args.duration_scale,
            session_scale=self._session_scale
        ):
            self._session_scale = updated_scale  # Обновляем накопленный гейн
            await self._send_pcm_bytes(pcm_chunk)

        if self.cli_args.silence_sentence > 0:
            silence = self._get_silence_bytes(self.cli_args.silence_sentence)
            await self._send_pcm_bytes(silence)

    async def _handle_legacy_synthesize(self, synthesize: Synthesize) -> bool:
        """Синтез монолитного блока текста."""
        if not synthesize.text:
            return True

        speaker_name = self._resolve_speaker(synthesize.voice.name if synthesize.voice else None)
        text = " ".join(synthesize.text.strip().splitlines())

        await self._ensure_audio_start()

        async for pcm_chunk, updated_scale in self.speech_tts.synthesize_stream(
            text=text,
            speaker_name=speaker_name,
            duration_scale=self.cli_args.duration_scale,
            session_scale=self._session_scale
        ):
            self._session_scale = updated_scale
            await self._send_pcm_bytes(pcm_chunk)

        await self.write_event(AudioStop().event())
        self.audio_started = False
        self._session_scale = None
        return True

    def _resolve_speaker(self, requested_voice: Optional[str] = None) -> str:
        if requested_voice and requested_voice in self.voice_map:
            return requested_voice
        if self._synthesize and self._synthesize.voice and self._synthesize.voice.name in self.voice_map:
            return self._synthesize.voice.name
        return self.default_speaker_name