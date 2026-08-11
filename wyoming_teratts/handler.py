"""Wyoming protocol event handler with text preprocessing and metrics logging."""

import asyncio
import logging
import time
from typing import Any, Optional

import regex as re
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .sentence_boundary import SentenceBoundaryDetector
from .speech_tts import SpeechTTS

log = logging.getLogger(__name__)


def preprocess_text_for_stress(
    text: str,
    accentor: Optional[Any],
) -> str:
    """
    Preprocess text by applying Silero stress to ALL words in the text.
    User manual '+' markers are preserved.
    """
    if accentor:
        try:
            processed = accentor(text)
            return processed
        except Exception as e:
            log.debug("Silero stress error: %s", e)

    return text


class SpeechEventHandler(AsyncEventHandler):
    """Wyoming event handler managing audio streaming and TTS synthesis."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: Any,
        speech_tts: SpeechTTS,
        voice_map: dict,
        default_speaker_name: str,
        accentor: Optional[Any] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.speech_tts = speech_tts
        self.voice_map = voice_map
        self.default_speaker_name = default_speaker_name
        self.accentor = accentor

        self.is_streaming: bool = False
        self.audio_started: bool = False
        self.sbd: Optional[SentenceBoundaryDetector] = None
        self._synthesize: Optional[Synthesize] = None
        self._session_scale: Optional[float] = None
        self._total_stream_audio_sec: float = 0.0

    def _get_silence_bytes(self, duration_sec: float) -> bytes:
        """Generate silent PCM16 LE audio bytes."""
        if duration_sec <= 0:
            return b""
        num_samples = int(self.speech_tts.sample_rate * duration_sec)
        return b"\x00\x00" * num_samples

    async def _send_pcm_bytes(self, pcm_bytes: bytes) -> None:
        """Split raw PCM bytes into Wyoming AudioChunks."""
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
                    channels=channels,
                ).event()
            )

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        try:
            # 1. Monolithic Text Synthesis (Legacy)
            if Synthesize.is_type(event.type):
                if self.is_streaming:
                    log.debug("Streaming is active, skipping legacy Synthesize event.")
                    return True

                self._session_scale = None
                synthesize = Synthesize.from_event(event)
                return await self._handle_legacy_synthesize(synthesize)

            # 2. Streaming Start
            if SynthesizeStart.is_type(event.type):
                if not self.cli_args.streaming:
                    log.debug("Streaming is disabled by CLI argument.")
                    return True

                self.is_streaming = True
                self.audio_started = False
                self._session_scale = None
                self._total_stream_audio_sec = 0.0
                self.sbd = SentenceBoundaryDetector(emit_break_markers=True)
                stream_start = SynthesizeStart.from_event(event)
                self._synthesize = Synthesize(text="", voice=stream_start.voice)

                voice_name = self._resolve_speaker(
                    stream_start.voice.name if stream_start.voice else None
                )
                log.debug("[STREAM START] Requested voice: %s", voice_name)
                return True

            # 3. Streaming Chunk
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

            # 4. Streaming Stop
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

                log.debug(
                    "[STREAM FINISH] Total stream audio: %.2fs",
                    self._total_stream_audio_sec,
                )

                self.is_streaming = False
                self.audio_started = False
                self._session_scale = None
                self.sbd = None
                self._synthesize = None
                return True

            return True

        except Exception as e:
            log.error("Error handling Wyoming event: %s", e, exc_info=True)
            await self.write_event(Error(text=str(e), code=e.__class__.__name__).event())
            self.is_streaming = False
            self.audio_started = False
            self._session_scale = None
            return False

    async def _ensure_audio_start(self) -> None:
        """Emit AudioStart event and initial silence if configured."""
        if not self.audio_started:
            await self.write_event(
                AudioStart(
                    rate=self.speech_tts.sample_rate,
                    width=self.speech_tts.sample_width,
                    channels=self.speech_tts.channels,
                ).event()
            )
            self.audio_started = True

            if self.cli_args.silence_before > 0:
                silence = self._get_silence_bytes(self.cli_args.silence_before)
                await self._send_pcm_bytes(silence)

    async def _send_pause(self, duration_sec: float) -> None:
        """Emit silent PCM audio chunk."""
        if self.audio_started and duration_sec > 0:
            log.debug("[PAUSE] Inserting %.2fs silence", duration_sec)
            silence = self._get_silence_bytes(duration_sec)
            await self._send_pcm_bytes(silence)

    async def _stream_sentence(self, sentence: str) -> None:
        """Synthesize sentence, emit chunks, and calculate RTFx metrics."""
        if not sentence.strip():
            return

        speaker_name = self._resolve_speaker()
        await self._ensure_audio_start()

        processed_sentence = preprocess_text_for_stress(sentence, self.accentor)

        start_time = time.perf_counter()
        total_pcm_bytes = 0

        async for pcm_chunk, updated_scale in self.speech_tts.synthesize_stream(
            text=processed_sentence,
            speaker_name=speaker_name,
            duration_scale=self.cli_args.duration_scale,
            session_scale=self._session_scale,
        ):
            self._session_scale = updated_scale
            total_pcm_bytes += len(pcm_chunk)
            await self._send_pcm_bytes(pcm_chunk)

        synth_time = time.perf_counter() - start_time
        bytes_per_sec = self.speech_tts.sample_rate * self.speech_tts.sample_width * self.speech_tts.channels
        audio_dur = total_pcm_bytes / bytes_per_sec if bytes_per_sec > 0 else 0.0
        self._total_stream_audio_sec += audio_dur
        rtfx = (audio_dur / synth_time) if synth_time > 0 else 0.0

        log.debug(
            '[DONE] Voice: %s | Text: "%s" | Audio: %.2fs | Synth: %.3fs | RTFx: %.1fx',
            speaker_name,
            sentence,
            audio_dur,
            synth_time,
            rtfx,
        )

        if self.cli_args.silence_sentence > 0:
            silence = self._get_silence_bytes(self.cli_args.silence_sentence)
            await self._send_pcm_bytes(silence)

    async def _handle_legacy_synthesize(self, synthesize: Synthesize) -> bool:
        """Handle legacy monolithic Synthesize event."""
        if not synthesize.text:
            return True

        speaker_name = self._resolve_speaker(synthesize.voice.name if synthesize.voice else None)
        text = " ".join(synthesize.text.strip().splitlines())

        await self._ensure_audio_start()

        processed_text = preprocess_text_for_stress(text, self.accentor)

        start_time = time.perf_counter()
        total_pcm_bytes = 0

        async for pcm_chunk, updated_scale in self.speech_tts.synthesize_stream(
            text=processed_text,
            speaker_name=speaker_name,
            duration_scale=self.cli_args.duration_scale,
            session_scale=self._session_scale,
        ):
            self._session_scale = updated_scale
            total_pcm_bytes += len(pcm_chunk)
            await self._send_pcm_bytes(pcm_chunk)

        synth_time = time.perf_counter() - start_time
        bytes_per_sec = self.speech_tts.sample_rate * self.speech_tts.sample_width * self.speech_tts.channels
        audio_dur = total_pcm_bytes / bytes_per_sec if bytes_per_sec > 0 else 0.0
        rtfx = (audio_dur / synth_time) if synth_time > 0 else 0.0

        log.info(
            '[DONE] Voice: %s | Text: "%s" | Audio: %.2fs | Synth: %.3fs | RTFx: %.1fx',
            speaker_name,
            text,
            audio_dur,
            synth_time,
            rtfx,
        )

        await self.write_event(AudioStop().event())
        self.audio_started = False
        self._session_scale = None
        return True

    def _resolve_speaker(self, requested_voice: Optional[str] = None) -> str:
        """Resolve speaker name from requested voice or fallback default."""
        if requested_voice and requested_voice in self.voice_map:
            return requested_voice
        if self._synthesize and self._synthesize.voice and self._synthesize.voice.name in self.voice_map:
            return self._synthesize.voice.name
        return self.default_speaker_name