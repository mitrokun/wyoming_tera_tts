"""Wyoming server entry point for TeraTTSv2 ONNX Engine."""

import argparse
from argparse import ArgumentParser
import asyncio
import contextlib
import logging
import re
import sys
from functools import partial
from typing import Any, Optional

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from .handler import SpeechEventHandler
from .speech_tts import SpeechTTS

DEFAULT_SPEAKER_NAME = "ru_f1"
MODEL_LANGUAGE = "ru-RU"
DEFAULT_VOICE_VERSION = "2.0"
ATTRIBUTION_NAME = "TeraSpace"
ATTRIBUTION_URL = "https://huggingface.co/TeraSpace/TeraTTSv2"
PROGRAM_NAME = "teratts-wyoming"
PROGRAM_DESCRIPTION = "Wyoming server for TeraTTSv2 ONNX Text-to-Speech"
PROGRAM_VERSION = "2.0"


class TeraTTSColorFormatter(logging.Formatter):
    """Custom ANSI color formatter in Piper style."""

    GRAY = "\033[90m"          # Серый цвет для всех обычных логов
    UBUNTU_GREEN = "\033[38;2;38;162;105m"
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        asctime = self.formatTime(record, "%H:%M:%S")
        record_msg = record.getMessage()

        # Если это итоговая строка синтеза [Final] — выводим всю строку зеленым
        if "[DONE]" in record_msg:
            return f"{self.UBUNTU_GREEN}{asctime} [{record.levelname}] {record.name}: {record_msg}{self.RESET}"

        # Все остальные логи выводим серым цветом
        return f"{self.GRAY}{asctime} [{record.levelname}] {record.name}: {record_msg}{self.RESET}"


log = logging.getLogger(__name__)


def parse_target_db(val: Optional[str]) -> Optional[float]:
    """Parse --target-db command line argument."""
    if val is None or str(val).lower() in ("none", "off", "false", "0", "disable"):
        return None
    try:
        return float(val)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid --target-db value: '{val}'. Expected float (e.g., -3.0) or 'off'."
        )


def load_silero_accentor() -> Optional[Any]:
    """Attempt to load Silero Stress accentor model."""
    try:
        from silero_stress import load_accentor
        accentor = load_accentor()
        log.info("Silero Stress model loaded via 'silero-stress' package.")
        return accentor
    except ImportError:
        pass

    try:
        import torch
        torch.set_num_threads(1)
        accentor = torch.hub.load(
            repo_or_dir="snakers4/silero-stress",
            model="silero_stress",
            trust_repo=True,
        )
        log.info("Silero Stress model loaded via torch.hub.")
        return accentor
    except Exception as e:
        log.warning("Failed to load Silero Stress accentor: %s", e)
        return None


async def main() -> None:
    parser = ArgumentParser(description="TeraTTSv2 Wyoming Server")

    parser.add_argument(
        "--uri",
        default="tcp://0.0.0.0:10201",
        help="Server URI (default: tcp://0.0.0.0:10201)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging.",
    )
    parser.add_argument("--samples-per-chunk", type=int, default=1024)

    # Silero Accentor Flag
    parser.add_argument(
        "--silero",
        action="store_true",
        help="Enable Silero Stress neural accentor for homograph correction.",
    )

    # TeraTTS Engine Options
    parser.add_argument(
        "--default-speaker",
        type=str,
        default=DEFAULT_SPEAKER_NAME,
        help=f"Default voice speaker (default: {DEFAULT_SPEAKER_NAME}).",
    )
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Duration scale factor (>1.0 slower, <1.0 faster).",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=16,
        help="Vocoder streaming chunk frames (smaller = lower latency).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU thread count (default: auto).",
    )
    parser.add_argument(
        "--ruaccent-mode",
        type=str,
        default="dictionary",
        choices=["dictionary", "full"],
        help="RuAccent mode: 'dictionary' (fast) or 'full' (neural).",
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="distilled",
        choices=["distilled", "teacher"],
        help="Diffusion model sampler: 'distilled' (fast) or 'teacher'.",
    )
    parser.add_argument(
        "--target-db",
        type=parse_target_db,
        default=None,
        help="Target volume level  (e.g. -6.0). Default: off.",
    )

    # Silence Delays
    parser.add_argument(
        "--silence-before",
        type=float,
        default=0.05,
        help="Silence padding before speech start (seconds).",
    )
    parser.add_argument(
        "--silence-sentence",
        type=float,
        default=0.20,
        help="Silence delay between sentences (seconds).",
    )
    parser.add_argument(
        "--silence-paragraph",
        type=float,
        default=0.60,
        help="Silence delay between paragraphs and dialogues (seconds).",
    )

    # Streaming Options
    parser.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="Disable sentence streaming mode.",
    )
    parser.set_defaults(streaming=True)

    args = parser.parse_args()

    # Configure logging with custom ANSI colors
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(TeraTTSColorFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    root_logger.addHandler(handler)

    log.info("Starting TeraTTSv2 Wyoming Server...")

    accentor = None
    if args.silero:
        log.info("Initializing Silero Stress Accentor...")
        accentor = load_silero_accentor()

    try:
        speech_tts_instance = SpeechTTS(
            threads=args.threads,
            ruaccent_mode=args.ruaccent_mode,
            diffusion_model=args.diffusion_model,
            chunk_frames=args.chunk_frames,
            target_db=args.target_db,
        )
    except Exception as e:
        log.critical("Failed to initialize TeraTTSv2 engine: %s", e)
        sys.exit(1)

    available_voices = speech_tts_instance.voices

    if args.default_speaker not in available_voices:
        fallback_speaker = getattr(
            speech_tts_instance.model.config, "default_voice", DEFAULT_SPEAKER_NAME
        )
        log.warning(
            "Default speaker '%s' not in %s. Falling back to '%s'.",
            args.default_speaker,
            available_voices,
            fallback_speaker,
        )
        args.default_speaker = fallback_speaker

    voices = []
    voice_map = {name: name for name in available_voices}

    for name in available_voices:
        voices.append(
            TtsVoice(
                name=name,
                description=name,  # Убрали "TeraTTS Voice ", теперь в названии только имя (напр. ru_f1)
                attribution=Attribution(name=ATTRIBUTION_NAME, url=ATTRIBUTION_URL),
                installed=True,
                version=DEFAULT_VOICE_VERSION,
                languages=[MODEL_LANGUAGE],
            )
        )

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name=PROGRAM_NAME,
                description=PROGRAM_DESCRIPTION,
                attribution=Attribution(
                    name=ATTRIBUTION_NAME,
                    url=ATTRIBUTION_URL,
                ),
                installed=True,
                version=PROGRAM_VERSION,
                voices=voices,
                supports_synthesize_streaming=args.streaming,
            )
        ]
    )

    handler_factory = partial(
        SpeechEventHandler,
        wyoming_info,
        args,
        speech_tts_instance,
        voice_map,
        args.default_speaker,
        accentor,
    )

    server = AsyncServer.from_uri(args.uri)
    log.info("Server is listening at %s", args.uri)
    await server.run(handler_factory)


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(main())
    finally:
        log.info("Server shutting down.")