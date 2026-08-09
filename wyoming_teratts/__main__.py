import os
import argparse
from argparse import ArgumentParser
import asyncio
import contextlib
import logging
import sys
from functools import partial
from typing import Optional

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from .handler import SpeechEventHandler
from .speech_tts import SpeechTTS

log = logging.getLogger(__name__)

DEFAULT_SPEAKER_NAME = "ru_f1"
MODEL_LANGUAGE = "ru-RU"
DEFAULT_VOICE_VERSION = "2.0"
ATTRIBUTION_NAME = "TeraSpace"
ATTRIBUTION_URL = "https://huggingface.co/TeraSpace/TeraTTSv2"
PROGRAM_NAME = "teratts-wyoming"
PROGRAM_DESCRIPTION = "Wyoming server for TeraTTSv2 ONNX Text-to-Speech"
PROGRAM_VERSION = "2.0"


def parse_target_db(val: Optional[str]) -> Optional[float]:
    """Парсит аргумент командной строки --target-db."""
    if val is None or str(val).lower() in ("none", "off", "false", "0", "disable"):
        return None
    try:
        return float(val)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid --target-db value: '{val}'. Expected float (e.g. -3.0) or 'off'."
        )


async def main() -> None:
    parser = ArgumentParser(description="TeraTTSv2 Wyoming Server")
    
    parser.add_argument(
        "--uri", 
        default="tcp://0.0.0.0:10201", 
        help="URI сервера (по умолчанию: tcp://0.0.0.0:12001)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Включить подробный логгинг (DEBUG).",
    )
    parser.add_argument("--samples-per-chunk", type=int, default=1024)

    # Параметры движка TeraTTSv2
    parser.add_argument(
        "--default-speaker",
        type=str,
        default=DEFAULT_SPEAKER_NAME,
        help=f"Голос по умолчанию. Выбор из: {SpeechTTS.AVAILABLE_VOICES}"
    )
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Коэффициент длительности (>1.0 медленнее, <1.0 быстрее)."
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=16,
        help="Размер потокового чанка вокодера (меньше = ниже задержка)."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Число потоков CPU (по умолчанию авто: половина логических ядер)."
    )
    parser.add_argument(
        "--ruaccent-mode",
        type=str,
        default="dictionary",
        choices=["dictionary", "full"],
        help="Режим ударений: 'dictionary' (быстрый) или 'full' (нейросеть)."
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="distilled",
        choices=["distilled", "teacher"],
        help="Модель сэмплера: 'distilled' (быстрый) или 'teacher'."
    )
    parser.add_argument(
        "--target-db",
        type=parse_target_db,
        default=None,
        help="Целевой уровень громкости в dBFS (например, -3.0). По умолчанию выключен ('off').",
    )

    # Задержки тишины (Piper-style)
    parser.add_argument(
        "--silence-before",
        type=float,
        default=0.05,
        help="Задержка тишины перед началом речи (сек)."
    )
    parser.add_argument(
        "--silence-sentence",
        type=float,
        default=0.25,
        help="Пауза тишины между предложениями (сек)."
    )
    parser.add_argument(
        "--silence-paragraph",
        type=float,
        default=0.60,
        help="Пауза тишины между абзацами и диалогами (сек)."
    )

    # Потоковый режим
    parser.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="Отключить стриминг по предложениям.",
    )
    parser.set_defaults(streaming=True)

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    log.info("Starting TeraTTSv2 Wyoming Server...")

    try:
        speech_tts_instance = SpeechTTS(
            threads=args.threads,
            ruaccent_mode=args.ruaccent_mode,
            diffusion_model=args.diffusion_model,
            chunk_frames=args.chunk_frames,
            target_db=args.target_db
        )
    except Exception as e:
        log.critical(f"Failed to initialize TeraTTSv2 engine: {e}")
        sys.exit(1)

    if args.default_speaker not in SpeechTTS.AVAILABLE_VOICES:
        log.warning(
            f"Default speaker '{args.default_speaker}' not in {SpeechTTS.AVAILABLE_VOICES}. "
            f"Falling back to '{DEFAULT_SPEAKER_NAME}'."
        )
        args.default_speaker = DEFAULT_SPEAKER_NAME

    voices = []
    voice_map = {name: name for name in SpeechTTS.AVAILABLE_VOICES}

    for name in SpeechTTS.AVAILABLE_VOICES:
        voices.append(TtsVoice(
            name=name,
            description=f"TeraTTS Voice {name}",
            attribution=Attribution(name=ATTRIBUTION_NAME, url=ATTRIBUTION_URL),
            installed=True,
            version=DEFAULT_VOICE_VERSION,
            languages=[MODEL_LANGUAGE]
        ))

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name=PROGRAM_NAME,
                description=PROGRAM_DESCRIPTION,
                attribution=Attribution(
                    name=ATTRIBUTION_NAME,
                    url=ATTRIBUTION_URL
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
    )

    server = AsyncServer.from_uri(args.uri)
    log.info(f"Server is listening at {args.uri}")
    await server.run(handler_factory)


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(main())
    finally:
        log.info("Server shutting down.")