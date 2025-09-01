#!/usr/bin/env python3
import argparse
import asyncio
import logging
import platform
import re
from functools import partial
from typing import Any

from pywhispercpp.model import Model
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import WhisperCppEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Name of whisper.cpp model to use (or auto)",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )
    parser.add_argument(
        "--language",
        help="Default language to set for transcription",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Size of beam during decoding (0 for greedy)",
    )
    parser.add_argument(
        "--patience",
        type=float,
        default=1.0,
        help="Patience factor for beam search (default: 1.0, see https://arxiv.org/abs/2204.05424)",
    )
    parser.add_argument(
        "--initial-prompt",
        help="Optional text to provide as a prompt for the first window",
    )
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    if not args.download_dir:
        # Download to first data dir by default
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    # Automatic configuration for ARM
    machine = platform.machine().lower()
    is_arm = ("arm" in machine) or ("aarch" in machine)
    if args.model == "auto":
        args.model = "tiny-q8_0" if is_arm else "base-q8_0"
        _LOGGER.debug("Model automatically selected: %s", args.model)

    if args.beam_size <= 0:
        args.beam_size = 1 if is_arm else 5
        _LOGGER.debug("Beam size automatically selected: %s", args.beam_size)

    # Resolve model name
    model_name = args.model
    match = re.match(
        r"^(tiny|base|small|medium|large-v[1-3]|large-v3-turbo)[.-]int8$", args.model
    )
    if match:
        model_size = match.group(1)
        model_name = f"{model_size}-q8_0"

    if args.language == "auto":
        # Whisper does not understand "auto"
        args.language = None

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="whisper.cpp",
                description="Whisper.cpp transcription",
                attribution=Attribution(
                    name="AbSadiki",
                    url="https://github.com/absadiki/pywhispercpp/",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=args.model,
                        description=args.model,
                        attribution=Attribution(
                            name="Georgi Gerganov",
                            url="https://huggingface.co/ggerganov/whisper.cpp/",
                        ),
                        installed=True,
                        languages=Model.available_languages(),  # pylint: disable=protected-access
                        version="1",
                    )
                ],
            )
        ],
    )

    # Load model
    _LOGGER.debug("Loading %s", args.model)
    whisper_model: Any = None

    whisper_model = Model(
        model_name,
        models_dir=args.download_dir,
        params_sampling_strategy=args.beam_size,
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    model_lock = asyncio.Lock()

    await server.run(
        partial(
            WhisperCppEventHandler,
            wyoming_info,
            args,
            whisper_model,
            model_lock,
            initial_prompt=args.initial_prompt,
        )
    )


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
