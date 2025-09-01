"""Event handler for clients of the server."""
import argparse
import asyncio
import logging
from typing import Optional

import numpy as np
from pywhispercpp.constants import WHISPER_SAMPLE_RATE
from pywhispercpp.model import Model
from soxr import resample
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)


class WhisperCppEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: Model,
        model_lock: asyncio.Lock,
        *args,
        initial_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.initial_prompt = initial_prompt
        self._language = self.cli_args.language
        self._raw: Optional[bytearray] = None
        self._rate: Optional[int] = None
        self._width: Optional[int] = None
        self._channels: Optional[int] = None

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self._raw is None:
                self._rate = chunk.rate
                self._width = chunk.width
                self._channels = chunk.channels
                self._raw = bytearray(chunk.audio)
            else:
                self._raw.extend(chunk.audio)

            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug(
                "Audio stopped. Transcribing with initial prompt=%s",
                self.initial_prompt,
            )
            # Concatenate in-memory audio chunks and convert to float32 PCM
            assert self._rate is not None

            max_value = 32768.0 if self._width == 2 else 128.0

            audio_data = np.frombuffer(
                self._raw, dtype=np.int16 if self._width == 2 else np.int8
            ).astype(np.float32)
            audio_data = audio_data.reshape(-1, self._channels)
            if self._channels > 1:
                pcmf32 = (audio_data[:, 0] + audio_data[:, 1]) / (max_value * 2)
            else:
                pcmf32 = audio_data / max_value

            # Resample if needed
            if self._rate != WHISPER_SAMPLE_RATE:
                pcmf32 = resample(pcmf32, self._rate, WHISPER_SAMPLE_RATE)

            self._raw = None
            self._rate = None
            self._width = None
            self._channels = None

            async with self.model_lock:
                segments = self.model.transcribe(
                    pcmf32,
                    beam_search={
                        "beam_size": self.cli_args.beam_size,
                        "patience": self.cli_args.patience,
                    }
                    if self.cli_args.beam_size > 0
                    else None,
                    language=self._language,
                    initial_prompt=self.initial_prompt or "",
                )

            text = " ".join(segment.text for segment in segments)
            _LOGGER.info(text)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            # Reset
            self._language = self.cli_args.language

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
                _LOGGER.debug("Language set to %s", transcribe.language)
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
