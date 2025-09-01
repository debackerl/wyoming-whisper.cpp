"""Event handler for clients of the server."""
import argparse
import asyncio
import logging
import os
import tempfile
import wave
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
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug(
                "Audio stopped. Transcribing with initial prompt=%s",
                self.initial_prompt,
            )
            assert self._wav_file is not None

            self._wav_file.close()
            self._wav_file = None

            with wave.open(self._wav_path, "rb") as wav_file:
                max_value = 32768.0 if wav_file.getsampwidth() == 2 else 128.0
                audio_data = wav_file.readframes(wav_file.getnframes())

                audio_data = np.frombuffer(
                    audio_data,
                    dtype=np.int16 if wav_file.getsampwidth() == 2 else np.int8,
                ).astype(np.float32)
                audio_data = audio_data.reshape(-1, wav_file.getnchannels())
                if wav_file.getnchannels() > 1:
                    pcmf32 = (audio_data[:, 0] + audio_data[:, 1]) / (max_value * 2)
                else:
                    pcmf32 = audio_data / max_value

                if wav_file.getframerate() != WHISPER_SAMPLE_RATE:
                    pcmf32 = resample(
                        pcmf32, wav_file.getframerate(), WHISPER_SAMPLE_RATE
                    )

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
