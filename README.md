# Wyoming Whisper.cpp

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [whisper.cpp](https://github.com/ggml-org/whisper.cpp) speech to text system.

This project was based on wyoming-faster-whisper. I wanted to adopt whisper.cpp instead to allow more backends. In particular, since mid 2025, the Vulkan
backend improved significantly, and offers excellent performance on all modern GPUs without installing much dependencies. You should try it for yourself,
and decide what's most performant for your hardware.

While not yet integrated, whisper.cpp has experimental support to detect turns of speakers. This could allow us to only return the transcript of the first
speaker, assuming that what follows is background chat. Or we may try extracting the timestamp of each turn of speaker to split the audio into segments,
and run a diarization model to keep only segments matching the first speaker.

## Local Install

Clone the repository and set up Python virtual environment:

``` sh
git clone https://github.com/debackerl/wyoming-whisper.cpp.git
cd wyoming-whisper.cpp
script/setup
```

Run a server anyone can connect to:

```sh
script/run --model tiny-int8 --language en --uri 'tcp://0.0.0.0:10300' --data-dir /data --download-dir /data
```

See [available models](https://absadiki.github.io/pywhispercpp/#pywhispercpp.constants.AVAILABLE_MODELS).
