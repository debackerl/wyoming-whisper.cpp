# Wyoming Faster Whisper

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [whisper.cpp](https://github.com/ggml-org/whisper.cpp) speech to text system.

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
