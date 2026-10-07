# Target-speaker extraction prototype

2026-10-07. Offline experiment on `codex/target-speaker-extraction`, based on
`901b846fa21554b4d625dcf94975878da0ee7490`. No app binaries, saved settings,
or personal reference were changed. No microphone audio was captured
or uploaded.

## Finding

Reference-conditioned extraction works in the prototype: swapping the enrollment
recording changes which voice is extracted from the same mixture. This is audio
extraction, beyond the current whole-mixture presence gate.

**The tested model is not ready for HushMic.** Initial runs exceeded the CPU
budget and lost target speech. The follow-up below improves the speech-loss case
and demonstrates lower CPU cost in later runs. The real 48 kHz pipeline below
exposes runtime and speaker-selection limits; live-worker latency, word
preservation and sustained performance remain unverified.

## Follow-up: warm the recurrent state

Feed 128 frames (1.024 seconds) of synthetic silence through the extractor after
computing the fixed enrollment and resetting its state. This supplies artificial
past context; it reads no microphone audio or future samples. It adds about
0.6–0.7 seconds of initialization work for the full model in these runs, excluded
from streaming RTF. It adds no queued audio in the offline runner. App startup
and total processing latency still need measurement.

With the same six-block INT8 graph and reference, the earlier public first-turn
failure changes from target projection magnitude 0.63 to 1.05, and output SI-SDR
from 2.46 to 13.65 dB (input: 6.12 dB). This establishes that synthetic past
context changes the result; it does not establish general word preservation.
Primed FP32/INT8 checks cover state reset, future-input independence and chunk
partitioning. The original trained architecture remains the default.

The full fifty-second public stream used 0.629 CPU seconds and 0.633 wall seconds
per audio second. Chunk cost p95 was 49.1 ms and maximum 67.9 ms for 64 ms of
audio. Normal overlap improved target/competitor projections by 13.2 and 20.9 dB;
the long target-free core reduced energy by 19.2 dB. That pause score excludes
200 ms of release and 30 ms before resumption. A separate quiet-without-competitor
screen retained target projection magnitudes 0.99–1.06 and improved SI-SDR.

Earlier runs exceeding one CPU second per audio second remain recorded. Priming
adds initialization work and does not explain lower per-chunk CPU measurements;
load, frequency and scheduling were not controlled across all runs. The full
pipeline has not passed its 0.8 CPU budget or 200 ms ceiling. Target-only speech
in the long stream also falls from 18.0 to 16.1 dB SI-SDR despite retaining its
projection magnitude, so distortion and word/onset preservation still need work.
The first public crop overlaps calibration as documented below. Later long-stream
intervals are calibration-disjoint. Quiet speech under a louder competitor was
recorded as exploratory evidence and remains outside acceptance scope.

A four-block depth ablation also kept up in later INT8 runs, but weakened
selection and retained only 0.55 target projection in the unprimed failure case.
It removes trained blocks without retraining; it is not preferred over the
primed full model. `--blocks 4` reproduces it without replacing the six-block
graphs or checkpoint. The export wrapper now explicitly retains inference mode.

```bash
nice -n 19 taskset -c 2 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  target/voice-reference-evaluation/venv/bin/python \
  scripts/prototype-target-extraction.py --prime-silence \
  --backend ort --int8 --static-int8 --check --case primed-public-reproduce \
  --mixture target/target-speaker-extraction/public-mixture.wav \
  --target target/target-speaker-extraction/public-target.wav \
  --competitor target/target-speaker-extraction/public-competitor.wav \
  --enrollment target/voice-reference-evaluation/enrollment.wav
```

Further candidate screening found the University of Washington model has
binaural input, noncommercial source terms and missing checkpoint links;
OpenSpeakerBeam-SS uses global utterance normalization despite its streaming
description; DAVE uses unmasked temporal attention and separates anonymous
streams; and the downloadable Voce SpeakerBeam-SS checkpoint lacks a model card,
matching public implementation and verified model terms. Pinned source and
acquisition evidence are summarized in the results file. No candidate was
packaged or added to the live app.

## Local app trial

The user has now requested an experimental app build. `scripts/run-extraction-trial.sh`
launches the frozen local build under `target/target-speaker-extraction/app-trial/`.
It uses a separate configuration, cache and 192-float reference; it never imports
the existing 40-float detector enrollment. Open **Target-speaker setup…
(experimental)**, record 20 seconds, and enable extraction explicitly. Recording
starts only on the enrollment button. Audio goes through the encoder in RAM and
is discarded; the saved reference has mode 0600. **Forget voice** removes this
trial's reference. The original trial launcher remains available at
`target/voice-reference-evaluation/app-trial/launch-trial.sh`.

This trial uses the four-block, four-frame INT8 graph, with the same synthetic
silence priming and bounded sinc conversion as the offline runner. It runs
extraction alone when enabled; High Quality/Light and suppression-strength
settings apply to ordinary DPDFNet when extraction is disabled. It remains a
depth ablation of the six-block trained model. The known selection weaknesses
and load-dependent runtime failures still apply. No model is included in a
release or public package.

The Python subprocess runs on normal scheduling; the existing Rust worker,
sample rings and alignment ledger keep inference and subprocess I/O off the
PipeWire real-time callback. The bridge emits one 480-sample packet per hop with
a fixed 5,760-sample engine delay. The worker adds 2,400 samples at the supported
480-sample graph quantum: **8,160 samples / 170 ms**, reported to the graph and
diagnostics. Late output is substituted with aligned silence by the existing
ledger. A dead bridge, non-finite audio, invalid packet or response timeout
reports extraction unavailable and emits silence until the chain is restarted.
There is no automatic switch to a different speaker model. The app allows up to
30 seconds for initial model loading; resets restore a saved primed state.

The live processor's runnable `--check` matches the offline output exactly on
the tested prefix and verifies reset, bypass and mute. The opt-in plugin test
uses a public reference, paced callbacks and checks exact bypass alignment,
silent mute, reset and nonzero finite extraction output:

```bash
target/voice-reference-evaluation/venv/bin/python scripts/experimental-live-extraction.py --check
HUSHMIC_TSE_PYTHON="$PWD/target/voice-reference-evaluation/venv/bin/python" \
HUSHMIC_TSE_SCRIPT="$PWD/scripts/experimental-live-extraction.py" \
HUSHMIC_TSE_REFERENCE="$PWD/target/target-speaker-extraction/plugin-public-reference.bin" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
cargo test --release -p dpdfnet-ladspa --test extraction -- --ignored --nocapture
```

The built plugin also passed an active PipeWire startup smoke with a temporary
public reference: the Fifine input was selected, extraction reported active INT8,
and reported latency was 8,160 samples. Consumer audio was discarded. The public
reference was removed and extraction disabled before opening user enrollment.
This is a startup check, not a call-duration reliability or personal-voice test.

The reference in that command is generated from the separate public enrollment
fixture, not from the microphone. These checks establish integration behavior,
not word preservation or a reliable runtime pass during real calls. The local
build depends on the already acquired ignored prototype assets and Python
environment. Rebuilding Rust does not update the frozen Python copies; copy the
three prototype scripts together, pin their repository root and freeze the
INT8 graph before retesting. The prepared build records their exact hashes in
its ignored `ready.json`.

## Follow-up: actual 48 kHz pipeline

`scripts/prototype-extraction-pipeline.py` now feeds bounded, stateful sinc
conversion from 48 to 16 kHz, extraction, conversion back to 48 kHz, and optionally
the actual native INT8 High Quality denoiser. Its Rust bridge reuses
`StreamDenoiser`, asserts the native engine, and rejects inference errors. No
production DSP, settings or enrollment changed. At the start of this work the
trial app was stopped; this harness does not start it.

Resampler partition checks match the whole-file transform exactly in both
directions. Native output is identical across unequal packet partitions, with
its 2,400-sample algorithmic delay drained and removed only for acoustic scoring.
Extractor checks cover upstream parity, reset, partitioning and future-input
independence. The five-block graph executes 128 QLinearConv and ten
DynamicQuantizeLSTM kernels per chunk; removing blocks remains an ablation,
not a separately trained compact model.

The full fifty-second public scenario was replayed on this shared i5-1235U
desktop. Most runs used one thread on CPU 2; two-thread runs used CPUs 0 and 2,
which are different physical cores. Other desktop workloads were active and
changed between runs. No other processes or power settings were changed.
Synthetic-silence priming and loading are excluded from streaming costs.

| Configuration | CPU seconds/audio second | Maximum paced output availability |
| --- | ---: | ---: |
| Six blocks + High Quality, four-frame chunks | 1.60 | 31,473 ms simulated |
| Six blocks alone, eight-frame chunks | 0.79 | 338 ms |
| Six blocks alone, four-frame chunks | 0.96 | 773 ms |
| Five blocks alone, eight-frame chunks | 0.86 | 419 ms |
| Four blocks alone, first run | 0.52 | 162 ms |
| Four blocks alone, repeat | 0.75 | 211 ms |
| Four blocks, four-frame chunks, first full stream | 0.54 | 184 ms |
| Four blocks, four-frame chunks, repeated full stream | 0.72 | 122 ms |
| Four blocks, four-frame chunks, quiet target, pinned | 0.82 | 579 ms |
| Four blocks, four-frame chunks, quiet target, normal scheduling | 0.47 | 124 ms |
| Six blocks alone, two threads | 1.93 | 3,303 ms |
| Six blocks, two threads, spinning disabled, ten-second crop | 1.57 | 651 ms |

Availability tracks the earliest real source sample in each emitted packet;
paced runs also record actual wall-clock availability. The screen reserves
another 10 ms for a host quantum, requiring availability within 190 ms. It is
not a PipeWire or production-worker latency pass. Earlier runners summed CPU
inside chunk processing; the final runner also counts CPU during paced waits,
including idle thread spinning. The two-thread CPU result above is therefore
a lower bound. The first cascade screen, with precomputed input downsampling,
used 0.71 CPU seconds/audio second but already missed latency at 329 ms. The
subsequent fully streaming eight-frame cascade used 1.05 and reached 3,343 ms
of simulated backlog. These variable results are retained, not averaged into
a shipping claim.

Adding High Quality after extraction improved normal-overlap projections to
16–25 dB and target-free reduction to about 41 dB. It worsened target-only
SI-SDR from about 15.8 dB for extraction alone to 13.2 dB (input: 18.0 dB).
Four blocks alone reduced the target-free core by 13.0 dB, with 10.1/12.6 dB
overlap selection and target projection magnitudes 0.84/0.91. Its target-only
SI-SDR was 11.9 dB. Five blocks improved those quality scores somewhat but
missed the runtime limits. The four-block repeat used the same fixture, graph
and enrollment inputs while its availability crossed 200 ms: one passing
runtime screen is insufficient.
Scores retain the calibration-overlap caveat; quiet speech under a louder
competitor remains outside acoustic acceptance.

Four blocks with four-frame chunks passed both fifty-second paced screens.
Ordinary quiet speech retained projection magnitudes 0.82–0.83 and slightly
improved SI-SDR (6.9–7.0 to 7.4–7.5 dB). Its pinned runtime failed; normal OS
scheduling passed. These are conditional runtime results, not a worker guarantee.
The independent LibriSpeech case gave 7.2/10.0 dB selection and 14.1 dB pause
reduction with the original reference. Swapping enrollment to the other speaker
reduced selection to 0.5/3.8 dB despite separate enrollment utterances. An FP32
control also gave only 0.7/3.9 dB, with FP32/INT8 waveform RMSE 0.00039. The
weak alternate identity is therefore present without INT8: changing precision
alone does not repair the reduced-depth model.

The saved, already aligned ordinary High Quality baseline scores 5.6 dB SI-SDR
on the same target-only core, versus 18.0 dB for the input. It is historical,
not a simultaneous runtime control. This puts the lower-than-input extraction
scores in context: waveform SI-SDR alone cannot establish an audible regression
relative to the accepted app. Word preservation and listening remain open.

**No configuration has reliably passed the shipping limits.** Cascading both
processors adds cost and distortion; extraction as the sole processor deserves
further work, but cutting trained blocks sacrifices fidelity. A properly trained
smaller model or a materially faster runtime is still needed, alongside model
terms, word/onset preservation and live-worker checks. Upsampling the 16 kHz
model output does not restore frequencies above 8 kHz.

The full model on the P-core pool reached 188 ms for the quiet screen, but used
0.85 CPU seconds/audio second. Its four-frame run with normal scheduling used
1.69 and fell 7.4 seconds behind. The already installed OpenVINO 2026.3.1 SDK
accepted both the convolution-INT8 and dynamic-INT8 graphs. Its profiling showed
128 INT8 convolution kernels but 18 FP32 reference LSTM sequences, including
when importing the dynamic-INT8 graph. Both variants exceeded one CPU
second/audio second and accumulated seconds of delay; its exposed devices were
CPU only. A separately installed, ignored ORT 1.30.0 Python wheel also missed
the quiet screen (0.93 CPU seconds/audio second, 301 ms availability). These
shared-desktop runs do not establish a controlled speed ranking between SDKs;
none establishes a faster passing full-model configuration.

Build the bridge and reproduce the four-block paced screen:

```bash
cargo build --release -p hushmic-denoiser --features native --example prototype_pipeline
PYTHONDONTWRITEBYTECODE=1 taskset -c 2 target/voice-reference-evaluation/venv/bin/python \
  scripts/prototype-extraction-pipeline.py --paced --check --extractor-only \
  --frames 4 --blocks 4 --case reproduce-pipeline-b4-f4
```

Omit `--extractor-only` to measure the native High Quality cascade. Use
`--frames 4 --export` for its four-frame graph, or `--blocks 5 --export` for
the five-block ablation. `--ort-threads 2 --no-spin` reproduces the alternative
thread setting, documented in [ONNX Runtime's threading guide](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
and checked through Context7. `--backend openvino` reuses the isolated SDK and
records executed kernels; it is not an app backend. Its CPU configuration follows
the [OpenVINO scheduling documentation](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device/performance-hint-and-thread-scheduling.rst).
WAVs, per-chunk timing traces and JSON remain in the ignored experiment directory.
Results are recorded in `docs/target-speaker-extraction-results.json`. Earlier
runner hashes were read at completion while files could be edited during runs;
they are labeled as completion hashes, not exact loaded-code provenance. The
final runner captures both sources at startup and executes the captured extractor
bytes. Its metadata identifies that policy explicitly.

## Candidate and implementation

The [REAL-TSE baseline](https://github.com/REAL-TSE/wesep-real-tse) publishes
`spk_emb_causal_100`, a 16 kHz mono BSRNN with an ECAPA enrollment encoder.
It has 25,049,608 parameters including the encoder. Its reference is computed
once and remains fixed; it cannot reuse our Lingometer 40-float reference.
The existing enrollment UI can be retained, but this model needs re-enrollment.

`scripts/prototype-target-extraction.py` loads the real checkpoint with
`weights_only=True` and strict tensor matching. It pins both source commits and
checkpoint/config hashes. It adds persistent state to the six temporal LSTMs
without changing the trained weights. The other bidirectional LSTMs operate
across frequency bands within each frame, not across future audio.

The runner replays 512-sample analysis windows with 128-sample hops, processes
eight frames per chunk, and emits overlap-added audio only when its contributing
frames are available. Input readiness is at most 88 ms for this chunk size.
This figure excludes computation, resampling, worker/host buffering, and the
optional denoiser; it is not total app latency. Chunk-wide dynamic INT8 scaling
is included in that readiness bound. The future-input regression check compares
only audio whose input deadline precedes the modified suffix.

The ONNX graph exposes twelve recurrent state tensors. Dynamic INT8 covers
LSTM/matrix kernels; static QDQ INT8 also covers convolutions. Enrollment, FFT,
normalization and synthesis retain floating point. Calibration uses five seconds
of the existing LibriVox fixture and its separate enrollment, not the LibriSpeech
evaluation audio. An actual CPU runtime profile verified 128 `QLinearConv`,
12 `DynamicQuantizeLSTM`, and 13 `DynamicQuantizeMatMul` executions per chunk.

## What was measured

The first fixture is the repository's ten-second LibriSpeech regression case:
target plus competitor, competitor alone during a four-second target pause, then
resumed overlap. Enrollment comes from different utterances. These speakers are
from train-clean-100, which overlaps the candidate's declared training corpus;
this is a development screen, not unseen-speaker certification.

| Case | Overlap target/competitor projection improvement | Target-free reduction |
| --- | --- | --- |
| LibriSpeech, intended enrollment, mixed INT8 | about 22–24 dB | about 35–36 dB |
| Same mixture, competitor enrolled | output switches toward competitor | competitor remains audible |
| Different LibriVox readings, mixed INT8 | 8.2 dB first turn; 21.0 dB resumed turn | 17.9 dB |
| Calibration-disjoint LibriVox crops, frozen mixed INT8 | 23.6 dB first turn; 13.2 dB resumed turn | 18.6 dB |

On LibriVox's first turn, target projection magnitude falls to 0.63 and SI-SDR
falls from 6.12 dB in the mixture to 2.46 dB after extraction. The resumed turn
improves from 8.96 to 17.53 dB. This inconsistency prevents a speech-preservation
pass. The public test begins with speech already present, so startup is part of
this result; it does not establish the cause of the loss. Quiet speech beneath
a louder competitor remains deferred and was not scored here.

Two seconds of the first public turn overlap INT8 calibration audio, so that
exploratory screen is not held out from calibration either. A separate fixed-model
run uses only source ranges `[8.5,11.5)`, `[13.5,17.5)` and `[5,8)`, disjoint from
the calibration range `[0,5)`. Both screens are retained in the results; neither
constitutes a real-call quality certification.

Projection values compare extracted PCM with separately available clean source
components. They do not prove word intelligibility. The checkpoint can reverse
waveform polarity; signed projection coefficients are therefore retained in the
JSON, and their magnitudes are used for selection ratios. Target-free energy is
compared with the input mixture, not the ordinary-denoising baseline. These values
do not certify the earlier 12 dB app acceptance requirement.

On the i5-1235U, unpinned one-thread mixed-INT8 runs used roughly 1.5–2.2 process
CPU seconds per audio second. Pinning to a performance core brought one run near
1.0, still above the existing 0.8 total-pipeline budget without any denoiser.
Two performance cores used 2.63 CPU seconds per audio second and took 1.37 wall
seconds per audio second. It cannot sustain our live stream with headroom.
Sequential final measurements and checks are saved in the accompanying results
file. An earlier FP32 run overlapped another benchmark and is excluded from CPU
comparisons.

The regression check compares cached streaming against the original trained
model, checks chunk partitioning, stream reset, future-input independence at the
declared deadline, and reports ONNX/quantization drift separately. FP32 streaming
maximum waveform error was below `2e-8`; ONNX FP32 waveform RMSE was about `2e-9`.
These are implementation checks, not acoustic acceptance tests.

## Reproduce on this checkout

Downloaded sources, checkpoints, graphs, public fixture components, PCM outputs,
logs and acquisition hashes are under the ignored directory
`target/target-speaker-extraction/`. Existing Python dependencies are reused from
`target/voice-reference-evaluation/venv`: PyTorch/torchaudio `2.9.1+cpu`, NumPy
`2.5.3`, SoundFile `0.14.0`, PyYAML `6.0.3`, ONNX `1.23.1`, ORT `1.27.0`.

```bash
taskset -c 0 target/voice-reference-evaluation/venv/bin/python \
  scripts/prototype-target-extraction.py \
  --backend ort --int8 --static-int8 --check --case reproduce \
  --target target/target-speaker-extraction/target.wav \
  --competitor target/target-speaker-extraction/competitor.wav
```

Use `--export` to regenerate the floating-point and INT8 graphs. Use `--enrollment`
with `competitor-enrollment.wav` for the reference swap. For the second test,
use `--mixture public-mixture.wav`, `--target public-target.wav`, and
`--competitor public-competitor.wav` under the same directory, together with
`--enrollment target/voice-reference-evaluation/enrollment.wav` and a distinct
`--case`. Audio and performance JSON are written under the ignored directory.
Use the corresponding `disjoint-*.wav` files for the calibration-disjoint run.

For reacquisition, clone the baseline at
`2a540977a348fbaa92e623210505430e2cec608d` into `upstream/` and WeSpeaker at
`8f53b6485d9f88a207bd17e7f8dba899495ec794` into `wespeaker/` there.
Download the [author-linked checkpoint archive](https://drive.google.com/uc?export=download&id=1M4UqK2A2EeHmQ0pCevYqBgaYn3RvklgC),
SHA256 `0a7e44050b1235db209181d7cda29ac93487fbb981ec768d3062b3243c49ce98`,
and extract `spk_emb_causal_100/{config.yaml,avg_model.pt}` into `models/`.
The script checks those file hashes before loading. The comparison components
are cropped from the existing sealed fixture; their hashes, crop ranges and
original attribution are recorded in the experiment artifacts/results.

No explicit trained-weight redistribution grant was found in the archive or
baseline README. Source-file Apache-2.0 notices do not resolve that question.
The weights are not added to the repository or release package. Shipping also
requires resolving the model terms, sustained combined-worker performance,
48 kHz adaptation, total latency, preservation/onsets, and real call quality.

The initial live status check found the existing trial in Light fallback with
protection unavailable (`Detector stream fell behind; ordinary denoising
continues`); its saved choice remains High Quality. CPU experiments may have
contributed to the lag. Chromium had two active microphone consumers, so the
chain was not restarted during possible call use. No experiment process remains
running. After the follow-up experiments, there were no microphone consumers,
so the frozen trial was relaunched with its saved High Quality and protection
requests. The chain is running; its runtime engine and detector await audio
while idle. The new extractor remains offline, and the personal reference was
not replaced or inspected.

Context7 documentation for PyTorch and ONNX Runtime was consulted for CPU
inference and quantization. The pinned implementation and
actual runtime measurements determine the findings above.
