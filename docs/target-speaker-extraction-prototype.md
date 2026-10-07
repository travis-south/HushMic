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
and demonstrates lower CPU cost in later runs, but combined-worker latency,
speech preservation and sustained performance under load remain unverified.

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
