# Test fixtures

Public clips used by the tests. All are 48 kHz mono. The noisy and golden clips are built by
`docs/demo/make-demos.py` from the sources credited in
`docs/demo/ASSETS.md` (a public-domain LibriVox reading mixed with public
or CC BY noise at +3 dB SNR).

| File | Content |
|------|---------|
| `noisy_public_48k.flac` | speech over a fan |
| `noisy_cafe_48k.flac` | speech over café chatter |
| `noisy_keyboard_48k.flac` | speech over mechanical keyboard typing |
| `golden_public_dpdfnet8.flac` | `noisy_public_48k.flac` through `dpdfnet8_48khz_hr` |
| `voice-enrollment.flac` | twenty seconds of independent enrollment speech, speaker 5789 |
| `voice-evaluation.flac` | target 5789 with competitor 6272, a target pause, and resumed target speech |

`crates/dpdfnet-ladspa/tests/adaptive_models.rs` runs the three noisy clips
through the adaptive engine with a scripted CPU load (sustained overload,
recovery, a flapping load with stalls) and checks what reaches the output.
`ci/tier1-system-test.sh` loops `noisy_public_48k.flac` through a real
PipeWire graph.

The voice fixtures come from [LibriSpeech](https://www.openslr.org/12/),
copyright 2014 Vassil Panayotov, with readings by LibriVox volunteers, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They were cropped,
resampled, mixed and encoded as PCM16 FLAC. Enrollment and evaluation use
different source files. `voice-fixtures.json` records source hashes, exact
crops, processing and labeled intervals. These fixtures check the complete
plugin path and lifecycle; they do not certify broader speaker discrimination.
