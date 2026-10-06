#!/usr/bin/env python3
"""Export the optional experimental Lingometer assets; never run by the app.

Use the CPU versions recorded in assets/models/voice-models.json. Downloads are
from the pinned upstream commit and SHA256 checked before source execution.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
COMMIT = "815c19c0fdf56275cc7bb5dcc0be71e281fe5ab0"
PINS = {
    "LICENSE": "5625186693186f3f2e0d1f673e0c75fcc9defc4ff90cc9332d6109abfd39b53a",
    "personal_VAD/ASPVAD.py": "2f8b5cfa47cbb53204396e808ef4db06a084e83094044e4423094e4e71dbf066",
    "personal_VAD/models/ckpt_librispeech_spk2338.pt": "91a90898b84b9e7d4090938b5e93eabc9d4cc91e6eb9d3beb81ba208d8e4f9c0",
}
PRESENCE_SHA = "2b0d4c83be8b773fc53e041f9090b0aefdd4c661865798771ca4226c8005a8bd"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, help="Optional already acquired pinned Lingometer checkout")
    parser.add_argument("--presence-source", type=Path, help="Copy the exact evaluated INT8 graph instead of re-exporting it")
    parser.add_argument("--output-root", type=Path, default=ROOT, help="Artifact destination root (default: repository)")
    args = parser.parse_args()
    root = args.output_root.resolve()
    import numpy as np
    import scipy
    from scipy.signal import firwin, lfilter
    import torch
    import torchaudio
    from torchaudio.functional.functional import _get_sinc_resample_kernel
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    versions = dict(numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                    torchaudio=torchaudio.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__)
    pinned = json.loads((ROOT / "assets/models/voice-models.json").read_text())
    if versions != pinned["versions"]:
        raise ValueError(f"Export requires the recorded CPU package versions: {pinned['versions']}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(False)
    assets = root / "assets/models"
    assets.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        upstream = args.upstream or Path(scratch)
        for name, digest in PINS.items():
            path = upstream / name
            if not args.upstream:
                path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f"https://raw.githubusercontent.com/junghunl/Lingometer/{COMMIT}/{name}", path)
            if sha(path) != digest:
                raise ValueError(f"Pinned upstream hash mismatch: {name}")
        spec = importlib.util.spec_from_file_location("voice_ASPVAD", upstream / "personal_VAD/ASPVAD.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.ASPVAD(in_dim=24, num_speakers=2338, modulation="FiLM", AS_chunk_size=500,
                             emb_dim=40, emb_num_layers=4, feat_num_layers=4, att_dim=8, pred_spk=True).eval().requires_grad_(False)
        state = torch.load(upstream / "personal_VAD/models/ckpt_librispeech_spk2338.pt", map_location="cpu", weights_only=True)
        state = state.get("model", state)
        expected = model.state_dict()
        assert len(state) == 149 and state.keys() == expected.keys()
        assert all(v.shape == expected[k].shape and v.dtype == expected[k].dtype and torch.isfinite(v).all() for k, v in state.items())
        model.load_state_dict(state, strict=True)
        assert sum(p.numel() for p in model.parameters()) == 131224
        assert sum(p.numel() for p in model.embedding_extractor.parameters()) == 9606
        before = {k: v.numpy().tobytes() for k, v in model.state_dict().items()}

        # Same bounded received-audio geometry as the evaluated stock candidate.
        class Presence(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_tcnn = model.feature_tcnn
                self.attentive_score = model.attentive_score
                self.p_classifier = model.p_classifier

            def forward(self, mel, reference):
                f = self.feature_tcnn(mel)[..., -20:]
                a, _ = self.attentive_score(f, reference)
                z = self.p_classifier(a[..., -11:])[:, -6]
                return z, torch.sigmoid(z)

        encoder_path = assets / "voice-reference-encoder.onnx"
        presence_path = assets / "voice-presence-int8.onnx"
        synthetic = torch.zeros(1, 24, 2001)
        with torch.inference_mode():
            torch.onnx.export(model.embedding_extractor, synthetic, str(encoder_path), dynamo=False, opset_version=17,
                              input_names=["mel"], output_names=["reference"])
        model.attentive_score.chunk_size = None
        if args.presence_source:
            if sha(args.presence_source) != PRESENCE_SHA:
                raise ValueError("Not the evaluated stock INT8 graph")
            shutil.copyfile(args.presence_source, presence_path)
        else:
            fp32 = Path(scratch) / "presence-fp32.onnx"
            with torch.inference_mode():
                torch.onnx.export(Presence().eval(), (torch.zeros(1, 24, 50), torch.zeros(1, 40)), str(fp32),
                                  dynamo=False, opset_version=17, input_names=["mel", "reference"], output_names=["logit", "probability"],
                                  dynamic_axes={"mel": {0: "B", 2: "T"}, "reference": {0: "B"}, "logit": {0: "B"}, "probability": {0: "B"}})
            quantize_dynamic(str(fp32), str(presence_path), weight_type=QuantType.QUInt8,
                             op_types_to_quantize=["Conv", "MatMul", "Gemm"], per_channel=False, reduce_range=False)
        for path in (encoder_path, presence_path):
            onnx.checker.check_model(onnx.load(path))
            if sha(path) != pinned["assets_sha256"][path.name]:
                raise ValueError(f"Export did not reproduce the evaluated asset: {path.name}")

        mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=24, win_length=400, hop_length=160)
        enroll_kernel, width = _get_sinc_resample_kernel(48000, 16000, 16000, dtype=torch.float32)
        assert width == 19 and enroll_kernel.shape == (1, 1, 41)
        coeffs = dict(LIVE_FIR=np.asarray(firwin(61, 1/3), dtype=np.float64), ENROLL_FIR=enroll_kernel.numpy().ravel(),
                      HANN=mel.spectrogram.window.numpy(), MEL=mel.mel_scale.fb.numpy().T.copy().ravel())
        coefficients = root / "crates/hushmic-denoiser/src/voice_coefficients.bin"
        coefficients.parent.mkdir(parents=True, exist_ok=True)
        coefficients.write_bytes(b"".join(v.astype("<f8" if name == "LIVE_FIR" else "<f4").tobytes() for name,v in coeffs.items()))
        source = """// Generated by scripts/export-voice-assets.py; exact upstream coefficient bits.
// MEL is [24 mel bands][201 FFT bins]. Enrollment FIR is cross-correlation.
const DATA: &[u8; 21548] = include_bytes!("voice_coefficients.bin");
const fn f32s<const N: usize>(offset: usize) -> [f32; N] {
    let mut values = [0.; N];
    let mut i = 0;
    while i < N {
        let k = offset + i * 4;
        values[i] = f32::from_bits(u32::from_le_bytes([DATA[k], DATA[k+1], DATA[k+2], DATA[k+3]]));
        i += 1;
    }
    values
}
const fn f64s<const N: usize>(offset: usize) -> [f64; N] {
    let mut values = [0.; N];
    let mut i = 0;
    while i < N {
        let k = offset + i * 8;
        values[i] = f64::from_bits(u64::from_le_bytes([DATA[k], DATA[k+1], DATA[k+2], DATA[k+3], DATA[k+4], DATA[k+5], DATA[k+6], DATA[k+7]]));
        i += 1;
    }
    values
}
pub(super) const LIVE_FIR: [f64; 61] = f64s(0);
pub(super) const ENROLL_FIR: [f32; 41] = f32s(488);
pub(super) const HANN: [f32; 400] = f32s(652);
pub(super) static MEL: [f32; 24 * 201] = f32s(2252);
"""
        (coefficients.with_suffix(".rs")).write_text(source)

        # Integer-only seeded PCM generation is identical in Rust and Python.
        raw = np.empty(960000, dtype=np.float32)
        x = 0x12345678
        for i in range(len(raw)):
            x ^= (x << 13) & 0xffffffff
            x ^= x >> 17
            x ^= (x << 5) & 0xffffffff
            raw[i] = np.float32((x >> 8) - 8388608) / np.float32(104857600)
        with torch.inference_mode():
            resampled = torchaudio.functional.resample(torch.from_numpy(raw), 48000, 16000)
            features = torch.log10(mel(resampled) + 1e-6)[None].contiguous()
            wanted_reference = model.embedding_extractor(features).numpy()
            # Independent causal streaming frontend oracle: no reflected right tail.
            live = lfilter(coeffs["LIVE_FIR"], [1.], raw[:480*160])[::3].astype(np.float32)[10:]
            live_features = torch.log10(mel(torch.from_numpy(live)) + 1e-6).numpy()
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning", "session.set_denormal_as_zero"):
            options.add_session_config_entry(key, "0")
        encoder = ort.InferenceSession(str(encoder_path), options, providers=["CPUExecutionProvider"])
        reference, = encoder.run(None, {"mel": features.numpy()})
        error = float(np.max(np.abs(reference - wanted_reference)))
        assert error <= 1e-4, error
        detector = ort.InferenceSession(str(presence_path), options, providers=["CPUExecutionProvider"])
        enrollment_frames = [0, 1, 2, 17, 1000, 1999, 2000]
        live_frames = [0, 1, 5, 44, 49, 100, 150]
        probabilities = []
        for j in [0, 1, 14, 44, 100]:
            window = np.ascontiguousarray(live_features[:, max(0, j-44):j+6][None])
            z, p = detector.run(None, {"mel": window, "reference": reference})
            probabilities.append(dict(source_hop=j, logit=float(z[0]), probability=float(p[0])))
        assert all(v.numpy().tobytes() == before[k] for k, v in model.state_dict().items())
        parity = dict(seed=0x12345678, raw_samples=len(raw), raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                      enrollment_frames=enrollment_frames, enrollment_features=features.numpy()[0][:, enrollment_frames].T.tolist(),
                      live_frames=live_frames, live_features=live_features[:, live_frames].T.tolist(),
                      reference=reference[0].tolist(), torch_reference=wanted_reference[0].tolist(), encoder_max_error=error,
                      probabilities=probabilities, enrollment_resampled_samples=320000, enrollment_feature_shape=[1,24,2001])
        (assets / "voice-frontend-parity.json").write_text(json.dumps(parity, indent=2) + "\n")
        shutil.copyfile(upstream / "LICENSE", assets / "Lingometer-LICENSE")
        manifest = dict(experimental=True, quality="Current stock candidate measured 0.9–7.4 dB competitor suppression; no 12 dB guarantee.",
                        upstream=f"https://github.com/junghunl/Lingometer/tree/{COMMIT}", license="Apache-2.0", upstream_sha256=PINS, evaluated_presence_sha256=PRESENCE_SHA,
                        versions=versions,
                        assets_sha256={p.name:sha(p) for p in (encoder_path,presence_path,assets/"voice-frontend-parity.json",assets/"Lingometer-LICENSE")},
                        encoder_input=[1,24,2001], encoder_output=[1,40], detector_physical_frames=50, detector_attention_frames=20,
                        reference_normalization="none", policy=dict(threshold=.5, release_hangover_ms=160, maximum_preroll_ms=70, minimum_mask_frame_probability=.1))
        (assets / "voice-models.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(dict(assets_sha256=manifest["assets_sha256"], encoder_max_error=error)))


if __name__ == "__main__":
    run()
