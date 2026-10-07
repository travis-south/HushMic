#!/usr/bin/env python3
"""Offline REAL-TSE experiment. Does not change HushMic or use personal enrollment."""
import argparse
import hashlib
import json
import os
import platform
import sys
import subprocess
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "target/target-speaker-extraction"
SOURCE = "2a540977a348fbaa92e623210505430e2cec608d"
SPEAKER_SOURCE = "8f53b6485d9f88a207bd17e7f8dba899495ec794"
MODEL = "spk_emb_causal_100"
WEIGHTS_SHA = "42b73219eefefcbba4b5da0dcb89dce292359c9f93b1a951c7b6d196da2155b3"
CONFIG_SHA = "80820eb78de6639160f7124d09cbacbc8397e321d8075721e89edb7cec50a0fa"


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_model(blocks=6):
    for folder, commit in [("upstream", SOURCE), ("wespeaker", SPEAKER_SOURCE)]:
        actual = subprocess.check_output(["git", "-C", str(WORK / folder), "rev-parse", "HEAD"], text=True).strip()
        assert actual == commit, f"Wrong {folder} source revision"
        assert not subprocess.check_output(["git", "-C", str(WORK / folder), "status", "--porcelain"], text=True).strip()
    # Import only inference modules: upstream package initializers pull in the
    # download CLI, Silero, and unrelated speaker architectures.
    for package, path in [("wesep", WORK / "upstream/wesep"),
                          ("wespeaker", WORK / "wespeaker/wespeaker")]:
        module = types.ModuleType(package)
        module.__path__ = [str(path)]
        sys.modules[package] = module
    from wespeaker.models import ecapa_tdnn
    factory = types.ModuleType("wespeaker.models.speaker_model")
    factory.get_speaker_model = lambda name: getattr(ecapa_tdnn, name)
    sys.modules[factory.__name__] = factory
    from wesep.models.tse_bsrnn_spk import TSE_BSRNN_SPK
    folder = WORK / "models" / MODEL
    assert digest(folder / "avg_model.pt") == WEIGHTS_SHA, "Wrong checkpoint bytes"
    assert digest(folder / "config.yaml") == CONFIG_SHA, "Wrong checkpoint configuration"
    config = yaml.safe_load((folder / "config.yaml").read_text())["model_args"]["tse_model"]
    config["speaker"]["speaker_model"]["speaker_encoder"]["pretrained"] = None
    model = TSE_BSRNN_SPK(config).eval()
    state = torch.load(folder / "avg_model.pt", map_location="cpu", weights_only=True)["models"][0]
    model.load_state_dict(state, strict=True)
    assert len(model.sep_model.separator.separation) == 6
    # Depth ablation only: these retained weights were trained with six blocks.
    model.sep_model.separator.separation = model.sep_model.separator.separation[:blocks]
    return model.eval()


def audio(path):
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    assert samples.shape[1] == 1 and np.isfinite(samples).all()
    samples = torch.from_numpy(samples[:, 0].copy()).unsqueeze(0)
    return torchaudio.functional.resample(samples, rate, 16000) if rate != 16000 else samples


def reference(model, enroll):
    torch.manual_seed(42)  # Preserve upstream enrollment dither, reproducibly.
    return {"spkemb": model.spk_ft.spkemb.compute(enroll)}


def spectrum(model, spec, cues):
    separator = model.sep_model
    ri = torch.stack((spec.real, spec.imag), 1)
    subbands = separator.band_split(spec)
    feature = separator.subband_norm(separator.band_split(ri))
    feature = model.spk_ft.spkemb.post(feature, cues["spkemb"].unsqueeze(1).unsqueeze(3))
    output = separator.band_masker(separator.separator(feature), subbands)
    return torch.complex(output[:, 0], output[:, 1])


def cached_forward(self, value):
    recurrent, self.cached_state = self.rnn(
        self.norm(value).transpose(1, 2).contiguous(), self.cached_state)
    return value + self.proj(recurrent).transpose(1, 2).contiguous()


def enable_cache(model):
    for block in model.sep_model.separator.separation:
        layer = block.band_rnn
        assert not layer.rnn.bidirectional
        layer.forward = types.MethodType(cached_forward, layer)
        layer.cached_state = None


def reset_cache(model):
    for block in model.sep_model.separator.separation:
        block.band_rnn.cached_state = None


class ExportCore(torch.nn.Module):
    """The same pretrained separator, with explicit recurrent input/output state."""
    def __init__(self, model):
        super().__init__()
        self.sep = model.sep_model
        self.fusion = model.spk_ft.spkemb

    def forward(self, ri, embedding, *states):
        feature = self.sep.subband_norm(self.sep.band_split(ri))
        feature = self.fusion.post(feature, embedding.unsqueeze(1).unsqueeze(3))
        batch, bands, dimension, frames = feature.shape
        feature = feature.reshape(batch, bands * dimension, frames)
        updated = []
        for i, block in enumerate(self.sep.separator.separation):
            layer = block.band_rnn
            value = feature.reshape(batch * bands, dimension, frames)
            recurrent, state = layer.rnn(layer.norm(value).transpose(1, 2).contiguous(),
                                         (states[i * 2], states[i * 2 + 1]))
            value = value + layer.proj(recurrent).transpose(1, 2)
            value = value.reshape(batch, bands, dimension, frames)
            value = value.permute(0, 3, 2, 1).reshape(batch * frames, dimension, bands)
            feature = block.band_comm(value).reshape(batch, frames, dimension, bands)
            feature = feature.permute(0, 3, 2, 1).reshape(batch, bands * dimension, frames)
            updated.extend(state)
        feature = feature.reshape(batch, bands, dimension, frames)
        result, offset = [], 0
        for i, width in enumerate(self.sep.band_split.band_width):
            mask = self.sep.band_masker.mask[i](feature[:, i])
            mask = mask.reshape(batch, 1, 2, 2, width, frames)
            mask = mask[:, :, 0] * torch.sigmoid(mask[:, :, 1])
            real, imag = ri[:, 0, offset:offset + width], ri[:, 1, offset:offset + width]
            result.append(torch.stack((real * mask[:, 0, 0] - imag * mask[:, 0, 1],
                                       imag * mask[:, 0, 0] + real * mask[:, 0, 1]), 1))
            offset += width
        return (torch.cat(result, 2), *updated)


def ort_core(model, cues, frames, threads, export, quantize, static_int8=False):
    import onnxruntime as ort
    assert set(cues) == {"spkemb"}, "ONNX pilot supports the embedding-conditioned candidate"
    blocks = len(model.sep_model.separator.separation)
    suffix = f"-b{blocks}" if blocks != 6 else ""
    graph = WORK / f"bsrnn-causal-f{frames}{suffix}.onnx"
    states = [torch.zeros(1, model.sep_model.nband, 256) for _ in range(2 * blocks)]
    names = [f"state_{i}" for i in range(2 * blocks)]
    if export:
        torch.onnx.export(ExportCore(model).eval(),
                          (torch.randn(1, 2, 257, frames), cues["spkemb"], *states),
                          graph, input_names=["spectrum", "reference", *names],
                          output_names=["extracted", *[f"next_{i}" for i in range(len(states))]],
                          opset_version=17, dynamo=False)
        assert not any(module.training for module in model.modules()), "Export changed inference mode"
    static_graph = graph.with_stem(graph.stem + "-conv-int8")
    if static_int8 and (export or not static_graph.exists()):
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        calibration = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
        # Separate public recordings from the LibriSpeech evaluation fixture.
        cal_audio = audio(ROOT / "target/voice-reference-evaluation/scenario.wav")[:, :5 * 16000]
        cal_reference = reference(model, audio(ROOT / "target/voice-reference-evaluation/enrollment.wav"))
        cal_spec = model.sep_model.stft(cal_audio)[-1]
        cache, records = [s.numpy() for s in states], []
        for first in range(0, cal_spec.shape[-1] - frames + 1, frames):
            spec = cal_spec[..., first:first + frames]
            values = {"spectrum": torch.stack((spec.real, spec.imag), 1).numpy(),
                      "reference": cal_reference["spkemb"].numpy(), **dict(zip(names, cache))}
            cache = calibration.run(None, values)[1:]
            if (first // frames) % 5 == 0:
                records.append(values)

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.records = iter(records)

            def get_next(self):
                return next(self.records, None)

        quantize_static(str(graph), str(static_graph), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        op_types_to_quantize=["Conv"], per_channel=True)
    if static_int8:
        graph = static_graph
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantized = graph.with_stem(graph.stem + "-int8")
        # First measure ORT's recurrent/linear INT8 kernels; convolution remains FP32.
        if export or not quantized.exists():
            quantize_dynamic(str(graph), str(quantized), weight_type=QuantType.QInt8,
                             op_types_to_quantize=["LSTM", "MatMul", "Gemm"], per_channel=True)
        graph = quantized
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
    cache = [s.numpy() for s in states]

    def reset():
        nonlocal cache
        cache = [s.numpy() for s in states]

    def run(_, spec, __):
        nonlocal cache
        # Static graph: pad only the final EOF chunk, then discard its extra output.
        count = spec.shape[-1]
        ri = torch.stack((spec.real, spec.imag), 1)
        ri = torch.nn.functional.pad(ri, (0, frames - count))
        values = session.run(None, {"spectrum": ri.numpy(), "reference": cues["spkemb"].numpy(),
                                    **dict(zip(names, cache))})
        cache = values[1:]
        ri = torch.from_numpy(values[0][..., :count])
        return torch.complex(ri[:, 0], ri[:, 1]).unsqueeze(1)

    run.reset = reset
    run.graph = graph
    return run


def extract(model, samples, cues, frames, core=spectrum, prime_silence=False):
    """Replay a stream at 16 kHz; bounded input windows and persistent LSTM state.

    Each frame reads only through center+256 samples. Outputs are emitted once
    no later synthesis frame can affect them. No full-utterance normalization.
    This measures the extractor alone, excluding PipeWire and resampling.
    """
    reset_cache(model)
    if hasattr(core, "reset"):
        core.reset()
    prime_start, prime_cpu_start = time.perf_counter(), time.process_time()
    if prime_silence:
        # Synthetic past only; no microphone samples or future input. Keep the
        # 128-frame history identical across FP32 chunk partitions.
        for first in range(0, 128, frames):
            core(model, torch.zeros(1, 257, min(frames, 128 - first),
                                    dtype=torch.complex64), cues)
    prime_wall, prime_cpu = time.perf_counter() - prime_start, time.process_time() - prime_cpu_start
    win, hop = model.sep_model.win, model.sep_model.stride
    assert win == 512 and hop == 128 and samples.shape[0] == 1
    window = torch.hann_window(win)
    accum, weight = torch.zeros(win), torch.zeros(win)
    output, costs, cpu_costs, readiness = [], [], [], []
    length = samples.shape[-1]
    # Reflect only the startup and actual EOF boundaries, as upstream STFT does.
    count = length // hop + 1
    for first in range(0, count, frames):
        n = min(frames, count - first)
        start, cpu_start = time.perf_counter(), time.process_time()
        centers = torch.arange(first, first + n)[:, None] * hop
        indices = centers + torch.arange(win)[None, :] - win // 2
        indices = torch.where(indices < 0, -indices, indices)
        indices = torch.where(indices >= length, 2 * length - 2 - indices, indices)
        frame_audio = samples[0, indices] * window
        spec = torch.fft.rfft(frame_audio).T.unsqueeze(0)
        separated = core(model, spec, cues)[0, 0].T
        synthesis = torch.fft.irfft(separated, n=win) * window
        for j in range(n):
            accum += synthesis[j]
            weight += window.square()
            source_start = (first + j) * hop - win // 2
            if source_start >= 0:
                take = min(hop, length - source_start)
                if take > 0:
                    output.append((accum[:take] / weight[:take].clamp_min(1e-12)).clone())
                    input_ready = min(length, (first + n - 1) * hop + win // 2)
                    readiness.append((input_ready - source_start) / 16)
            accum = torch.cat((accum[hop:], torch.zeros(hop)))
            weight = torch.cat((weight[hop:], torch.zeros(hop)))
        costs.append((time.perf_counter() - start) * 1000)
        cpu_costs.append(time.process_time() - cpu_start)
    # At EOF no future frames will arrive; flush the synthesis tail.
    remainder = length - sum(x.numel() for x in output)
    if remainder:
        output.append(accum[:remainder] / weight[:remainder].clamp_min(1e-12))
    result = torch.cat(output)
    assert len(result) == length and result.isfinite().all()
    return result.numpy(), {
        "wall_rtf": sum(costs) / 1000 / (length / 16000),
        "process_cpu_rtf": sum(cpu_costs) / (length / 16000),
        "chunk_ms": frames * 8,
        "chunk_cost_ms_mean": float(np.mean(costs)),
        "chunk_cost_ms_p95": float(np.percentile(costs, 95)),
        "chunk_cost_ms_max": max(costs),
        "input_readiness_ms_max": max(readiness),
        "priming_silence_ms": 1024 if prime_silence else 0,
        "priming_wall_ms": prime_wall * 1000 if prime_silence else 0,
        "priming_cpu_ms": prime_cpu * 1000 if prime_silence else 0,
        "note": "Extractor only. Readiness excludes computation, resampling, worker and host buffering.",
    }


def check(model, mixture, enroll, cues, frames, core, prime_silence=False):
    """One runnable regression check for equivalence, state reset and causality."""
    small = mixture[:, :8192]
    torch.manual_seed(42)
    reset_cache(model)
    ordinary = model(small, enroll)[0, 0].numpy()
    cached, _ = extract(model, small, cues, frames)
    error = float(np.max(np.abs(ordinary - cached)))
    assert error < 2e-4, f"Upstream/streaming parity failed: {error}"
    if prime_silence:
        cached, _ = extract(model, small, cues, frames, prime_silence=True)
    changed = small.clone()
    changed[:, 4096:] = torch.randn_like(changed[:, 4096:]) * 0.2
    future, _ = extract(model, changed, cues, frames, prime_silence=prime_silence)
    # Chunk-wide INT8 activation scales may use every frame in the current
    # chunk. Only output whose declared input deadline precedes the change
    # must remain identical; the maximum readiness is (frames-1)*128 + 512.
    emitted = 4096 - ((frames - 1) * 128 + 512)
    causal_error = float(np.max(np.abs(cached[:emitted] - future[:emitted])))
    assert causal_error < 2e-5, f"Future input affects emitted audio: {causal_error}"
    partitioned, _ = extract(model, small, cues, 3, prime_silence=prime_silence)
    partition_error = float(np.max(np.abs(cached - partitioned)))
    assert partition_error < 2e-4, f"Chunk boundary/state parity failed: {partition_error}"
    output, _ = extract(model, small, cues, frames, core, prime_silence)
    reset_output, _ = extract(model, small, cues, frames, core, prime_silence)
    reset_error = float(np.max(np.abs(output - reset_output)))
    assert reset_error < 1e-6, f"State leaked across streams: {reset_error}"
    backend_future, _ = extract(model, changed, cues, frames, core, prime_silence)
    backend_causal_error = float(np.max(np.abs(output[:emitted] - backend_future[:emitted])))
    assert backend_causal_error < 2e-5, f"Backend reads future audio: {backend_causal_error}"
    return {"upstream_max_abs_error": error, "future_input_max_abs_error": causal_error,
            "chunk_partition_max_abs_error": partition_error, "reset_max_abs_error": reset_error,
            "backend_future_input_max_abs_error": backend_causal_error,
            "backend_vs_torch_rmse": float(np.sqrt(np.mean((output - cached) ** 2)))}


def metrics(output, mixture, target, competitor, intervals):
    result = []
    for start, end in intervals:
        a, b = round(start * 16000), round(end * 16000)
        y, t, c, m = output[a:b], target[a:b], competitor[a:b], mixture[a:b]
        entry = {"seconds": [start, end]}
        if float(np.dot(t, t)) < 1e-8:
            key = "competitor_only_reduction_db" if float(np.dot(c, c)) >= 1e-8 else "noise_only_reduction_db"
            entry[key] = float(10 * np.log10((np.dot(m, m) + 1e-12) / (np.dot(y, y) + 1e-12)))
        else:
            coefficients = np.linalg.lstsq(np.stack((t, c), 1), y, rcond=None)[0]
            entry["target_projection_gain"] = float(coefficients[0])
            entry["competitor_projection_gain"] = float(coefficients[1])
            if float(np.dot(c, c)) >= 1e-8:
                entry["target_to_competitor_projection_improvement_db"] = float(20 * np.log10((abs(coefficients[0]) + 1e-12) / (abs(coefficients[1]) + 1e-12)))
            else:
                entry["competitor_absent"] = True
            def si_sdr(value):
                value, desired = value - value.mean(), t - t.mean()
                projection = desired * np.dot(value, desired) / (np.dot(desired, desired) + 1e-12)
                return float(10 * np.log10((np.dot(projection, projection) + 1e-12) / (np.dot(value - projection, value - projection) + 1e-12)))
            entry["output_si_sdr_db"] = si_sdr(y)
            entry["mixture_si_sdr_db"] = si_sdr(m)
        result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--blocks", type=int, choices=(4, 6), default=6,
                        help="Four-block depth ablation; six is the trained architecture")
    parser.add_argument("--prime-silence", action="store_true",
                        help="Warm recurrent state with 1.024 s of synthetic past silence")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--backend", choices=("torch", "ort"), default="torch")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--static-int8", action="store_true")
    parser.add_argument("--mixture", type=Path, default=ROOT / "tests/fixtures/voice-evaluation.flac")
    parser.add_argument("--enrollment", type=Path, default=ROOT / "tests/fixtures/voice-enrollment.flac")
    parser.add_argument("--case", default="librispeech-short")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--target", type=Path)
    parser.add_argument("--competitor", type=Path)
    args = parser.parse_args()
    assert 1 <= args.frames <= 16 and 1 <= args.threads <= 12
    assert args.case.replace("-", "").replace("_", "").isalnum()
    assert not args.static_int8 or (args.backend == "ort" and args.int8)
    assert not args.int8 or args.backend == "ort", "INT8 pilot uses ONNX Runtime"
    assert not args.prime_silence or args.frames == 8, "Priming pilot uses eight-frame chunks"
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    with torch.inference_mode():
        model = load_model(args.blocks)
        mixture = audio(args.mixture)
        enroll = audio(args.enrollment)
        cues = reference(model, enroll)
        core = spectrum
        if args.backend == "ort":
            core = ort_core(model, cues, args.frames, args.threads, args.export, args.int8, args.static_int8)
        enable_cache(model)
        checks = check(model, mixture, enroll, cues, args.frames, core, args.prime_silence) if args.check else None
        output, performance = extract(model, mixture, cues, args.frames, core, args.prime_silence)
        performance.update({"model": MODEL, "backend": args.backend, "threads": args.threads,
                            "separation_blocks": args.blocks, "depth_ablation": args.blocks != 6,
                            "partial_int8": args.int8, "static_convolution_int8": args.static_int8,
                            "checkpoint_sha256": WEIGHTS_SHA,
                            "script_sha256": digest(Path(__file__)),
                            "input_sha256": digest(args.mixture), "enrollment_sha256": digest(args.enrollment),
                            "source_commit": SOURCE, "speaker_source_commit": SPEAKER_SOURCE,
                            "torch_version": torch.__version__, "torchaudio_version": torchaudio.__version__,
                            "machine": platform.platform(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
                            "command": sys.argv,
                            "stream_can_keep_up": performance["wall_rtf"] < 1,
                            "extractor_cpu_budget_pass": performance["process_cpu_rtf"] <= 0.8,
                            "checks": checks})
        if args.backend == "ort":
            performance["graph_sha256"] = digest(core.graph)
        if args.target or args.competitor:
            assert args.target and args.competitor
            target, competitor = audio(args.target)[0].numpy(), audio(args.competitor)[0].numpy()
            assert len(target) == len(competitor) == len(output)
            assert len(output) == 160000, "Short fixture score intervals require ten seconds"
            performance["acoustics"] = metrics(output, mixture[0].numpy(), target, competitor,
                                                [(0, 3), (3.2, 6.8), (7, 10)])
        precision = "conv-int8" if args.static_int8 else ("int8" if args.int8 else "fp32")
        tag = f"{args.case}-{MODEL}-{args.backend}-{precision}-t{args.threads}-f{args.frames}"
        if args.blocks != 6:
            tag += f"-b{args.blocks}"
        if args.prime_silence:
            tag += "-primed"
        sf.write(WORK / f"{tag}.wav", output, 16000, subtype="FLOAT")
        (WORK / f"{tag}.json").write_text(json.dumps(performance, indent=2) + "\n")
        print(json.dumps(performance, indent=2), flush=True)


if __name__ == "__main__":
    main()
