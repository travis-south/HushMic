#!/usr/bin/env python3
"""Offline TSE -> bounded sinc upsampling -> actual native INT8 DPDFNet."""
import argparse
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

# Keep provenance tied to the code loaded at startup, not later file edits.
RUNNER_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
EXTRACTOR_PATH = Path(__file__).with_name("prototype-target-extraction.py")
EXTRACTOR_SOURCE = EXTRACTOR_PATH.read_bytes()
EXTRACTOR_SHA = hashlib.sha256(EXTRACTOR_SOURCE).hexdigest()

import numpy as np
import soundfile as sf
import torch
import torchaudio

spec = importlib.util.spec_from_file_location("extraction", EXTRACTOR_PATH)
tse = importlib.util.module_from_spec(spec)
exec(compile(EXTRACTOR_SOURCE, str(EXTRACTOR_PATH), "exec"), tse.__dict__)


def openvino_core(model, cues, frames):
    # Reuse the installed isolated SDK; no dependency or app runtime changes.
    sys.path.append(str(tse.ROOT / "target/openvino-bench/venv/lib/python3.12/site-packages"))
    import openvino as ov
    suffix = f"-b{len(model.sep_model.separator.separation)}" if len(model.sep_model.separator.separation) != 6 else ""
    graph = tse.WORK / f"bsrnn-causal-f{frames}{suffix}-conv-int8-int8.onnx"
    compiled = ov.Core().compile_model(str(graph), "CPU", {
        "PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": 1,
        "NUM_STREAMS": 1, "INFERENCE_PRECISION_HINT": "f32", "PERF_COUNT": True})
    request = compiled.create_infer_request()
    names = [f"state_{i}" for i in range(2 * len(model.sep_model.separator.separation))]
    cache = []

    def reset():
        nonlocal cache
        cache = [np.zeros(compiled.input(name).shape, np.float32) for name in names]

    def run(_, spectrum, __):
        nonlocal cache
        count = spectrum.shape[-1]
        ri = torch.nn.functional.pad(torch.stack((spectrum.real, spectrum.imag), 1), (0, frames - count))
        values = request.infer({"spectrum": ri.numpy(), "reference": cues["spkemb"].numpy(), **dict(zip(names, cache))})
        cache = [values[compiled.output(f"next_{i}")].copy() for i in range(len(names))]
        ri = torch.from_numpy(values[compiled.output("extracted")][..., :count].copy())
        return torch.complex(ri[:, 0], ri[:, 1]).unsqueeze(1)

    reset()
    run.reset, run.graph, run.version = reset, graph, ov.get_version()
    run.profile = lambda: [{"node": p.node_name, "type": p.node_type, "execution": p.exec_type}
                           for p in request.get_profiling_info() if p.real_time.total_seconds() > 0]
    return run


class Sinc:
    """Retain FIR history and bounded lookahead for 16/48 kHz conversion."""
    def __init__(self, input_rate, output_rate):
        self.resample = torchaudio.transforms.Resample(input_rate, output_rate)
        self.stride = input_rate // self.resample.gcd
        self.buffer = torch.zeros(self.resample.width)
        self.fed = self.emitted = 0

    def push(self, samples, eof=False):
        self.fed += len(samples)
        self.buffer = torch.cat((self.buffer, torch.from_numpy(samples)))
        if eof:
            self.buffer = torch.nn.functional.pad(self.buffer, (0, self.resample.width + self.stride))
        count = min(max(0, (len(self.buffer) - self.resample.kernel.shape[-1]) // self.stride + 1),
                    (self.fed + self.stride - 1) // self.stride - self.emitted)
        if not count:
            return np.empty(0, np.float32)
        output = torch.nn.functional.conv1d(self.buffer[None, None], self.resample.kernel, stride=self.stride)
        output = output[0, :, :count].T.reshape(-1).numpy().copy()
        if eof:
            output = output[:(self.fed * self.resample.new_freq + self.resample.orig_freq - 1) // self.resample.orig_freq
                            - self.emitted * self.resample.kernel.shape[0]]
        self.buffer = self.buffer[count * self.stride:]
        self.emitted += count
        return output


class Native:
    def __init__(self, executable, model):
        self.process = subprocess.Popen([str(executable), str(model)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.latency = struct.unpack("<I", self.read(4))[0]
        assert self.latency == 2400
        self.cpu_start = self.cpu()

    def read(self, size):
        data = bytearray()
        while len(data) < size:
            block = self.process.stdout.read(size - len(data))
            if not block:
                raise RuntimeError("native bridge exited during packet")
            data.extend(block)
        return bytes(data)

    def cpu(self):
        fields = Path(f"/proc/{self.process.pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")

    def push(self, samples):
        assert len(samples) <= 9600 and np.isfinite(samples).all()
        self.process.stdin.write(struct.pack("<I", len(samples)) + samples.astype("<f4", copy=False).tobytes())
        self.process.stdin.flush()
        count = struct.unpack("<I", self.read(4))[0]
        assert count <= 12480
        output = np.frombuffer(self.read(count * 4), dtype="<f4").copy()
        assert np.isfinite(output).all()
        return output

    def close(self):
        self.process.stdin.close()
        self.process.stdout.close()
        assert self.process.wait(timeout=10) == 0


def check(executable, model):
    for packet in (struct.pack("<I", 9601), struct.pack("<If", 1, float("nan"))):
        rejected = subprocess.run([str(executable), str(model)], input=packet, capture_output=True, timeout=10)
        assert rejected.returncode != 0 and len(rejected.stdout) == 4
    torch.manual_seed(42)
    samples = torch.randn(6001) * 0.03
    errors = []
    for input_rate, output_rate in [(48000, 16000), (16000, 48000)]:
        sampler = Sinc(input_rate, output_rate)
        outputs = [sampler.push(part.numpy()) for part in samples.split(137)]
        outputs.append(sampler.push(np.empty(0, np.float32), eof=True))
        actual = np.concatenate(outputs)
        expected = sampler.resample(samples).numpy()
        assert len(actual) == len(expected) and np.max(np.abs(actual - expected)) < 1e-6
        errors.append(float(np.max(np.abs(actual - expected))))
    outputs = []
    for size in (480, 1383):
        native = Native(executable, model)
        try:
            result = [native.push(actual[i:i + size]) for i in range(0, len(actual), size)]
            result.append(native.push(np.empty(0, np.float32)))
            outputs.append(np.concatenate(result)[native.latency:native.latency + len(actual)])
        finally:
            native.close()
    assert len(outputs[0]) == len(actual) and np.array_equal(*outputs)
    return {"resampler_max_abs_error": max(errors),
            "native_chunk_partition_max_abs_error": float(np.max(np.abs(outputs[0] - outputs[1])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixture", type=Path, default=tse.ROOT / "target/voice-reference-evaluation/scenario.wav")
    parser.add_argument("--enrollment", type=Path, default=tse.ROOT / "target/voice-reference-evaluation/enrollment.wav")
    parser.add_argument("--model", type=Path, default=tse.ROOT / "assets/models/dpdfnet8_48khz_hr.onnx")
    parser.add_argument("--bridge", type=Path, default=tse.ROOT / "target/release/examples/prototype_pipeline")
    parser.add_argument("--case", default="combined-public-long")
    parser.add_argument("--frames", type=int, choices=(4, 8), default=8)
    parser.add_argument("--blocks", type=int, choices=(4, 5, 6), default=6)
    parser.add_argument("--backend", choices=("ort", "openvino"), default="ort")
    parser.add_argument("--ort-threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--no-spin", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--paced", action="store_true", help="Feed at real input cadence rather than saturating a core")
    parser.add_argument("--extractor-only", action="store_true", help="Use extraction as the speech processor; omit DPDFNet")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    assert args.case.replace("-", "").replace("_", "").isalnum()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with torch.inference_mode():
        checks = check(args.bridge, args.model) if args.check else None
        model = tse.load_model(args.blocks)
        raw, input_rate = sf.read(args.mixture, dtype="float32", always_2d=True)
        assert input_rate == 48000 and raw.shape[1] == 1 and np.isfinite(raw).all()
        raw = raw[:, 0].copy()
        mixture = torch.zeros(1, (len(raw) + 2) // 3)
        cues = tse.reference(model, tse.audio(args.enrollment))
        if args.backend == "ort" or args.export:
            core = tse.ort_core(model, cues, args.frames, args.ort_threads, args.export, True, True, not args.no_spin)
        if args.backend == "openvino":
            assert args.ort_threads == 1 and not args.no_spin
            core = openvino_core(model, cues, args.frames)
        tse.enable_cache(model)
        if args.check:
            # Includes the newly shaped INT8 graph, not just the resamplers.
            checks["extractor"] = tse.check(model, tse.audio(args.mixture), tse.audio(args.enrollment),
                                            cues, args.frames, core, True)
        downsample, sampler = Sinc(48000, 16000), Sinc(16000, 48000)
        native = None if args.extractor_only else Native(args.bridge, args.model)
        latency = native.latency if native else 0
        outputs = []
        read, filled = 0, 0

        def source(ready):
            nonlocal read, filled
            end = min(len(raw), ready * 3 + downsample.resample.width)
            if end > read:
                converted = downsample.push(raw[read:end], eof=end == len(raw))
                mixture[0, filled:filled + len(converted)] = torch.from_numpy(converted)
                read, filled = end, filled + len(converted)
            assert filled >= ready

        def sink(samples):
            upsampled = sampler.push(samples)
            outputs.append(native.push(upsampled) if native and len(upsampled) else upsampled)

        try:
            extracted, performance = tse.extract(model, mixture, cues, args.frames, core, True, sink, source, args.paced)
            flush_wall, flush_cpu = time.perf_counter(), time.process_time()
            tail = sampler.push(np.empty(0, np.float32), eof=True)
            outputs.append(native.push(tail) if native else tail)
            outputs.append(native.push(np.empty(0, np.float32)) if native else np.empty(0, np.float32))
            flush_wall = time.perf_counter() - flush_wall
            flush_cpu = time.process_time() - flush_cpu
            native_cpu = native.cpu() - native.cpu_start if native else 0
        finally:
            if native:
                native.close()
        cleaned = np.concatenate(outputs)[latency:latency + len(extracted) * 3]
        assert len(cleaned) == len(extracted) * 3 and np.isfinite(cleaned).all()
        # ponytail: offline release/deadline simulation; actual PipeWire worker
        # and host scheduling remain a separate integration gate.
        events = performance.pop("chunks")
        flush_finish = events[-1]["actual_finish_ms"] + flush_wall * 1000
        events.extend([{"input_ready_samples": len(extracted), "wall_ms": flush_wall * 1000, "actual_finish_ms": flush_finish},
                       {"input_ready_samples": len(extracted), "wall_ms": 0, "actual_finish_ms": flush_finish}])
        finish, emitted, delays, actual_delays, trace = 0.0, 0, [], [], []
        input_lookahead = downsample.resample.width / input_rate
        for event, output in zip(events, outputs, strict=True):
            ready_time = min(len(raw) / input_rate, event["input_ready_samples"] / 16000 + input_lookahead)
            finish = max(finish, ready_time) + event["wall_ms"] / 1000
            first_real = max(emitted, latency)
            if emitted + len(output) > first_real and first_real - latency < len(cleaned):
                delays.append((finish - (first_real - latency) / 48000) * 1000)
                actual_delays.append(event["actual_finish_ms"] - (first_real - latency) / 48)
            trace.append({**event, "output_samples": len(output), "simulated_finish_ms": finish * 1000})
            emitted += len(output)
        duration = len(extracted) / 16000
        performance.update({"checks": checks, "native_engine": "native-int8" if native else None,
                            "denoiser": str(args.model) if native else None, "model": tse.MODEL,
                            "model_sha256": tse.digest(args.model) if native else None, "graph_sha256": tse.digest(core.graph),
                            "checkpoint_sha256": tse.WEIGHTS_SHA, "source_commit": tse.SOURCE,
                            "torch_version": torch.__version__, "torchaudio_version": torchaudio.__version__,
                            "script_sha256": RUNNER_SHA, "extractor_script_sha256": EXTRACTOR_SHA,
                            "source_hash_capture": "startup; extractor executed from captured bytes",
                            "bridge_sha256": tse.digest(args.bridge) if native else None, "command": sys.argv,
                            "input_sha256": tse.digest(args.mixture), "enrollment_sha256": tse.digest(args.enrollment),
                            "cpu_affinity": sorted(os.sched_getaffinity(0)),
                            "frames": args.frames,
                            "backend": args.backend,
                            "recurrent_precision": "ONNX dynamic INT8; verify runtime profile" if args.backend == "openvino" else "int8",
                            "ort_threads": args.ort_threads,
                            "ort_spinning": not args.no_spin,
                            "separation_blocks": args.blocks, "depth_ablation": args.blocks != 6,
                            "paced": args.paced,
                            "extractor_only": args.extractor_only,
                            "combined_cpu_rtf": performance["stream_process_cpu_rtf"] + (native_cpu + flush_cpu) / duration,
                            "combined_wall_rtf": performance["wall_rtf"] + flush_wall / duration,
                            "native_cpu_rtf": native_cpu / duration, "native_cpu_resolution_seconds": 1 / os.sysconf("SC_CLK_TCK"),
                            "output_availability_ms_max": max(delays), "output_availability_ms_p95": float(np.percentile(delays, 95)),
                            "output_availability_over_190ms_packets": sum(delay > 190 for delay in delays),
                            "paced_actual_availability_ms_max": max(actual_delays) if args.paced else None,
                            "input_resample_lookahead_ms": input_lookahead * 1000,
                            "note": "Bounded 48->16 kHz resampling, TSE, bounded 16->48 kHz resampling, and optional actual native denoising measured together. Startup and TSE priming excluded. Offline worker scheduling simulation; 190ms allows one 10ms host quantum within 200ms. Live worker and PipeWire untested."})
        if args.backend == "openvino":
            performance.update({"openvino_version": core.version, "openvino_execution": core.profile()})
        else:
            performance["onnxruntime_version"] = sys.modules["onnxruntime"].__version__
        sf.write(tse.WORK / f"{args.case}-extracted.wav", extracted, 16000, subtype="FLOAT")
        sf.write(tse.WORK / f"{args.case}.wav", cleaned, 48000, subtype="FLOAT")
        (tse.WORK / f"{args.case}.json").write_text(json.dumps(performance, indent=2) + "\n")
        (tse.WORK / f"{args.case}-trace.json").write_text(json.dumps(trace) + "\n")
        print(json.dumps(performance, indent=2), flush=True)


if __name__ == "__main__":
    main()
