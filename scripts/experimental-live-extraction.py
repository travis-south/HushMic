#!/usr/bin/env python3
"""Local-only INT8 extraction bridge. Reuses the measured four-block prototype."""
import argparse
from collections import deque
import importlib.util
from pathlib import Path
import struct
import sys

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("pipeline", Path(__file__).with_name("prototype-extraction-pipeline.py"))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
DELAY = 4320  # 90 ms extraction delay; denoiser and worker each add 50 ms.


def read_exact(stream, count):
    value = bytearray()
    while len(value) < count:
        part = stream.read(count - len(value))
        if not part:
            raise EOFError("Extraction input closed")
        value.extend(part)
    return bytes(value)


class Stream:
    def __init__(self, model, cues):
        self.model, self.cues = model, cues
        self.core = p.tse.ort_core(model, cues, 4, 1, False, True, True)
        self.window = torch.hann_window(512)
        for _ in range(32):
            self.core(self.model, torch.zeros(1, 257, 4, dtype=torch.complex64), self.cues)
        self.primed = self.core.snapshot()
        self.reset()

    def reset(self):
        self.core.restore(self.primed)
        self.down, self.up = p.Sinc(48000, 16000), p.Sinc(16000, 48000)
        self.buffer = torch.empty(0)
        self.base = self.frame = 0
        self.accum, self.weight = torch.zeros(512), torch.zeros(512)
        self.output = deque([0.0] * DELAY)
        self.raw = deque([0.0] * DELAY)
        self.mode = 1

    def push(self, samples, mode=1):
        assert len(samples) == 480 and np.isfinite(samples).all() and mode in (1, 2, 3)
        if mode != self.mode:
            self.reset()  # Clear recurrent/audio history across bypass and mute.
            self.mode = mode
        self.raw.extend(samples)
        raw = np.array([self.raw.popleft() for _ in range(480)], np.float32)
        if mode != 1:
            return raw if mode == 2 else np.zeros(480, np.float32)
        self.buffer = torch.cat((self.buffer, torch.from_numpy(self.down.push(samples))))
        while self.base + len(self.buffer) >= (self.frame + 3) * 128 + 256:
            centers = torch.arange(self.frame, self.frame + 4)[:, None] * 128
            indices = centers + torch.arange(512)[None, :] - 256
            indices = torch.where(indices < 0, -indices, indices) - self.base
            spectrum = torch.fft.rfft(self.buffer[indices] * self.window).T[None]
            separated = self.core(self.model, spectrum, self.cues)[0, 0].T
            synthesis = torch.fft.irfft(separated, n=512) * self.window
            emitted = []
            for j in range(4):
                self.accum += synthesis[j]
                self.weight += self.window.square()
                if self.frame + j >= 2:
                    emitted.append((self.accum[:128] / self.weight[:128].clamp_min(1e-12)).clone())
                self.accum = torch.cat((self.accum[128:], torch.zeros(128)))
                self.weight = torch.cat((self.weight[128:], torch.zeros(128)))
            self.output.extend(self.up.push(torch.cat(emitted).numpy()))
            self.frame += 4
            keep = max(0, self.frame * 128 - 256)
            self.buffer = self.buffer[keep - self.base:].clone()
            self.base = keep
        assert 480 <= len(self.output) <= DELAY + 1536, "Extraction stream lost alignment"
        output = np.array([self.output.popleft() for _ in range(480)], np.float32)
        assert np.isfinite(output).all()
        return output


def check(model):
    cues = p.tse.reference(model, p.tse.audio(p.tse.ROOT / "tests/fixtures/voice-enrollment.flac"))
    samples = p.sf.read(p.tse.WORK / "quiet-alone-mixture.wav", dtype="float32")[0][:96000]
    stream = Stream(model, cues)
    actual = np.concatenate([stream.push(x) for x in samples.reshape(-1, 480)])
    down, up = p.Sinc(48000, 16000), p.Sinc(16000, 48000)
    converted = down.push(samples, eof=True)
    core = p.tse.ort_core(model, cues, 4, 1, False, True, True)
    p.tse.enable_cache(model)
    outputs = []
    extracted, _ = p.tse.extract(model, torch.from_numpy(converted)[None], cues, 4, core, True,
                                lambda x: outputs.append(up.push(x)))
    expected = np.concatenate(outputs)
    error = np.max(np.abs(actual[DELAY:] - expected[:len(actual) - DELAY]))
    assert error < 2e-5, error
    stream.reset()
    repeat = np.concatenate([stream.push(x) for x in samples.reshape(-1, 480)])
    assert np.array_equal(actual, repeat)
    for mode in (2, 3):
        stream.reset()
        output = np.concatenate([stream.push(x, mode) for x in samples.reshape(-1, 480)])
        expected = np.concatenate((np.zeros(DELAY), samples))[:len(samples)] if mode == 2 else np.zeros(len(samples))
        assert np.array_equal(output, expected)
    print(f"Live/offline parity max error {error}; reset, bypass and mute passed", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enroll", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with torch.inference_mode():
        model = p.tse.load_model(4)
        if args.check:
            check(model)
            return
        if args.enroll:
            raw = np.frombuffer(read_exact(sys.stdin.buffer, 20 * 48000 * 4), dtype="<f4").copy()
            assert np.isfinite(raw).all() and np.sqrt(np.mean(raw ** 2)) >= 1e-4, "Recording is silent or invalid"
            assert np.mean(np.abs(raw) >= 0.999) < 0.01, "Recording is clipped; lower microphone gain"
            enroll = p.torchaudio.functional.resample(torch.from_numpy(raw)[None], 48000, 16000)
            cues = p.tse.reference(model, enroll)
            raw.fill(0)
            enroll.zero_()
            sys.stdout.buffer.write(cues["spkemb"].numpy().astype("<f4").tobytes())
            return
        reference = np.frombuffer(args.reference.read_bytes(), dtype="<f4").copy()
        assert reference.shape == (192,) and np.isfinite(reference).all() and np.any(reference)
        stream = Stream(model, {"spkemb": torch.from_numpy(reference)[None]})
        sys.stdout.buffer.write(b"HMTSE001" + struct.pack("<I", DELAY))
        sys.stdout.buffer.flush()
        while True:
            try:
                mode = struct.unpack("<I", read_exact(sys.stdin.buffer, 4))[0]
            except EOFError:
                return
            if mode == 0:
                stream.reset()
                sys.stdout.buffer.write(struct.pack("<I", 0))
            else:
                samples = np.frombuffer(read_exact(sys.stdin.buffer, 480 * 4), dtype="<f4").copy()
                output = stream.push(samples, mode)
                sys.stdout.buffer.write(struct.pack("<I", 480) + output.astype("<f4").tobytes())
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
