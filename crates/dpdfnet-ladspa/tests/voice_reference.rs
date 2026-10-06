//! Actual callback/worker test of the experimental stock INT8 candidate.
mod common;
use dpdfnet_ladspa::{get_ladspa_descriptor, PLUGIN_LATENCY_SAMPLES};
use hushmic_denoiser::voice::{enroll, write_reference, HOLD_SAMPLES};
use ladspa::{Plugin, PortConnection, PortData};
use std::{
    cell::RefCell,
    time::{Duration, Instant},
};
const Q: usize = 480;
fn cycle(plugin: &mut Box<dyn Plugin + Send>, input: &[f32], db: f32, mode: f32) -> Vec<f32> {
    let d = get_ladspa_descriptor(0).unwrap();
    let mut output = vec![0.0; input.len()];
    let ports = [
        PortConnection {
            port: d.ports[0],
            data: PortData::AudioInput(input),
        },
        PortConnection {
            port: d.ports[1],
            data: PortData::AudioOutput(RefCell::new(&mut output)),
        },
        PortConnection {
            port: d.ports[2],
            data: PortData::ControlInput(&db),
        },
        PortConnection {
            port: d.ports[3],
            data: PortData::ControlInput(&mode),
        },
    ];
    plugin.run(input.len(), &ports.iter().collect::<Vec<_>>());
    output
}
fn plugin() -> Box<dyn Plugin + Send> {
    let d = get_ladspa_descriptor(0).unwrap();
    let mut p = (d.new)(&d, 48000);
    p.activate();
    p
}
fn stream(p: &mut Box<dyn Plugin + Send>, input: &[f32], db: f32, mode: f32) -> Vec<f32> {
    let mut output = Vec::new();
    let mut next = Instant::now();
    for hop in input.as_chunks::<Q>().0 {
        output.extend(cycle(p, hop, db, mode));
        next += Duration::from_millis(10);
        std::thread::sleep(next.saturating_duration_since(Instant::now()));
    }
    assert!(output.iter().all(|v| v.is_finite()));
    output
}
fn flac(name: &str) -> Vec<f32> {
    let mut reader =
        claxon::FlacReader::open(common::repo_root().join("tests/fixtures").join(name)).unwrap();
    assert_eq!(reader.streaminfo().sample_rate, 48000);
    assert_eq!(reader.streaminfo().channels, 1);
    reader
        .samples()
        .map(|v| v.unwrap() as f32 / 32768.0)
        .collect()
}
fn energy(samples: &[f32]) -> f64 {
    samples.iter().map(|v| (*v as f64).powi(2)).sum()
}
#[test]
fn real_reference_gate_preserves_alignment_modes_and_failure_fallback() {
    let (Some(model), Some(runtime)) = (
        common::model_path("dpdfnet2_48khz_hr.onnx"),
        common::runtime_path(),
    ) else {
        return;
    };
    std::env::set_var("HUSHMIC_MODEL_PATH", model);
    std::env::set_var("ORT_DYLIB_PATH", &runtime);
    hushmic_denoiser::init_runtime(runtime).unwrap();
    let reference = enroll(
        &flac("voice-enrollment.flac"),
        &common::model_path("voice-reference-encoder.onnx").expect("committed encoder"),
    )
    .unwrap();
    let directory =
        std::env::temp_dir().join(format!("hushmic-plugin-voice-{}", std::process::id()));
    let path = directory.join("reference");
    write_reference(&path, &reference).unwrap();
    std::env::set_var("HUSHMIC_VOICE_REFERENCE", &path);
    std::env::set_var(
        "HUSHMIC_VOICE_MODEL_PATH",
        common::model_path("voice-presence-int8.onnx").expect("committed detector"),
    );
    let mut input = flac("voice-evaluation.flac");
    input.resize(input.len() + 30 * Q, 0.0);
    std::env::set_var("HUSHMIC_VOICE_REQUESTED", "0");
    let mut ordinary = plugin();
    let baseline = stream(&mut ordinary, &input, 0.0, 0.0);
    drop(ordinary);
    std::env::set_var("HUSHMIC_VOICE_REQUESTED", "1");
    let mut protected = plugin();
    let gated = stream(&mut protected, &input, 0.0, 0.0);
    let delay = PLUGIN_LATENCY_SAMPLES + HOLD_SAMPLES;
    assert_eq!(delay, 9120);
    assert!(delay <= 9600);
    let mut ratios = Vec::new();
    for (start, end) in [(0, 3), (3, 7), (7, 10)] {
        let b =
            &baseline[start * 48000 + PLUGIN_LATENCY_SAMPLES..end * 48000 + PLUGIN_LATENCY_SAMPLES];
        let g = &gated[start * 48000 + delay..end * 48000 + delay];
        let ratio = energy(g) / energy(b);
        ratios.push(ratio);
        // Every emitted gate sample is either retained final mixture or zero.
        let differences: Vec<_> = g
            .iter()
            .zip(b)
            .enumerate()
            .filter(|(_, (g, b))| g.abs() >= 1e-7 && (*g - *b).abs() >= 1e-6)
            .collect();
        eprintln!(
            "gate differences interval{start}: {} first {:?}",
            differences.len(),
            differences.first()
        );
        assert!(differences.is_empty());
    }
    eprintln!("actual plugin stock trial energy ratios target+C {:.6}, competitor-only {:.6}, resumed target+C {:.6}; competitor reduction {:.3} dB",ratios[0],ratios[1],ratios[2],-10.0*ratios[1].log10());
    assert!(
        ratios[0] > 0.5 && ratios[2] > 0.5,
        "enrolled speech was mostly lost: {ratios:?}"
    );
    assert!(
        ratios[1] < 0.9,
        "final strength mixture bypassed speaker gate: {ratios:?}"
    );
    // Reset discards the old temporal history, keeps the fixed enrolled identity.
    protected.activate();
    let silence = stream(&mut protected, &vec![0.0; 25 * Q], 60.0, 0.0);
    assert!(silence.iter().all(|v| v.abs() < 1e-6));
    let muted = stream(&mut protected, &input[..25 * Q], 0.0, 2.0);
    assert!(muted.iter().all(|v| v.abs() < 1e-6));
    // Bypass and detector failure have the same fixed 190 ms impulse delay.
    let mut impulse = vec![0.0; 40 * Q];
    let at = 10 * Q + 7;
    impulse[at] = 1.0;
    protected.activate();
    let bypass = stream(&mut protected, &impulse, 0.0, 1.0);
    assert_eq!(bypass.iter().position(|v| v.abs() > 0.5), Some(at + delay));
    drop(protected);
    // Non-hop host cycles retain alignment and the 200 ms supported ceiling.
    for quantum in [512usize, 960] {
        let mut p = plugin();
        let at = 10 * quantum + 7;
        let mut input = vec![0.0; 40 * quantum];
        input[at] = 1.0;
        let mut output = Vec::new();
        let mut next = Instant::now();
        for chunk in input.chunks_exact(quantum) {
            output.extend(cycle(&mut p, chunk, 0.0, 1.0));
            next += Duration::from_secs_f64(quantum as f64 / 48000.0);
            std::thread::sleep(next.saturating_duration_since(Instant::now()));
        }
        let latency = hushmic_denoiser::LATENCY_SAMPLES
            + HOLD_SAMPLES
            + dpdfnet_ladspa::required_lead(quantum);
        assert_eq!(latency, 9600);
        assert_eq!(
            output.iter().position(|v| v.abs() > 0.5),
            Some(at + latency)
        );
    }
    // Benchmark the same complete worker engine, not just its neural graph.
    use dpdfnet_ladspa::{HopEngine, Tier};
    for (model, tier) in [
        ("dpdfnet2_48khz_hr.onnx", Tier::Light),
        ("dpdfnet8_48khz_hr.onnx", Tier::Quality),
    ] {
        let denoiser = dpdfnet_ladspa::load_denoiser(&common::model_path(model).unwrap()).unwrap();
        let policy: Box<dyn dpdfnet_ladspa::Policy> = Box::new(dpdfnet_ladspa::Pinned(tier));
        let (quality, light) = if tier == Tier::Quality {
            (Some(denoiser), None)
        } else {
            (None, Some(denoiser))
        };
        let mut combined = dpdfnet_ladspa::voice::ProtectedEngine::new(
            dpdfnet_ladspa::AdaptiveEngine::new(quality, light, policy),
        );
        combined.set_mode(hushmic_denoiser::Mode::Process);
        combined.set_attenuation_limit_db(60.0);
        let mut costs = Vec::new();
        let mut output = [0.0; Q];
        for hop in input.as_chunks::<Q>().0 {
            let start = Instant::now();
            combined.process_hop(hop, &mut output).unwrap();
            costs.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        let mean = costs.iter().sum::<f64>() / costs.len() as f64;
        costs.sort_by(f64::total_cmp);
        eprintln!(
            "combined {tier:?} worker ms per10ms mean {mean:.3}, p95 {:.3}, max {:.3}",
            costs[costs.len() * 95 / 100],
            costs[costs.len() - 1]
        );
        assert!(mean < 8.0, "combined worker exceeds sustained 80% budget");
    }
    std::env::set_var("HUSHMIC_VOICE_MODEL_PATH", directory.join("missing.onnx"));
    let mut failed = plugin();
    let fallback = stream(&mut failed, &impulse, 0.0, 0.0);
    assert_eq!(
        fallback.iter().position(|v| v.abs() > 0.5),
        Some(at + delay)
    );
    drop(failed);
    std::fs::remove_dir_all(directory).unwrap();
    std::env::remove_var("HUSHMIC_VOICE_REQUESTED");
    std::env::remove_var("HUSHMIC_VOICE_REFERENCE");
    std::env::remove_var("HUSHMIC_VOICE_MODEL_PATH");
}
