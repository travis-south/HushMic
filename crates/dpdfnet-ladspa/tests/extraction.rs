//! Opt-in check using the locally acquired prototype and a public reference.
use dpdfnet_ladspa::get_ladspa_descriptor;
use ladspa::{PortConnection, PortData};
use std::{
    cell::RefCell,
    time::{Duration, Instant},
};

#[test]
#[ignore = "Requires HUSHMIC_TSE_PYTHON, SCRIPT and public REFERENCE; see prototype doc"]
fn experimental_plugin_alignment_and_lifecycle() {
    std::env::set_var("HUSHMIC_VOICE_REQUESTED", "1");
    assert!(std::env::var_os("HUSHMIC_TSE_SCRIPT").is_some());
    let descriptor = get_ladspa_descriptor(0).unwrap();
    let mut plugin = (descriptor.new)(&descriptor, 48000);
    let input: Vec<f32> = (0..48000 * 4)
        .map(|i| (i as f32 * 0.031).sin() * 0.03)
        .collect();
    let noise_fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/target-speaker-extraction/quiet-alone-mixture.f32");
    let bytes =
        std::fs::read(noise_fixture).expect("Prepare the public noise fixture as documented");
    let speech: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert_eq!(speech.len(), 48000 * 10);
    for mode in [1.0, 2.0, 0.0] {
        let input = if mode == 0.0 { &speech } else { &input };
        plugin.activate();
        let mut actual = Vec::new();
        let mut next = Instant::now();
        for hop in input.as_chunks::<480>().0 {
            let mut output = [0.0; 480];
            let ports = [
                PortConnection {
                    port: descriptor.ports[0],
                    data: PortData::AudioInput(hop),
                },
                PortConnection {
                    port: descriptor.ports[1],
                    data: PortData::AudioOutput(RefCell::new(&mut output)),
                },
                PortConnection {
                    port: descriptor.ports[2],
                    data: PortData::ControlInput(&100.0),
                },
                PortConnection {
                    port: descriptor.ports[3],
                    data: PortData::ControlInput(&mode),
                },
            ];
            plugin.run(480, &ports.iter().collect::<Vec<_>>());
            actual.extend(output);
            next += Duration::from_millis(10);
            std::thread::sleep(next.saturating_duration_since(Instant::now()));
        }
        assert!(actual.iter().all(|v| v.is_finite()));
        if mode == 1.0 {
            let prefix = actual[..9120]
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);
            assert!(
                prefix < 1e-6,
                "Bypass arrived before its declared delay: {prefix}"
            );
            let error = actual[9120..]
                .iter()
                .zip(&input[..input.len() - 9120])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                error < 1e-6,
                "Bypass lost alignment or worker deadlines: {error}"
            );
        } else if mode == 2.0 {
            assert!(actual.iter().all(|v| *v == 0.0));
        } else {
            let source = &input[..3 * 48000];
            let clean = &actual[9120..3 * 48000 + 9120];
            let projection = clean
                .iter()
                .zip(source)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>()
                / source.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
            assert!(
                projection.abs() > 0.5,
                "Enrolled speech was mostly lost: {projection}"
            );
            eprintln!("cascade regression: retained source projection {projection:.3}");
            let a = (3.2 * 48000.0) as usize;
            let b = (6.8 * 48000.0) as usize;
            let energy =
                |samples: &[f32]| samples.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
            assert!(
                energy(&actual[a + 9120..b + 9120]) < energy(&input[a..b]) * 1e-4,
                "Extraction bypassed ordinary noise suppression"
            );
        }
        plugin.deactivate();
    }
}
