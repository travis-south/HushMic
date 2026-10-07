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
    for mode in [1.0, 2.0, 0.0] {
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
            assert!(actual[..8160].iter().all(|v| *v == 0.0));
            assert_eq!(
                &actual[8160..],
                &input[..input.len() - 8160],
                "Bypass lost alignment or worker deadlines"
            );
        } else if mode == 2.0 {
            assert!(actual.iter().all(|v| *v == 0.0));
        } else {
            assert!(
                actual.iter().any(|v| v.abs() > 1e-6),
                "Extractor emitted only silence"
            );
        }
        plugin.deactivate();
    }
}
