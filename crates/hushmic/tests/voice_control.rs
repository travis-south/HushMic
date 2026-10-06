use hushmic::{
    control::{self, Request},
    controller, diagnostics,
};
#[test]
fn explicit_protection_controls_and_reports() {
    assert_eq!(
        control::parse_request(&["protection", "enable"]),
        Ok(Request::SetProtection(true))
    );
    assert_eq!(
        control::parse_request(&["protection", "disable"]),
        Ok(Request::SetProtection(false))
    );
    assert_eq!(
        control::parse_request(&["protection", "status"]),
        Ok(Request::ProtectionStatus)
    );
    assert!(control::parse_request(&["protection", "auto"]).is_err());
    let report = diagnostics::parse_protection_line(
        b"host noise [dpdfnet-ladspa] protection: unavailable (missing reference)\n",
    )
    .unwrap();
    assert_eq!(report.state, "unavailable");
    assert_eq!(report.reason.as_deref(), Some("missing reference"));
    assert!(
        diagnostics::parse_protection_line(b"[dpdfnet-ladspa] protection: activegarbage").is_none()
    );
    assert!(
        diagnostics::parse_protection_line(b"[dpdfnet-ladspa] protection: unavailable (torn")
            .is_none()
    );
    assert_eq!(
        diagnostics::protection(false, true, controller::RunMode::Suppress),
        diagnostics::Protection::disabled()
    );
    assert_eq!(
        diagnostics::protection(true, true, controller::RunMode::Mute).state,
        "inactive"
    );
    assert_eq!(
        diagnostics::parse_latency_line(b"host [dpdfnet-ladspa] latency: 9120\n"),
        Some(9120)
    );
    assert!(diagnostics::parse_latency_line(b"[dpdfnet-ladspa] latency: 9120garbage").is_none());
    assert!(diagnostics::parse_latency_line(b"[dpdfnet-ladspa] latency: 0").is_none());
    assert_eq!(controller::latency_samples(true), 9120);
    assert_eq!(controller::protection_latency_samples(512), 9600);
    assert_eq!(controller::protection_latency_samples(960), 9600);
    for quantum in [480, 512, 960] {
        assert!(controller::protection_quantum_supported(quantum));
    }
    for quantum in [0, 700, 1024, 2048] {
        assert!(!controller::protection_quantum_supported(quantum));
    }
}
