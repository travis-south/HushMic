//! Final-mixture gate and its fixed audio hold, owned exclusively by the worker.
use crate::{AdaptiveEngine, HopEngine, Policy, Tier};
use hushmic_denoiser::{
    voice::{Presence, HOLD_SAMPLES},
    Denoiser, Mode, HOP,
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

type Engine = AdaptiveEngine<Denoiser, Box<dyn Policy>>;
pub struct ProtectedEngine {
    engine: Engine,
    detector: Option<Presence>,
    requested: bool,
    delay: Vec<f32>,
    position: usize,
    hop: usize,
    probabilities: [f32; 64],
    masks: [bool; 64],
    release: usize,
    mode: Mode,
    reason: Option<String>,
    announced: String,
    overloaded: usize,
    reported_latency: usize,
}
impl ProtectedEngine {
    pub fn new(engine: Engine) -> Self {
        let requested = std::env::var("HUSHMIC_VOICE_REQUESTED").as_deref() == Ok("1");
        let loaded = if requested {
            (|| {
                let reference = PathBuf::from(
                    std::env::var_os("HUSHMIC_VOICE_REFERENCE")
                        .ok_or("No voice reference configured")?,
                );
                let model = PathBuf::from(
                    std::env::var_os("HUSHMIC_VOICE_MODEL_PATH")
                        .ok_or("No presence model configured")?,
                );
                Presence::new(&model, hushmic_denoiser::voice::read_reference(&reference)?)
            })()
        } else {
            Err(String::new())
        };
        let (detector, reason) = match loaded {
            Ok(d) => (Some(d), None),
            Err(e) => (None, requested.then_some(e)),
        };
        let mut this = Self {
            engine,
            detector,
            requested,
            delay: vec![0.0; if requested { HOLD_SAMPLES } else { 0 }],
            position: 0,
            hop: 0,
            probabilities: [0.0; 64],
            masks: [false; 64],
            release: 0,
            mode: Mode::Process,
            reason,
            announced: String::new(),
            overloaded: 0,
            reported_latency: 0,
        };
        if !requested {
            this.reason = std::env::var("HUSHMIC_VOICE_UNAVAILABLE_REASON").ok();
        }
        this.announce();
        this
    }
    fn announce(&mut self) {
        let state = if !self.requested {
            self.reason
                .as_ref()
                .map(|r| format!("unavailable ({r})"))
                .unwrap_or_else(|| "disabled".to_owned())
        } else if self.mode != Mode::Process || self.engine.live() == Tier::Raw {
            "inactive (bypass or mute)".to_owned()
        } else if let Some(reason) = &self.reason {
            format!("unavailable ({reason})")
        } else {
            "active".to_owned()
        };
        if state != self.announced {
            crate::log::contract_line(&format!("protection: {state}"));
            self.announced = state;
        }
    }
    fn fail(&mut self, reason: String) {
        self.detector = None;
        self.reason = Some(reason);
        self.announce();
    }
    fn decision(&mut self, source: usize, probability: f32) {
        let slot = source % 64;
        self.probabilities[slot] = probability;
        self.masks[slot] = false;
        if probability >= 0.5 {
            self.release = source + 16;
            for earlier in source.saturating_sub(7)..=source {
                if self.probabilities[earlier % 64] >= 0.1 {
                    self.masks[earlier % 64] = true;
                }
            }
        }
        if self.release > 0 && source <= self.release && probability >= 0.1 {
            self.masks[slot] = true;
        }
    }
}
impl HopEngine for ProtectedEngine {
    fn process_hop(&mut self, input: &[f32; HOP], output: &mut [f32; HOP]) -> Result<(), String> {
        let start = Instant::now();
        if self.mode == Mode::Process && self.engine.live() != Tier::Raw {
            if let Some(detector) = self.detector.as_mut() {
                match detector.process_hop(input) {
                    Ok(Some((j, p))) => self.decision(j, p),
                    Ok(None) => {}
                    Err(e) => self.fail(e),
                }
            }
        }
        // Charge detector work to the existing adaptive tier cost model too.
        self.engine.note_pause(start.elapsed());
        let result = self.engine.process_hop(input, output);
        if !self.delay.is_empty() {
            let open = self.mode != Mode::Process
                || self.engine.live() == Tier::Raw
                || self.detector.is_none()
                || (self.hop >= 14 && self.masks[(self.hop - 14) % 64]);
            for sample in output.iter_mut() {
                let delayed = self.delay[self.position];
                self.delay[self.position] = *sample;
                self.position = (self.position + 1) % self.delay.len();
                *sample = if open { delayed } else { 0.0 };
            }
        }
        if self.detector.is_some() {
            self.overloaded = if start.elapsed() > Duration::from_millis(10) {
                self.overloaded + 1
            } else {
                0
            };
            if self.overloaded >= 20 {
                self.fail(
                    "Combined detector and denoiser cannot keep up; ordinary denoising continues"
                        .into(),
                );
            }
        }
        self.hop += 1;
        self.announce();
        result
    }
    fn reset(&mut self) {
        self.engine.reset();
        if let Some(d) = self.detector.as_mut() {
            d.reset();
        }
        self.delay.fill(0.0);
        self.position = 0;
        self.hop = 0;
        self.probabilities.fill(0.0);
        self.masks.fill(false);
        self.release = 0;
        self.overloaded = 0;
        self.reported_latency = 0;
        self.announce();
    }
    fn set_mode(&mut self, mode: Mode) {
        if self.mode != mode {
            if let Some(d) = self.detector.as_mut() {
                d.reset();
            }
            self.probabilities.fill(0.0);
            self.masks.fill(false);
            self.release = 0;
            self.hop = 0;
            self.delay.fill(0.0);
            self.position = 0;
        }
        self.mode = mode;
        self.engine.set_mode(mode);
        self.announce();
    }
    fn set_attenuation_limit_db(&mut self, db: f32) {
        self.engine.set_attenuation_limit_db(db);
    }
    fn set_lag_hops(&mut self, hops: u32) {
        self.engine.set_lag_hops(hops);
        if hops >= 4 && self.detector.is_some() {
            self.fail("Detector stream fell behind; ordinary denoising continues".into());
        }
    }
    fn note_pause(&mut self, pause: Duration) {
        self.engine.note_pause(pause);
    }
    fn note_output_lead(&mut self, lead: usize) {
        let latency = lead + hushmic_denoiser::LATENCY_SAMPLES + self.delay.len();
        if latency != self.reported_latency {
            crate::log::contract_line(&format!("latency: {latency}"));
            self.reported_latency = latency;
        }
        if self.requested && latency > 9600 && self.detector.is_some() {
            self.fail(format!(
                "Host output lead {lead} samples exceeds the 200 ms protection budget"
            ));
        }
    }
}
