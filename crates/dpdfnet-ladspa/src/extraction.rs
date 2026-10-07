//! Opt-in local experiment; all subprocess I/O stays on the existing worker.
use crate::HopEngine;
use hushmic_denoiser::{Denoiser, Mode, HOP};
use std::io::{self, Read, Write};
use std::os::{fd::AsRawFd, unix::process::CommandExt};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

pub struct Extraction {
    denoiser: Denoiser,
    child: Child,
    input: ChildStdin,
    output: ChildStdout,
    mode: Mode,
    failed: bool,
    lead: usize,
}

fn read_bounded(pipe: &mut ChildStdout, bytes: &mut [u8], timeout: Duration) -> io::Result<()> {
    let deadline = Instant::now() + timeout;
    let mut at = 0;
    while at < bytes.len() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Extraction bridge timed out",
            ));
        }
        let mut fd = libc::pollfd {
            fd: pipe.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let ready = unsafe {
            libc::poll(
                &mut fd,
                1,
                remaining.as_millis().min(i32::MAX as u128) as i32,
            )
        };
        if ready < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        if ready == 0 {
            continue;
        }
        let count = pipe.read(&mut bytes[at..])?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Extraction bridge exited",
            ));
        }
        at += count;
    }
    Ok(())
}

impl Extraction {
    pub fn new() -> Result<Self, String> {
        let model_path = crate::model_path();
        let denoiser = crate::load_denoiser(&model_path).map_err(|e| e.to_string())?;
        let mut command = Command::new(
            std::env::var_os("HUSHMIC_TSE_PYTHON").ok_or("Missing extraction Python")?,
        );
        command
            .arg(std::env::var_os("HUSHMIC_TSE_SCRIPT").ok_or("Missing extraction script")?)
            .arg("--reference")
            .arg(
                std::env::var_os("HUSHMIC_TSE_REFERENCE")
                    .ok_or("Learn an extraction voice first")?,
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());
        unsafe {
            command.pre_exec(|| {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM as libc::c_ulong) == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = command.spawn().map_err(|e| e.to_string())?;
        let input = child.stdin.take().ok_or("Extraction input missing")?;
        let output = child.stdout.take().ok_or("Extraction output missing")?;
        let mut engine = Self {
            denoiser,
            child,
            input,
            output,
            mode: Mode::Process,
            failed: false,
            lead: 0,
        };
        let mut handshake = [0; 12];
        read_bounded(&mut engine.output, &mut handshake, Duration::from_secs(30))
            .map_err(|e| e.to_string())?;
        if &handshake[..8] != b"HMTSE001"
            || u32::from_le_bytes(handshake[8..].try_into().unwrap()) != 4320
        {
            return Err("Extraction bridge has an incompatible delay or protocol".into());
        }
        let tier = if crate::plan_tiers(&model_path, false).1 == crate::Tier::Light {
            "light"
        } else {
            "quality"
        };
        crate::log::contract_line(&format!(
            "engine: {tier} (experimental target-speaker extraction, four-block ablation)"
        ));
        #[cfg(feature = "native")]
        let denoiser_engine = engine.denoiser.engine().to_string();
        #[cfg(not(feature = "native"))]
        let denoiser_engine = "onnx";
        crate::log::contract_line(&crate::inference_line(
            crate::model_name(&model_path),
            &format!("target extraction int8 + {denoiser_engine}"),
            None,
        ));
        crate::log::contract_line("protection: active");
        Ok(engine)
    }

    fn fail(&mut self, reason: &str) {
        if !self.failed {
            crate::log::contract_line(&format!(
                "protection: unavailable ({reason}; experimental output is silent)"
            ));
            let _ = self.child.kill();
            self.failed = true;
        }
    }
}

impl Drop for Extraction {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl HopEngine for Extraction {
    fn process_hop(&mut self, input: &[f32; HOP], output: &mut [f32; HOP]) -> Result<(), String> {
        output.fill(0.0);
        if self.failed {
            return Err("Extraction bridge unavailable".into());
        }
        let result = (|| {
            if input.iter().any(|v| !v.is_finite()) {
                return Err(io::Error::other("Non-finite microphone audio"));
            }
            let mode: u32 = match self.mode {
                Mode::Process => 1,
                Mode::Bypass => 2,
                Mode::Mute => 3,
                _ => 3,
            };
            let mut packet = [0; 4 + HOP * 4];
            packet[..4].copy_from_slice(&mode.to_le_bytes());
            for (bytes, sample) in packet[4..].as_chunks_mut::<4>().0.iter_mut().zip(input) {
                bytes.copy_from_slice(&sample.to_le_bytes());
            }
            self.input.write_all(&packet)?;
            let mut response = [0; 4 + HOP * 4];
            read_bounded(&mut self.output, &mut response, Duration::from_millis(200))?;
            if u32::from_le_bytes(response[..4].try_into().unwrap()) != HOP as u32 {
                return Err(io::Error::other("Invalid extraction packet"));
            }
            let samples: [f32; HOP] = std::array::from_fn(|i| {
                f32::from_le_bytes(response[4 + i * 4..8 + i * 4].try_into().unwrap())
            });
            if samples.iter().any(|v| !v.is_finite()) {
                return Err(io::Error::other("Non-finite extraction output"));
            }
            self.denoiser
                .process_hop(&samples, output)
                .map_err(|e| io::Error::other(e.to_string()))
        })();
        result.map_err(|e: io::Error| {
            output.fill(0.0);
            self.fail(&e.to_string());
            e.to_string()
        })
    }

    fn reset(&mut self) {
        self.denoiser.reset();
        if self.failed {
            return;
        }
        let result = (|| {
            self.input.write_all(&0u32.to_le_bytes())?;
            let mut ack = [0; 4];
            read_bounded(&mut self.output, &mut ack, Duration::from_secs(2))?;
            if ack != [0; 4] {
                return Err(io::Error::other("Invalid extraction reset"));
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.fail(&e.to_string());
        }
        self.lead = 0;
    }

    fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        self.denoiser.set_mode(mode);
        if !self.failed {
            crate::log::contract_line(if mode == Mode::Process {
                "protection: active"
            } else {
                "protection: inactive (bypass or mute)"
            });
        }
    }
    fn set_attenuation_limit_db(&mut self, db: f32) {
        self.denoiser.set_attenuation_limit_db(db);
    }
    fn note_output_lead(&mut self, lead: usize) {
        if self.lead != lead {
            crate::log::contract_line(&format!(
                "latency: {}",
                4320 + hushmic_denoiser::LATENCY_SAMPLES + lead
            ));
            self.lead = lead;
        }
        if 4320 + hushmic_denoiser::LATENCY_SAMPLES + lead > 9600 {
            self.fail("Graph quantum exceeds the 200 ms extraction budget");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stalled_and_closed_bridge_reads_are_bounded() {
        let mut child = Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let mut output = child.stdout.take().unwrap();
        assert_eq!(
            read_bounded(&mut output, &mut [0; 1], Duration::from_millis(10))
                .unwrap_err()
                .kind(),
            io::ErrorKind::TimedOut
        );
        drop(input);
        assert_eq!(
            read_bounded(&mut output, &mut [0; 1], Duration::from_secs(1))
                .unwrap_err()
                .kind(),
            io::ErrorKind::UnexpectedEof
        );
        child.wait().unwrap();
    }
}
