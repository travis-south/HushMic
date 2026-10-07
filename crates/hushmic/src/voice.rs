//! Explicit local enrollment. Audio stays in RAM and only the reference is saved.
use crate::abtest::stream::{self, F32Reader, Header, SampleEndian};
use eframe::egui;
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

pub fn extraction_trial() -> bool {
    std::env::var_os("HUSHMIC_TSE_SCRIPT").is_some()
}

pub fn reference_path() -> PathBuf {
    if extraction_trial() {
        return std::env::var_os("HUSHMIC_TSE_REFERENCE")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                crate::config::Config::path().with_file_name("extraction-reference.bin")
            });
    }
    crate::config::Config::path().with_file_name("voice-reference.bin")
}

fn save_reference(reference: Vec<f32>) -> Result<(), String> {
    if !extraction_trial() {
        return hushmic_denoiser::voice::write_reference(
            &reference_path(),
            &reference
                .try_into()
                .map_err(|_| "Invalid detector reference")?,
        );
    }
    if reference.len() != 192
        || reference.iter().any(|v| !v.is_finite())
        || reference.iter().all(|v| *v == 0.0)
    {
        return Err("Invalid extraction reference".into());
    }
    use std::os::unix::fs::OpenOptionsExt;
    let path = reference_path();
    std::fs::create_dir_all(path.parent().ok_or("Reference needs a parent directory")?)
        .map_err(|e| e.to_string())?;
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temp)?;
        for sample in reference {
            file.write_all(&sample.to_le_bytes())?;
        }
        file.sync_all()?;
        std::fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temp);
    }
    result.map_err(|e| e.to_string())
}

fn extraction_reference(raw: &[f32], cancel: Arc<AtomicBool>) -> Result<Vec<f32>, String> {
    let mut command =
        Command::new(std::env::var_os("HUSHMIC_TSE_PYTHON").ok_or("Missing extraction Python")?);
    command
        .arg(std::env::var_os("HUSHMIC_TSE_SCRIPT").ok_or("Missing extraction script")?)
        .arg("--enroll")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped());
    let mut child = command.spawn().map_err(|e| e.to_string())?;
    let result = (|| {
        let mut input = std::io::BufWriter::new(child.stdin.take().ok_or("Encoder input missing")?);
        for sample in raw {
            if cancel.load(Ordering::Acquire) {
                return Err("Enrollment canceled".to_owned());
            }
            input
                .write_all(&sample.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
        input.flush().map_err(|e| e.to_string())?;
        drop(input);
        let mut reader = CaptureRead {
            pipe: child.stdout.take().ok_or("Encoder output missing")?,
            cancel,
            deadline: Instant::now() + Duration::from_secs(30),
        };
        let mut bytes = [0; 192 * 4];
        reader
            .read_exact(&mut bytes)
            .map_err(|e| format!("Extraction enrollment failed: {e}"))?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    })();
    let _ = child.kill();
    let _ = child.wait();
    result
}

struct CaptureRead {
    pipe: std::process::ChildStdout,
    cancel: Arc<AtomicBool>,
    deadline: Instant,
}
impl Read for CaptureRead {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.cancel.load(Ordering::Acquire) {
                return Err(io::Error::other("Enrollment canceled"));
            }
            if Instant::now() >= self.deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Microphone did not provide 20 seconds of audio",
                ));
            }
            let mut fd = libc::pollfd {
                fd: self.pipe.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            let result = unsafe { libc::poll(&mut fd, 1, 100) };
            if result > 0 {
                return self.pipe.read(out);
            }
            if result < 0 {
                let e = io::Error::last_os_error();
                if e.kind() != io::ErrorKind::Interrupted {
                    return Err(e);
                }
            }
        }
    }
}

fn capture(node: &str, cancel: Arc<AtomicBool>) -> Result<Vec<f32>, String> {
    if !crate::config::can_own_profile(node) {
        return Err("Select a physical microphone before enrollment".into());
    }
    let sources = crate::pipewire::sources_snapshot()
        .ok_or("Microphone list unavailable; retry when PipeWire is running")?;
    if !crate::pipewire::filter_real(&sources)
        .iter()
        .any(|source| source.name == node)
    {
        return Err("Selected physical microphone is unavailable".into());
    }
    let target = crate::pipewire::record_target(node);
    let mut command = Command::new("pw-record");
    command
        .args([
            "--target",
            &target,
            "--rate",
            "48000",
            "--channels",
            "1",
            "--format",
            "f32",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    use std::os::unix::process::CommandExt;
    unsafe {
        command.pre_exec(|| {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM as libc::c_ulong) == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .map_err(|e| format!("Could not record microphone: {e}"))?;
    let result = (|| {
        let mut pipe = CaptureRead {
            pipe: child.stdout.take().ok_or("Capture pipe missing")?,
            cancel: cancel.clone(),
            deadline: Instant::now() + Duration::from_secs(25),
        };
        let (endian, prefix) = match stream::read_header(&mut pipe).map_err(|e| e.to_string())? {
            Header::Raw(b) => (SampleEndian::Le, b.to_vec()),
            Header::Parsed(info) if info.channels == 1 && info.sample_rate == 48000 => {
                (info.endian, Vec::new())
            }
            _ => return Err("Microphone returned an unsupported audio format".into()),
        };
        let mut reader = F32Reader::new_with_prefix(pipe, endian, 4096, &prefix);
        let mut raw = vec![0.0; hushmic_denoiser::voice::ENROLL_SAMPLES];
        let mut chunk = Vec::new();
        let recording = (|| {
            let mut at = 0;
            while at < raw.len() {
                if !reader.read_samples(&mut chunk).map_err(|e| e.to_string())? {
                    return Err("Microphone stopped before enrollment completed".into());
                }
                let n = chunk.len().min(raw.len() - at);
                raw[at..at + n].copy_from_slice(&chunk[..n]);
                at += n;
            }
            let _ = child.kill();
            let _ = child.wait();
            if extraction_trial() {
                return extraction_reference(&raw, cancel.clone());
            }
            let paths = crate::controller::Paths::resolve();
            hushmic_denoiser::init_runtime(&paths.dylib).map_err(|e| e.to_string())?;
            if cancel.load(Ordering::Acquire) {
                return Err("Enrollment canceled".into());
            }
            hushmic_denoiser::voice::enroll(
                &raw,
                &paths.model_dir.join("voice-reference-encoder.onnx"),
            )
            .map(|reference| reference.to_vec())
        })();
        raw.fill(0.0);
        chunk.fill(0.0);
        recording
    })();
    let _ = child.kill();
    let _ = child.wait();
    result
}

type Recording = (
    Arc<AtomicBool>,
    mpsc::Receiver<Result<Vec<f32>, String>>,
    Instant,
);

struct VoiceApp {
    node: String,
    message: String,
    requested: bool,
    availability: String,
    enrolled: bool,
    recording: Option<Recording>,
    status: Option<mpsc::Receiver<(i32, String)>>,
    last_status: Instant,
}
impl VoiceApp {
    fn request(&mut self, action: &str) -> bool {
        let (code, out) = crate::control::client_run(&["protection".into(), action.into()]);
        self.message = if code == 0 {
            "Updated. Use Test my mic to hear the current output.".into()
        } else {
            out
        };
        self.last_status = Instant::now() - Duration::from_secs(2);
        code == 0
    }
}
impl Drop for VoiceApp {
    fn drop(&mut self) {
        if let Some((cancel, _, _)) = &self.recording {
            cancel.store(true, Ordering::Release);
        }
    }
}
impl eframe::App for VoiceApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        let result = self
            .recording
            .as_ref()
            .and_then(|(_, rx, _)| match rx.try_recv() {
                Ok(value) => Some(value),
                Err(mpsc::TryRecvError::Disconnected) => {
                    Some(Err("Enrollment worker stopped".into()))
                }
                Err(mpsc::TryRecvError::Empty) => None,
            });
        if let Some(result) = result {
            let (cancel, _, _) = self.recording.take().unwrap();
            if !cancel.load(Ordering::Acquire) {
                self.message = match result.and_then(save_reference) {
                    Ok(()) => {
                        self.enrolled = true;
                        if self.request("reload") {
                            if extraction_trial() {
                                "Voice learned locally. Enable target-speaker extraction below."
                                    .into()
                            } else {
                                "Voice learned locally. Enable pause protection explicitly below."
                                    .into()
                            }
                        } else {
                            format!("Reference saved, but live reload failed: {}", self.message)
                        }
                    }
                    Err(e) => {
                        format!("Enrollment failed: {e}. Your previous reference is unchanged.")
                    }
                };
            }
        }
        if self.status.is_none() && self.last_status.elapsed() >= Duration::from_secs(1) {
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(crate::control::client_run(&[
                    "protection".into(),
                    "status".into(),
                ]));
            });
            self.status = Some(rx);
            self.last_status = Instant::now();
        }
        if let Some((code, out)) = self.status.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.status = None;
            if code == 0 {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&out) {
                    self.availability = format!(
                        "{}: {}{}",
                        if extraction_trial() {
                            "Extraction"
                        } else {
                            "Protection"
                        },
                        value["state"].as_str().unwrap_or("unavailable"),
                        value["reason"]
                            .as_str()
                            .map(|r| format!(" ({r})"))
                            .unwrap_or_default()
                    );
                    self.requested = value["requested"].as_bool().unwrap_or(false);
                    self.enrolled = value["enrolled"].as_bool().unwrap_or(false);
                }
            } else {
                self.availability = out;
            }
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(if extraction_trial() { "Target-speaker extraction · experimental" } else { "Voice setup · experimental" });
            ui.label(if extraction_trial() {
                "INT8 four-block extractor. Speaker selection can be weak and CPU load can cause gaps. Extraction followed by ordinary denoising. Processing delay: 190 ms at the supported 10 ms graph quantum. High Quality / Light and suppression strength control the denoiser."
            } else { "Pause protection uses one fixed voice reference. It may still pass other speakers; quieter speech under a louder speaker is deferred." });
            ui.add_space(10.0);
            ui.label(format!("Physical microphone: {}", self.node));
            ui.label("Record 20 seconds of your voice alone in a quiet room. Speak naturally and continuously, with the microphone at your usual distance. Avoid television, other voices, and clipping.");
            ui.label("Everything runs locally. The recording is discarded; only your voice reference is saved.");
            ui.add_space(10.0);
            let busy = self.recording.is_some();
            if ui.add_enabled(!busy, egui::Button::new(if self.enrolled { "Replace voice · record 20 seconds" } else { "Learn voice · record 20 seconds" })).clicked() {
                let cancel = Arc::new(AtomicBool::new(false));
                let (tx, rx) = mpsc::channel();
                let flag = cancel.clone();
                let node = self.node.clone();
                let wake = ctx.clone();
                std::thread::spawn(move || { let _ = tx.send(capture(&node, flag)); wake.request_repaint(); });
                self.recording = Some((cancel, rx, Instant::now()));
                self.message = "Recording your physical microphone…".into();
            }
            if let Some((cancel, _, start)) = &self.recording {
                ui.label(format!("Recording / processing: {:.0} seconds", start.elapsed().as_secs_f32()));
                if ui.button("Cancel enrollment").clicked() { cancel.store(true, Ordering::Release); self.message = "Canceled. Previous reference retained.".into(); }
            }
            let mut enabled = self.requested;
            if ui.add_enabled(!busy && (self.enrolled || self.requested), egui::Checkbox::new(&mut enabled, if extraction_trial() { "Enable target-speaker extraction" } else { "Enable pause protection" })).changed() { self.request(if enabled { "enable" } else { "disable" }); }
            if ui.add_enabled(!busy && self.enrolled, egui::Button::new("Forget voice")).clicked() { self.request("forget"); }
            ui.add_space(10.0);
            ui.label(&self.message);
            ui.label(if self.enrolled { "Reference saved locally." } else { "No enrolled voice." });
            ui.label(&self.availability);
        });
        ctx.request_repaint_after(Duration::from_millis(100));
    }
}

pub fn run(node: String) -> Result<(), String> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([560.0, 500.0]),
        ..Default::default()
    };
    eframe::run_native(
        "HushMic voice setup",
        options,
        Box::new(move |_| {
            Ok(Box::new(VoiceApp {
                node,
                message: "Ready. Recording starts only when you click Learn or Replace.".into(),
                availability: "Checking live protection status…".into(),
                requested: crate::config::Config::load().pause_protection,
                enrolled: reference_path().exists(),
                recording: None,
                status: None,
                last_status: Instant::now() - Duration::from_secs(2),
            }))
        }),
    )
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_and_deadline_interrupt_a_stalled_capture() {
        let mut child = Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let flag = Arc::new(AtomicBool::new(true));
        let mut reader = CaptureRead {
            pipe: child.stdout.take().unwrap(),
            cancel: flag.clone(),
            deadline: Instant::now() + Duration::from_secs(1),
        };
        // Header parsing uses read_exact, which retries Interrupted forever.
        // Exercise cancellation before any bytes and within both header reads.
        for prefix in [b"".as_slice(), b"RI", b"RIFF"] {
            assert_eq!(
                stream::read_header(&mut prefix.chain(&mut reader))
                    .err()
                    .unwrap()
                    .kind(),
                io::ErrorKind::Other
            );
        }
        flag.store(false, Ordering::Release);
        reader.deadline = Instant::now();
        assert_eq!(
            stream::read_header(&mut reader).err().unwrap().kind(),
            io::ErrorKind::TimedOut
        );
        child.kill().unwrap();
        child.wait().unwrap();
    }
}
