//! Offline streaming bridge: u32 LE length + f32 LE mono 48 kHz packets.
//! A zero-length packet drains the stream and exits. Startup returns latency.
use hushmic_denoiser::{Denoiser, Engine, Inference, StreamDenoiser};
use std::io::{self, BufWriter, Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args_os().nth(1).ok_or("expected model path")?;
    let mut denoiser = Denoiser::from_file_with(model, Inference::NativeInt8)?;
    assert_eq!(
        denoiser.engine(),
        Engine::NativeInt8,
        "native engine required"
    );
    denoiser.set_attenuation_limit_db(100.0);
    let mut output = BufWriter::new(io::stdout().lock());
    output.write_all(&(denoiser.latency_samples() as u32).to_le_bytes())?;
    output.flush()?;
    let mut stream = StreamDenoiser::new(denoiser);
    let mut input = io::stdin().lock();
    loop {
        let mut header = [0; 4];
        input.read_exact(&mut header)?;
        let count = u32::from_le_bytes(header) as usize;
        if count > 9600 {
            return Err("packet exceeds 200 ms".into());
        }
        let samples = if count == 0 {
            vec![0.0; stream.pending()]
        } else {
            let mut bytes = vec![0; count * 4];
            input.read_exact(&mut bytes)?;
            let samples: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|s| f32::from_le_bytes(*s))
                .collect();
            if !samples.iter().all(|s| s.is_finite()) {
                return Err("non-finite input".into());
            }
            samples
        };
        let cleaned = stream.process(&samples).to_vec();
        if let Some(error) = stream.take_error() {
            return Err(error.into());
        }
        if !cleaned.iter().all(|s| s.is_finite()) {
            return Err("non-finite output".into());
        }
        output.write_all(&(cleaned.len() as u32).to_le_bytes())?;
        for sample in cleaned {
            output.write_all(&sample.to_le_bytes())?;
        }
        output.flush()?;
        if count == 0 {
            break;
        }
    }
    Ok(())
}
