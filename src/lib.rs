use atomic_float::AtomicF32;
use nih_plug::prelude::*;
use nih_plug_vizia::ViziaState;
use std::sync::Arc;

mod editor;

/// The time it takes for the peak meter to decay by 12 dB after switching to complete silence.
const PEAK_METER_DECAY_MS: f64 = 150.0;

/// This is mostly identical to the gain example, minus some fluff, and with a GUI.
pub struct Croaker {
    params: Arc<CroakerParams>,

    /// Needed to normalize the peak meter's response based on the sample rate.
    output_peak_meter_decay_weight: f32,
    /// The current data for the peak meter. This is stored as an [`Arc`] so we can share it between
    /// the GUI and the audio processing parts. If you have more state to share, then it's a good
    /// idea to put all of that in a struct behind a single `Arc`.
    ///
    /// This is stored as voltage gain.
    output_peak_meter: Arc<AtomicF32>,

    // Input gain peak meter members
    input_peak_meter_decay_weight: f32,
    input_peak_meter: Arc<AtomicF32>,
}

#[derive(Params)]
struct CroakerParams {
    /// The editor state, saved together with the parameter state so the custom scaling can be
    /// restored.
    #[persist = "editor-state"]
    editor_state: Arc<ViziaState>,

    #[id = "input-gain"]
    pub input_gain: FloatParam,

    #[id = "output-gain"]
    pub output_gain: FloatParam,

    #[id = "dry-wet"]
    pub dry_wet_ratio: FloatParam,

    #[id = "saturation"]
    pub saturation: FloatParam,
}

impl Default for Croaker {
    fn default() -> Self {
        Self {
            params: Arc::new(CroakerParams::default()),

            output_peak_meter_decay_weight: 1.0,
            output_peak_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            input_peak_meter_decay_weight: 1.0,
            input_peak_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
        }
    }
}

impl Default for CroakerParams {
    fn default() -> Self {
        Self {
            editor_state: editor::default_state(),

            input_gain: FloatParam::new(
                "Input Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-30.0),
                    max: util::db_to_gain(30.0),
                    factor: FloatRange::gain_skew_factor(-30.0, 30.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),

            output_gain: FloatParam::new(
                "Output Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-30.0),
                    max: util::db_to_gain(30.0),
                    factor: FloatRange::gain_skew_factor(-30.0, 30.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),

            saturation: FloatParam::new(
                "Saturation",
                0.5,
                FloatRange::Linear {
                    min: 0.0,
                    max: 0.999,
                },
            )
            .with_smoother(SmoothingStyle::Linear(50.0))
            .with_value_to_string(formatters::v2s_f32_rounded(2)),

            dry_wet_ratio: FloatParam::new(
                "Dry/wet",
                0.5,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(50.0))
            .with_value_to_string(formatters::v2s_f32_rounded(2)),
        }
    }
}

impl Plugin for Croaker {
    const NAME: &'static str = "croaker";
    const VENDOR: &'static str = "renzofrog";
    const URL: &'static str = "https://www.renzofrog.com";
    const EMAIL: &'static str = "renzomledesma@gmail.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const DEFAULT_INPUT_CHANNELS: u32 = 2;
    const DEFAULT_OUTPUT_CHANNELS: u32 = 2;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            self.params.clone(),
            self.input_peak_meter.clone(),
            self.output_peak_meter.clone(),
            self.params.editor_state.clone(),
        )
    }

    fn accepts_bus_config(&self, config: &BusConfig) -> bool {
        // This works with any symmetrical IO layout
        config.num_input_channels == config.num_output_channels && config.num_input_channels > 0
    }

    fn initialize(
        &mut self,
        _bus_config: &BusConfig,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // After `PEAK_METER_DECAY_MS` milliseconds of pure silence, the peak meter's value should
        // have dropped by 12 dB
        self.input_peak_meter_decay_weight = 0.25f64
            .powf((buffer_config.sample_rate as f64 * PEAK_METER_DECAY_MS / 1000.0).recip())
            as f32;

        self.output_peak_meter_decay_weight = 0.25f64
            .powf((buffer_config.sample_rate as f64 * PEAK_METER_DECAY_MS / 1000.0).recip())
            as f32;

        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        for channel_samples in buffer.iter_samples() {
            let mut input_amplitude = 0.0;
            let mut output_amplitude = 0.0;
            let num_samples = channel_samples.len();

            let input_gain = self.params.input_gain.smoothed.next();
            let output_gain = self.params.output_gain.smoothed.next();
            let a = self.params.saturation.smoothed.next();
            let dry_wet_ratio = self.params.dry_wet_ratio.smoothed.next();
            for sample in channel_samples {
                // Apply input gain
                *sample *= input_gain;
                input_amplitude += *sample;

                // Apply saturation
                let k = 2.0 * a / (1.0 - a);
                let wet = ((1.0 + k) * *sample) / (1.0 + k * (*sample).abs());

                // Apply dry/wet
                *sample = (*sample * (1.0 - dry_wet_ratio)) + (wet * dry_wet_ratio);

                // Apply output gain
                *sample *= output_gain;
                output_amplitude += *sample;
            }

            // To save resources, a plugin can (and probably should!) only perform expensive
            // calculations that are only displayed on the GUI while the GUI is open
            if self.params.editor_state.is_open() {
                input_amplitude = (input_amplitude / num_samples as f32).abs();
                output_amplitude = (output_amplitude / num_samples as f32).abs();

                let current_input_peak_meter = self
                    .input_peak_meter
                    .load(std::sync::atomic::Ordering::Relaxed);
                let current_output_peak_meter = self
                    .output_peak_meter
                    .load(std::sync::atomic::Ordering::Relaxed);

                let get_new_peak_meter = |amp: f32, meter: f32, weight: f32| {
                    if amp > meter {
                        amp
                    } else {
                        meter * weight + amp * (1.0 - weight)
                    }
                };

                let new_input_peak_meter = get_new_peak_meter(
                    input_amplitude,
                    current_input_peak_meter,
                    self.input_peak_meter_decay_weight,
                );
                let new_output_peak_meter = get_new_peak_meter(
                    output_amplitude,
                    current_output_peak_meter,
                    self.output_peak_meter_decay_weight,
                );

                self.input_peak_meter
                    .store(new_input_peak_meter, std::sync::atomic::Ordering::Relaxed);
                self.output_peak_meter
                    .store(new_output_peak_meter, std::sync::atomic::Ordering::Relaxed);
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for Croaker {
    const CLAP_ID: &'static str = "renzofrog_plugins";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("saturator/distorter/makes sound go croak");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Utility,
        ClapFeature::Distortion,
    ];
}

impl Vst3Plugin for Croaker {
    const VST3_CLASS_ID: [u8; 16] = *b"renzofrogcroaker";
    const VST3_CATEGORIES: &'static str = "Fx|Distortion";
}

nih_export_vst3!(Croaker);
