use atomic_float::AtomicF32;
use fx::{dc_filter::DcFilter, oversampling::HalfbandFilter, waveshapers::*};
use nih_plug::prelude::*;
use nih_plug_vizia::ViziaState;
use std::sync::Arc;

mod assets;
mod editor;
mod fonts;

const PEAK_METER_DECAY_MS: f64 = 150.0;

#[derive(Enum, Debug, PartialEq, Eq, Clone, Copy)]
pub enum DistortionType {
    #[id = "saturation"]
    #[name = "Saturator"]
    Saturation,

    #[id = "hard-clipping"]
    #[name = "Hard clipper"]
    HardClipping,

    #[id = "fuzzy-rectifier"]
    #[name = "Fuzzy rectifier"]
    FuzzyRectifier,

    #[id = "shockley-diode-rectifier"]
    #[name = "Diode rectifier"]
    ShockleyDiodeRectifier,

    #[id = "dropout"]
    #[name = "Dropout"]
    Dropout,

    #[id = "double-soft-clipper"]
    #[name = "Double soft clipper"]
    DoubleSoftClipper,

    #[id = "wavefolding"]
    #[name = "Wavefolder"]
    Wavefolding,
}

pub fn process_sample(distortion_type: &DistortionType, drive: f32, input_sample: f32) -> f32 {
    match distortion_type {
        DistortionType::Saturation => get_saturator_output(drive, input_sample),
        DistortionType::HardClipping => get_hard_clipper_output(drive, input_sample),
        DistortionType::FuzzyRectifier => get_fuzzy_rectifier_output(drive, input_sample),
        DistortionType::ShockleyDiodeRectifier => {
            get_shockley_diode_rectifier_output(drive, input_sample)
        }
        DistortionType::Dropout => get_dropout_output(drive, input_sample),
        DistortionType::DoubleSoftClipper => get_double_soft_clipper_output(drive, input_sample),
        DistortionType::Wavefolding => get_wavefolder_output(drive, input_sample),
    }
}

pub struct Croaker {
    params: Arc<CroakerParams>,

    // Input/output gain peak meter members
    peak_meter_decay_weight: f32,
    output_peak_meter: Arc<AtomicF32>,
    input_peak_meter: Arc<AtomicF32>,

    upsampler: HalfbandFilter,
    downsampler: HalfbandFilter,
    dc_filter: DcFilter,
    oversample_factor: usize,
}

#[derive(Params)]
struct CroakerParams {
    #[persist = "editor-state"]
    editor_state: Arc<ViziaState>,

    #[id = "input-gain"]
    pub input_gain: FloatParam,

    #[id = "output-gain"]
    pub output_gain: FloatParam,

    #[id = "dry-wet"]
    pub dry_wet_ratio: FloatParam,

    #[id = "drive"]
    pub drive: FloatParam,

    #[id = "distortion-type"]
    pub distortion_type: EnumParam<DistortionType>,
}

impl Default for Croaker {
    fn default() -> Self {
        Self {
            params: Arc::new(CroakerParams::default()),

            peak_meter_decay_weight: 1.0,
            output_peak_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            input_peak_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),

            upsampler: HalfbandFilter::new(8, true),
            downsampler: HalfbandFilter::new(8, true),
            dc_filter: DcFilter::default(),
            oversample_factor: 4,
        }
    }
}

impl Default for CroakerParams {
    fn default() -> Self {
        Self {
            editor_state: editor::default_state(),

            input_gain: FloatParam::new(
                "Drive",
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
                "Gain",
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

            drive: FloatParam::new(
                "Shape",
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

            distortion_type: EnumParam::new("Type", DistortionType::Saturation),
        }
    }
}

impl Croaker {
    pub fn update_peak_meter(
        &self,
        amplitude: f32,
        num_samples: usize,
        peak_meter: &Arc<AtomicF32>,
    ) -> f32 {
        let amplitude = (amplitude / num_samples as f32).abs();
        let current_peak_meter = peak_meter.load(std::sync::atomic::Ordering::Relaxed);
        let new_peak_meter = if amplitude > current_peak_meter {
            amplitude
        } else {
            current_peak_meter * self.peak_meter_decay_weight
                + amplitude * (1.0 - self.peak_meter_decay_weight)
        };

        peak_meter.store(new_peak_meter, std::sync::atomic::Ordering::Relaxed);

        amplitude
    }
}

impl Plugin for Croaker {
    const NAME: &'static str = "croaker";
    const VENDOR: &'static str = "renzofrog";
    const URL: &'static str = "https://www.renzofrog.com";
    const EMAIL: &'static str = "renzomledesma@gmail.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),

        aux_input_ports: &[],
        aux_output_ports: &[],

        names: PortNames::const_default(),
    }];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            self.params.clone(),
            self.input_peak_meter.clone(),
            self.output_peak_meter.clone(),
            self.params.editor_state.clone(),
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        _buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // After `PEAK_METER_DECAY_MS` milliseconds of pure silence, the peak meter's value should
        // have dropped by 12 dB
        self.peak_meter_decay_weight = 0.25f64
            .powf((_buffer_config.sample_rate as f64 * PEAK_METER_DECAY_MS / 1000.0).recip())
            as f32;

        let fs = _buffer_config.sample_rate;
        if fs >= 88200. {
            self.oversample_factor = 1;
        } else {
            self.oversample_factor = 4;
        }
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        for channel_samples in buffer.iter_samples() {
            let input_gain = self.params.input_gain.smoothed.next();
            let output_gain = self.params.output_gain.smoothed.next();
            let drive = self.params.drive.smoothed.next();
            let dry_wet_ratio = self.params.dry_wet_ratio.smoothed.next();
            let distortion_type = self.params.distortion_type.value();

            // UI variables
            let mut input_amplitude = 0.0;
            let mut output_amplitude = 0.0;
            let num_samples = channel_samples.len();

            // Do processing
            for sample in channel_samples {
                // Apply DC filter
                *sample = self.dc_filter.process(*sample);

                // Apply input gain after visualizing original input amplitude
                input_amplitude += *sample;
                *sample *= input_gain;

                // Oversample if needed
                let processed_sample = if self.oversample_factor == 4 {
                    // Zero-stuff
                    let input = *sample;
                    let mut frame = [input, 0., 0., 0.];

                    // Apply processing
                    for i in 0..frame.len() {
                        // Run input through half-band filter
                        frame[i] = self.upsampler.process(frame[i]);

                        // Apply distortion
                        let wet = process_sample(&distortion_type, drive, frame[i]);

                        // Downsample through half-band filter
                        frame[i] = self.downsampler.process(wet);
                    }

                    // Get output after downsampling
                    frame[0]
                } else {
                    // Don't oversample if not needed
                    process_sample(&distortion_type, drive, *sample)
                };

                // Apply dry/wet and rewrite buffer
                let processed_sample =
                    (*sample * (1.0 - dry_wet_ratio)) + (processed_sample * dry_wet_ratio);

                let output = processed_sample * output_gain;
                output_amplitude += output;
                *sample = output;
            }

            // To save resources, a plugin can (and probably should!) only perform expensive
            // calculations that are only displayed on the GUI while the GUI is open
            if self.params.editor_state.is_open() {
                self.update_peak_meter(input_amplitude, num_samples, &self.input_peak_meter);
                self.update_peak_meter(output_amplitude, num_samples, &self.output_peak_meter);
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

    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Distortion];
}

nih_export_vst3!(Croaker);
