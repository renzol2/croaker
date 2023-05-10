use atomic_float::AtomicF32;
use fx::digital::{bitcrush_sample, floating_point_quantize};
use fx::moorer_verb::MoorerReverb;
use nih_plug::prelude::*;
use nih_plug_vizia::ViziaState;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use fx::{
    biquad::{BiquadFilterType, StereoBiquadFilter},
    dc_filter::DcFilter,
    delay_line::StereoDelay,
    oversampling::HalfbandFilter,
    waveshapers::*,
    DEFAULT_SAMPLE_RATE, FLUTTER_MAX_FREQUENCY_RATIO, FLUTTER_MAX_LFO_FREQUENCY,
    MAX_DELAY_TIME_SECONDS, WOW_MAX_FREQUENCY_RATIO, WOW_MAX_LFO_FREQUENCY,
};

mod assets;
mod editor;
mod fonts;

/// After `PEAK_METER_DECAY_MS` milliseconds of pure silence, the peak meter's value should have dropped by 12 dB
const PEAK_METER_DECAY_MS: f64 = 150.0;
/// The cutoff frequency for pre/post filtering
const FILTER_CUTOFF_HZ: f32 = 8000.0;
/// Multiplier to oversample by
const OVERSAMPLING_FACTOR: usize = 4;
/// Min number of bits for bitcrushing
const MIN_BITS: usize = 2;
/// Max number of bits for bitcrushing
const MAX_BITS: usize = 32;
/// Maximum constant value to add for floating point rounding error
const MAX_CONSTANT: f32 = 1_000_000.;
/// Max decimate value, because going past this value literally causes silence
const MAX_DECIMATE_VALUE: f32 = 0.9;

// TODO: rename everything here to be flanger
// Chorus constants
const CHORUS_LFO_RATE_HZ: f32 = 0.1;
const CHORUS_LFO_AMOUNT: f32 = 0.03;
const CHORUS_WIDTH: f32 = 0.03;
const CHORUS_FEEDBACK: f32 = 0.85;

#[derive(Enum, Debug, PartialEq, Eq, Clone, Copy)]
/// Distortion algorithms available in plugin
pub enum DistortionType {
    #[id = "saturation"]
    #[name = "Saturator"]
    Saturation,

    #[id = "hard-clipping"]
    #[name = "Hard clipper"]
    HardClipping,

    #[id = "harder-clipping"]
    #[name = "Harder clipper"]
    HarderClipping,

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
    #[name = "Double soft clip"]
    DoubleSoftClipper,

    #[id = "wavefolding"]
    #[name = "Wavefolder"]
    Wavefolding,
}

/// Run sample through nonlinear waveshapers from `fx` depending on the distortion type
pub fn distort_sample(distortion_type: &DistortionType, drive: f32, input_sample: f32) -> f32 {
    match distortion_type {
        DistortionType::Saturation => get_saturator_output(drive, input_sample),
        DistortionType::HardClipping => get_hard_clipper_output(drive, input_sample),
        DistortionType::HarderClipping => get_saturating_hard_clipper_output(drive, input_sample),
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

    // Distortion
    upsampler: (HalfbandFilter, HalfbandFilter),
    downsampler: (HalfbandFilter, HalfbandFilter),
    dc_filters: (DcFilter, DcFilter),
    prefilter: StereoBiquadFilter,
    postfilter: StereoBiquadFilter,
    oversample_factor: usize,

    // Vibrato
    wow_vibrato: StereoDelay,
    flutter_vibrato: StereoDelay,

    // Reverb section
    reverb: MoorerReverb,

    // Filter section
    lpf: StereoBiquadFilter,
    should_update_lpf: Arc<AtomicBool>,
    hpf: StereoBiquadFilter,
    should_update_hpf: Arc<AtomicBool>,

    // Fun section
    chorus: StereoDelay,
}

#[derive(Params)]
struct CroakerParams {
    #[persist = "editor-state"]
    editor_state: Arc<ViziaState>,

    // Distortion parameters
    #[id = "drive"]
    pub drive: FloatParam,
    #[id = "output-gain"]
    pub output_gain: FloatParam,
    #[id = "shape"]
    pub shape: FloatParam,
    #[id = "distortion-type"]
    pub distortion_type: EnumParam<DistortionType>,

    // Vibrato parameters
    #[id = "wow"]
    pub wow: FloatParam,
    #[id = "flutter"]
    pub flutter: FloatParam,
    #[id = "width"]
    pub width: FloatParam,

    // Reverb parameters
    #[id = "room-size"]
    pub room_size: FloatParam,
    #[id = "dampening"]
    pub damping: FloatParam,
    #[id = "reverb_width"]
    pub reverb_width: FloatParam,
    #[id = "reverb-dry-wet"]
    pub reverb_dry_wet: FloatParam,

    // Filter section parameters
    #[id = "lpf-freq"]
    pub lpf_freq: FloatParam,
    #[id = "hpf-freq"]
    pub hpf_freq: FloatParam,
    #[id = "resonance"]
    pub resonance: FloatParam,

    #[id = "crush"]
    pub decimate: FloatParam,
    #[id = "hiss"]
    pub crunch: FloatParam,
    #[id = "chorus"]
    pub chorus: FloatParam,

    // General parameters
    #[id = "dry-wet"]
    pub dry_wet_ratio: FloatParam,
    #[id = "bypass"]
    pub bypass: BoolParam,
}

impl Default for Croaker {
    fn default() -> Self {
        // Setup distortion pre/post filters
        let mut prefilter = StereoBiquadFilter::new();
        let mut postfilter = StereoBiquadFilter::new();

        // Biquad parameters tuned by ear
        let fc = FILTER_CUTOFF_HZ / DEFAULT_SAMPLE_RATE as f32; // hz, using default sample rate
        let gain = 18.0; // dB
        let q = 0.1;
        prefilter.set_biquads(BiquadFilterType::HighShelf, fc, q, gain);
        postfilter.set_biquads(BiquadFilterType::LowShelf, fc, q, -gain);

        // Setup filter-section filters
        let mut lpf = StereoBiquadFilter::new();
        let mut hpf = StereoBiquadFilter::new();

        // Biquad parameters tuned by ear
        lpf.set_filter_type(BiquadFilterType::LowPass);
        hpf.set_filter_type(BiquadFilterType::HighPass);

        // Thread-shared variables
        let should_update_lpf = Arc::new(AtomicBool::new(true));
        let should_update_hpf = Arc::new(AtomicBool::new(true));

        Self {
            params: Arc::new(CroakerParams::new(
                should_update_lpf.clone(),
                should_update_hpf.clone(),
            )),

            peak_meter_decay_weight: 1.0,
            output_peak_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            input_peak_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),

            upsampler: (HalfbandFilter::new(8, true), HalfbandFilter::new(8, true)),
            downsampler: (HalfbandFilter::new(8, true), HalfbandFilter::new(8, true)),
            dc_filters: (DcFilter::default(), DcFilter::default()),
            prefilter,
            postfilter,
            oversample_factor: OVERSAMPLING_FACTOR,

            wow_vibrato: StereoDelay::new(MAX_DELAY_TIME_SECONDS, DEFAULT_SAMPLE_RATE),
            flutter_vibrato: StereoDelay::new(MAX_DELAY_TIME_SECONDS, DEFAULT_SAMPLE_RATE),

            reverb: MoorerReverb::new(DEFAULT_SAMPLE_RATE),

            lpf,
            should_update_lpf,
            hpf,
            should_update_hpf,

            chorus: StereoDelay::new(MAX_DELAY_TIME_SECONDS, DEFAULT_SAMPLE_RATE),
        }
    }
}

impl CroakerParams {
    fn new(should_update_lpf: Arc<AtomicBool>, should_update_hpf: Arc<AtomicBool>) -> Self {
        Self {
            editor_state: editor::default_state(),

            drive: FloatParam::new(
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

            shape: FloatParam::new(
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
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(50.0))
            .with_value_to_string(formatters::v2s_f32_rounded(2)),

            distortion_type: EnumParam::new("Type", DistortionType::Saturation),

            bypass: BoolParam::new("Bypass", false),

            wow: FloatParam::new(
                "Wow",
                0.05,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 1.0,
                    factor: FloatRange::skew_factor(-1.5),
                },
            )
            .with_smoother(SmoothingStyle::Exponential(50.0))
            .with_value_to_string(formatters::v2s_f32_rounded(2)),

            flutter: FloatParam::new(
                "Flutter",
                0.1,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 1.0,
                    factor: FloatRange::skew_factor(-1.5),
                },
            )
            .with_smoother(SmoothingStyle::Exponential(50.0))
            .with_value_to_string(formatters::v2s_f32_rounded(2)),

            width: FloatParam::new("Width", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Exponential(50.0))
                .with_value_to_string(formatters::v2s_f32_rounded(2)),

            reverb_dry_wet: FloatParam::new(
                "Dry/wet",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(50.0))
            .with_value_to_string(formatters::v2s_f32_rounded(2)),

            room_size: FloatParam::new("Room size", 0.2, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Linear(50.0))
                .with_value_to_string(formatters::v2s_f32_rounded(2)),

            damping: FloatParam::new("Damping", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Linear(50.0))
                .with_value_to_string(formatters::v2s_f32_rounded(2)),

            reverb_width: FloatParam::new("Width", 0.8, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Linear(50.0))
                .with_value_to_string(formatters::v2s_f32_rounded(2)),

            lpf_freq: FloatParam::new(
                "LPF Freq.",
                20_000.,
                FloatRange::Skewed {
                    min: 15.0,
                    max: 22_000.0,
                    factor: FloatRange::skew_factor(-2.2),
                },
            )
            .with_callback(Arc::new({
                let should_update_lpf = should_update_lpf.clone();
                move |_| should_update_lpf.store(true, Ordering::SeqCst)
            }))
            .with_smoother(SmoothingStyle::Logarithmic(20.0))
            .with_unit(" Hz")
            .with_value_to_string(formatters::v2s_f32_rounded(2)),

            hpf_freq: FloatParam::new(
                "HPF Freq.",
                20.,
                FloatRange::Skewed {
                    min: 15.0,
                    max: 22_000.0,
                    factor: FloatRange::skew_factor(-2.2),
                },
            )
            .with_callback(Arc::new({
                let should_update_hpf = should_update_hpf.clone();
                move |_| should_update_hpf.store(true, Ordering::SeqCst)
            }))
            .with_smoother(SmoothingStyle::Logarithmic(20.0))
            .with_unit(" Hz")
            .with_value_to_string(formatters::v2s_f32_rounded(2)),

            resonance: FloatParam::new(
                "Resonance",
                0.7,
                FloatRange::Skewed {
                    min: 0.1,
                    max: 18.0,
                    factor: FloatRange::skew_factor(-2.2),
                },
            )
            .with_callback(Arc::new({
                let should_update_lpf = should_update_lpf.clone();
                let should_update_hpf = should_update_hpf.clone();
                move |_| {
                    should_update_lpf.store(true, Ordering::SeqCst);
                    should_update_hpf.store(true, Ordering::SeqCst);
                }
            }))
            .with_smoother(SmoothingStyle::Logarithmic(20.0))
            .with_value_to_string(formatters::v2s_f32_rounded(2)),

            decimate: FloatParam::new(
                "Decimate",
                0.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 1.0,
                    factor: FloatRange::skew_factor(1.0),
                },
            )
            .with_smoother(SmoothingStyle::Exponential(50.0))
            .with_value_to_string(formatters::v2s_f32_rounded(2)),

            crunch: FloatParam::new(
                "Crunch",
                0.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 1.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_smoother(SmoothingStyle::Exponential(50.0))
            .with_value_to_string(formatters::v2s_f32_rounded(2)),

            chorus: FloatParam::new("Flanger", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Exponential(50.0))
                .with_value_to_string(formatters::v2s_f32_rounded(2)),
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

    pub fn check_and_update_filter_coefficients(&mut self, sample_rate: f32) {
        let q = self.params.resonance.smoothed.next();
        self.update_lpf_coefficients(sample_rate, q);
        self.update_hpf_coefficients(sample_rate, q);
    }

    pub fn update_hpf_coefficients(&mut self, sample_rate: f32, q: f32) {
        // Check if we should update HPF coefficients
        if self
            .should_update_hpf
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let frequency = self.params.hpf_freq.smoothed.next();
            let fc = frequency / sample_rate;

            self.hpf.set_biquads(BiquadFilterType::HighPass, fc, q, 0.0);
        }
    }

    pub fn update_lpf_coefficients(&mut self, sample_rate: f32, q: f32) {
        // Check if we should update LPF coefficients
        if self
            .should_update_lpf
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let frequency = self.params.lpf_freq.smoothed.next();
            let fc = frequency / sample_rate;

            self.lpf.set_biquads(BiquadFilterType::LowPass, fc, q, 0.0);
        }
    }

    fn update_filter_section(&mut self, sample_rate: f32) {
        if self.params.resonance.smoothed.is_smoothing() {
            let q = self.params.resonance.smoothed.next();
            self.lpf.set_q(q);
            self.hpf.set_q(q);
        }
        if self.params.lpf_freq.smoothed.is_smoothing() {
            self.lpf
                .set_fc(self.params.lpf_freq.smoothed.next() / sample_rate);
        }
        if self.params.hpf_freq.smoothed.is_smoothing() {
            self.hpf
                .set_fc(self.params.hpf_freq.smoothed.next() / sample_rate);
        }
    }

    fn update_reverb(&mut self) {
        let room_size_smoothed = &self.params.room_size.smoothed;
        let damping_smoothed = &self.params.damping.smoothed;
        let width_smoothed = &self.params.reverb_width.smoothed;

        // Update reverbs while parameters smooth
        if room_size_smoothed.is_smoothing() {
            self.reverb.set_room_size(room_size_smoothed.next());
        }
        if damping_smoothed.is_smoothing() {
            self.reverb.set_damping(damping_smoothed.next());
        }
        if width_smoothed.is_smoothing() {
            self.reverb.set_width(width_smoothed.next());
        }
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
        self.peak_meter_decay_weight = 0.25f64
            .powf((_buffer_config.sample_rate as f64 * PEAK_METER_DECAY_MS / 1000.0).recip())
            as f32;

        let fs = _buffer_config.sample_rate;
        if fs >= 88200. {
            self.oversample_factor = 1;
        } else {
            self.oversample_factor = OVERSAMPLING_FACTOR;
        }

        self.reverb
            .generate_filters(_buffer_config.sample_rate as usize);
        self.update_reverb();

        self.chorus
            .resize_buffers(MAX_DELAY_TIME_SECONDS, _buffer_config.sample_rate as usize);
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // If the plugin is bypassed, stop updating the meters and prevent processing
        if self.params.bypass.value() {
            self.update_peak_meter(0., buffer.samples(), &self.input_peak_meter);
            self.update_peak_meter(0., buffer.samples(), &self.output_peak_meter);
            return ProcessStatus::Normal;
        }

        let sample_rate = _context.transport().sample_rate;
        self.check_and_update_filter_coefficients(sample_rate);

        for mut channel_samples in buffer.iter_samples() {
            // Update processors while smoothing
            self.update_reverb();
            self.update_filter_section(sample_rate);

            // Distortion parameters
            let drive = self.params.drive.smoothed.next();
            let output_gain = self.params.output_gain.smoothed.next();
            let shape = self.params.shape.smoothed.next();
            let dry_wet_ratio = self.params.dry_wet_ratio.smoothed.next();
            let distortion_type = self.params.distortion_type.value();

            // Vibrato parameters
            let wow = self.params.wow.smoothed.next();
            let flutter = self.params.flutter.smoothed.next();
            let width = self.params.width.smoothed.next();
            let phase_offset = width * 0.5; // only offset right phase by a maximum of 180 degrees

            // UI variables
            let mut input_amplitude = 0.0;
            let mut output_amplitude = 0.0;
            let num_samples = channel_samples.len();

            // Get input signal, and update input peak meter value
            let in_l = *channel_samples.get_mut(0).unwrap();
            let in_r = *channel_samples.get_mut(1).unwrap();
            input_amplitude += in_l + in_r;

            // Remove DC offset
            let processed_l = self.dc_filters.0.process(in_l) * drive;
            let processed_r = self.dc_filters.1.process(in_r) * drive;

            // Apply processing
            let (wet_l, wet_r) = if self.oversample_factor == OVERSAMPLING_FACTOR {
                // Begin upsampling block
                let mut frame_l = [processed_l, 0., 0., 0.];
                let mut frame_r = [processed_r, 0., 0., 0.];

                for i in 0..OVERSAMPLING_FACTOR {
                    // Upsample
                    frame_l[i] = self.upsampler.0.process(frame_l[i]);
                    frame_r[i] = self.upsampler.1.process(frame_r[i]);

                    // Apply pre-filtering
                    let prefiltered = self.prefilter.process((frame_l[i], frame_r[i]));
                    frame_l[i] = prefiltered.0;
                    frame_r[i] = prefiltered.1;

                    // Apply distortion
                    frame_l[i] = distort_sample(&distortion_type, shape, frame_l[i]);
                    frame_r[i] = distort_sample(&distortion_type, shape, frame_r[i]);

                    // Apply post-filtering
                    let postfiltered = self.postfilter.process((frame_l[i], frame_r[i]));
                    frame_l[i] = postfiltered.0;
                    frame_r[i] = postfiltered.1;

                    if wow > 0.0 {
                        let wow_processed_samples = self.wow_vibrato.process_with_vibrato(
                            (frame_l[i], frame_r[i]),
                            WOW_MAX_LFO_FREQUENCY / OVERSAMPLING_FACTOR as f32,
                            wow * WOW_MAX_FREQUENCY_RATIO / (OVERSAMPLING_FACTOR as f32).powf(2.),
                            phase_offset,
                        );
                        frame_l[i] = wow_processed_samples.0;
                        frame_r[i] = wow_processed_samples.1;
                    }

                    // Apply flutter
                    if flutter > 0.0 {
                        let flutter_processed_samples = self.flutter_vibrato.process_with_vibrato(
                            (frame_l[i], frame_r[i]),
                            FLUTTER_MAX_LFO_FREQUENCY / OVERSAMPLING_FACTOR as f32,
                            flutter * FLUTTER_MAX_FREQUENCY_RATIO
                                / (OVERSAMPLING_FACTOR as f32).powf(2.),
                            phase_offset,
                        );
                        frame_l[i] = flutter_processed_samples.0;
                        frame_r[i] = flutter_processed_samples.1;
                    }

                    // Downsample through half-band filter
                    frame_l[i] = self.downsampler.0.process(frame_l[i]);
                    frame_r[i] = self.downsampler.1.process(frame_r[i]);
                }

                (frame_l[0], frame_r[0])
            } else {
                let distorted_l = distort_sample(&distortion_type, shape, processed_l);
                let distorted_r = distort_sample(&distortion_type, shape, processed_r);
                // TODO: add wow/flutter
                (distorted_l, distorted_r)
            };

            // Apply dry/wet
            let out_l = (in_l * (1.0 - dry_wet_ratio)) + (wet_l * dry_wet_ratio);
            let out_r = (in_r * (1.0 - dry_wet_ratio)) + (wet_r * dry_wet_ratio);

            // Add reverb
            let (reverb_l, reverb_r) = self.reverb.tick((out_l, out_r));
            let reverb_dry_wet = self.params.reverb_dry_wet.smoothed.next();
            let out_l = out_l * (1. - reverb_dry_wet) + reverb_l * reverb_dry_wet;
            let out_r = out_r * (1. - reverb_dry_wet) + reverb_r * reverb_dry_wet;

            // Run signal through filter section
            let (out_l, out_r) = self.lpf.process((out_l, out_r));
            let (out_l, out_r) = self.hpf.process((out_l, out_r));

            // Run signal through... fun things :)
            // Bitcrush
            let decimate = self.params.decimate.smoothed.next();
            let (out_l, out_r) = if decimate > 0.0 {
                let decimate = decimate * MAX_DECIMATE_VALUE;
                let bits = -decimate * (MAX_BITS - MIN_BITS) as f32 + MAX_BITS as f32;
                let bitcrushed = (bitcrush_sample(out_l, bits), bitcrush_sample(out_r, bits));
                (
                    get_wavefolder_output(decimate, bitcrushed.0),
                    get_wavefolder_output(decimate, bitcrushed.1),
                )
            } else {
                (out_l, out_r)
            };

            // tom7's floating point addition rounding error
            let crunch = self.params.crunch.smoothed.next();
            let (out_l, out_r) = (
                floating_point_quantize(out_l, crunch * MAX_CONSTANT),
                floating_point_quantize(out_r, crunch * MAX_CONSTANT),
            );

            // Apply chorus
            let chorus_depth = self.params.chorus.smoothed.next();
            let (out_l, out_r) = self.chorus.process_with_chorus(
                (out_l, out_r),
                CHORUS_LFO_RATE_HZ,
                CHORUS_LFO_AMOUNT,
                CHORUS_WIDTH,
                chorus_depth,
                CHORUS_FEEDBACK,
            );

            // Apply output gain, and update output peak meter value
            let out_l = out_l * output_gain;
            let out_r = out_r * output_gain;
            output_amplitude += out_l + out_r;

            *channel_samples.get_mut(0).unwrap() = out_l;
            *channel_samples.get_mut(1).unwrap() = out_r;

            // Update peak meters if UI is open
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
    const CLAP_DESCRIPTION: Option<&'static str> = Some("make sound go croak");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Distortion,
        ClapFeature::Filter,
        ClapFeature::Reverb,
    ];
}

impl Vst3Plugin for Croaker {
    const VST3_CLASS_ID: [u8; 16] = *b"renzofrogcroaker";

    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Distortion,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Reverb,
    ];
}

nih_export_clap!(Croaker);
nih_export_vst3!(Croaker);
