use atomic_float::AtomicF32;
use nih_plug::prelude::{util, Editor, GuiContext, Param, ParamPtr};
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{create_vizia_editor, ViziaState, ViziaTheming};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use std::vec;

use crate::assets::{register_fantasque_sans_mono, FANTASQUE_SANS_MONO};
use crate::CroakerParams;

const STYLE: &str = include_str!("style.css");
pub const WINDOW_WIDTH: u32 = 550;
pub const WINDOW_HEIGHT: u32 = 600;

#[derive(Lens)]
struct Data {
    pub gui_context: Arc<dyn GuiContext>,
    params: Arc<CroakerParams>,
    input_peak_meter: Arc<AtomicF32>,
    output_peak_meter: Arc<AtomicF32>,
    distortion_types: Vec<String>,
}

// `ParamChangeEvent` enum credits to Fredemus and geom3trik
// https://github.com/Fredemus/va-filter/blob/b3a3b2ae4004738fb7ae5a114e372fdda6213fb1/src/ui.rs#L36
// Under GPLv3 License
#[derive(Debug)]
pub enum ParamChangeEvent {
    BeginSet(ParamPtr),
    EndSet(ParamPtr),
    SetParam(ParamPtr, f32),
}

impl Model for Data {
    fn event(&mut self, _cx: &mut EventContext, event: &mut Event) {
        event.map(|event, _| match event {
            ParamChangeEvent::SetParam(param_ptr, new_value) => {
                unsafe {
                    self.gui_context
                        .raw_set_parameter_normalized(*param_ptr, *new_value)
                };
            }
            ParamChangeEvent::BeginSet(param_ptr) => {
                unsafe { self.gui_context.raw_begin_set_parameter(*param_ptr) };
            }
            ParamChangeEvent::EndSet(param_ptr) => {
                unsafe { self.gui_context.raw_end_set_parameter(*param_ptr) };
            }
        })
    }
}

// Makes sense to also define this here, makes it a bit easier to keep track of
pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (WINDOW_WIDTH, WINDOW_HEIGHT))
}

pub(crate) fn create(
    params: Arc<CroakerParams>,
    input_peak_meter: Arc<AtomicF32>,
    output_peak_meter: Arc<AtomicF32>,
    editor_state: Arc<ViziaState>,
) -> Option<Box<dyn Editor>> {
    create_vizia_editor(editor_state, ViziaTheming::Custom, move |cx, context| {
        register_fantasque_sans_mono(cx);
        cx.add_theme(STYLE);

        Data {
            gui_context: context.clone(),
            params: params.clone(),
            input_peak_meter: input_peak_meter.clone(),
            output_peak_meter: output_peak_meter.clone(),
            distortion_types: vec![
                "Saturate".to_string(),
                "Hard clip".to_string(),
                "Fuzzy rectifier".to_string(),
                "Diode rectifier".to_string(),
                "Dropout".to_string(),
                "Double soft clip".to_string(),
                "Wavefolder".to_string(),
            ],
        }
        .build(cx);

        ResizeHandle::new(cx);

        // UI
        VStack::new(cx, |cx| {
            // Title
            Label::new(cx, "🐸🌱💫")
                .font_family(vec![FamilyOwned::Name(String::from(FANTASQUE_SANS_MONO))])
                .font_size(30.0)
                .height(Pixels(50.0))
                .child_top(Stretch(1.0));

            // Distortion knobs
            Label::new(cx, "Distortion")
                .font_family(vec![FamilyOwned::Name(String::from(FANTASQUE_SANS_MONO))])
                .font_size(15.0)
                .height(Pixels(50.0))
                .font_weight(Weight::BOLD)
                .child_top(Stretch(1.0))
                .top(Pixels(-10.0))
                .bottom(Pixels(-10.0));
            HStack::new(cx, |cx| {
                // Input gain control
                make_knob(cx, params.drive.as_ptr(), |params| &params.drive);

                // Saturation control
                make_knob(cx, params.shape.as_ptr(), |params| &params.shape);

                // Distortion type
                make_knob(cx, params.distortion_type.as_ptr(), |params| {
                    &params.distortion_type
                });

                // Output gain control
                make_knob(cx, params.output_gain.as_ptr(), |params| {
                    &params.output_gain
                });
            })
            .child_space(Pixels(5.0))
            .bottom(Pixels(-10.0))
            .class("knobs");

            // Vibrato knobs
            Label::new(cx, "Tape")
                .font_family(vec![FamilyOwned::Name(String::from(FANTASQUE_SANS_MONO))])
                .font_size(15.0)
                .height(Pixels(50.0))
                .font_weight(Weight::BOLD)
                .child_top(Stretch(1.0))
                .top(Pixels(-20.0))
                .bottom(Pixels(-10.0));
            HStack::new(cx, |cx| {
                // Wow control
                make_knob(cx, params.wow.as_ptr(), |params| &params.wow);

                // Flutter control
                make_knob(cx, params.flutter.as_ptr(), |params| &params.flutter);

                // Width control
                make_knob(cx, params.width.as_ptr(), |params| &params.width);

                // Dry/wet control
                make_knob(cx, params.dry_wet_ratio.as_ptr(), |params| {
                    &params.dry_wet_ratio
                });
            })
            .child_space(Pixels(4.0))
            .class("knobs")
            .bottom(Pixels(15.0));

            HStack::new(cx, |cx| {
                // Meters
                VStack::new(cx, |cx| {
                    // Input gain
                    VStack::new(cx, |cx| {
                        // Input gain label
                        Label::new(cx, "Input level").font_size(15.0);

                        // Input gain meter
                        PeakMeter::new(
                            cx,
                            Data::input_peak_meter.map(|input_peak_meter| {
                                util::gain_to_db(input_peak_meter.load(Ordering::Relaxed))
                            }),
                            Some(Duration::from_millis(600)),
                        );
                    })
                    .col_between(Pixels(7.0))
                    .top(Pixels(0.0))
                    .bottom(Pixels(-20.0)); // meters can be a little closer

                    // Output gain
                    VStack::new(cx, |cx| {
                        // Output gain label
                        Label::new(cx, "Output level").font_size(15.0);

                        // Output gain meter
                        PeakMeter::new(
                            cx,
                            Data::output_peak_meter.map(|peak_meter| {
                                util::gain_to_db(peak_meter.load(Ordering::Relaxed))
                            }),
                            Some(Duration::from_millis(600)),
                        );
                    })
                    .child_space(Pixels(10.0))
                    .bottom(Pixels(10.0));
                })
                .child_left(Stretch(1.0))
                .child_right(Stretch(1.0))
                .class("gain_meters");

                // Bypass button
                Button::new(
                    cx,
                    |cx| {
                        // FIXME: doing it this way causes a debug assertion error.
                        // not sure how to fix this, but it's not breaking anything major...
                        cx.emit(ParamChangeEvent::SetParam(
                            Data::params.get(cx).bypass.as_ptr(),
                            // We pass the opposite value of the current state as a float, because parameter is a float
                            !Data::params.get(cx).bypass.value() as i32 as f32,
                        ));
                    },
                    |cx| Label::new(cx, "Bypass"),
                )
                .top(Pixels(42.0))
                .class("bypass");
            })
            .display(Display::Flex)
            .col_between(Pixels(10.0));

            Label::new(cx, "croaker is a renzofrog plugin. handle with care 💜")
                .font_family(vec![FamilyOwned::Name(String::from(FANTASQUE_SANS_MONO))])
                .font_style(FontStyle::Italic)
                .font_size(10.0)
                .child_bottom(Stretch(1.0));
        })
        .row_between(Pixels(0.0))
        .child_left(Stretch(1.0))
        .child_right(Stretch(1.0))
        .class("container");
    })
}

// Knob helper function credits to Fredemus and geom3trik
// https://github.com/Fredemus/va-filter/blob/b3a3b2ae4004738fb7ae5a114e372fdda6213fb1/src/ui.rs#L192
// Under GPLv3 License
fn make_knob<P, F>(cx: &mut Context, param_ptr: ParamPtr, params_to_param: F) -> Handle<VStack>
where
    P: Param,
    F: 'static + Fn(&Arc<CroakerParams>) -> &P + Copy,
{
    VStack::new(cx, move |cx| {
        // Param name
        Label::new(
            cx,
            Data::params.map(move |params| params_to_param(params).name().to_owned()),
        )
        .bottom(Pixels(7.0));

        // Units
        Label::new(
            cx,
            Data::params.map(move |params| params_to_param(params).to_string()),
        )
        .font_size(11.0);

        // Knob
        Knob::custom(
            cx,
            0.5,
            Data::params.map(move |params| params_to_param(params).modulated_normalized_value()),
            move |cx, lens| {
                TickKnob::new(
                    cx,
                    Percentage(80.0),
                    Pixels(4.),
                    Percentage(50.0),
                    270.0,
                    KnobMode::Continuous,
                )
                .value(lens.clone())
                .class("tick");
                ArcTrack::new(
                    cx,
                    false,
                    Percentage(100.0),
                    Percentage(10.),
                    -135.,
                    135.,
                    KnobMode::Continuous,
                )
                .value(lens)
                .class("track")
            },
        )
        .on_changing(move |cx, val| {
            cx.emit(
                // setter.set_parameter_normalized(param, val);
                // ParamChangeEvent::AllParams(param_index, val),
                ParamChangeEvent::SetParam(param_ptr, val),
            )
        })
        .on_mouse_down(move |cx, _| {
            cx.emit(
                // setter.set_parameter_normalized(param, val);
                ParamChangeEvent::BeginSet(param_ptr),
            )
        })
        .on_mouse_up(move |cx, _| {
            cx.emit(
                // setter.set_parameter_normalized(param, val);
                ParamChangeEvent::EndSet(param_ptr),
            )
        });
    })
    .child_space(Stretch(0.1))
    .child_left(Stretch(1.0))
    .child_right(Stretch(1.0))
    .width(Pixels(90.))
}
