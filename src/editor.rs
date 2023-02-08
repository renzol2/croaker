use atomic_float::AtomicF32;
use nih_plug::prelude::{util, Editor, GuiContext, Param, ParamPtr};
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaState, ViziaTheming};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use std::vec;

use nih_plug_vizia::vizia::views::PopupEvent;

use crate::CroakerParams;

const STYLE: &str = include_str!("style.css");
pub const WINDOW_WIDTH: u32 = 650;
pub const WINDOW_HEIGHT: u32 = 500;

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

    DistortionEvent(usize),
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
            ParamChangeEvent::DistortionEvent(index) => {
                unsafe {
                    self.gui_context
                        .raw_begin_set_parameter(self.params.distortion_type.as_ptr())
                };
                unsafe {
                    self.gui_context.raw_set_parameter_normalized(
                        self.params.distortion_type.as_ptr(),
                        *index as f32 / self.distortion_types.len() as f32,
                    )
                };
                unsafe {
                    self.gui_context
                        .raw_end_set_parameter(self.params.distortion_type.as_ptr())
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
    ViziaState::from_size(WINDOW_WIDTH, WINDOW_HEIGHT)
}

pub(crate) fn create(
    params: Arc<CroakerParams>,
    input_peak_meter: Arc<AtomicF32>,
    output_peak_meter: Arc<AtomicF32>,
    editor_state: Arc<ViziaState>,
) -> Option<Box<dyn Editor>> {
    create_vizia_editor(editor_state, ViziaTheming::Custom, move |cx, context| {
        assets::register_noto_sans_light(cx);
        assets::register_noto_sans_thin(cx);
        cx.add_theme(STYLE);

        Data {
            gui_context: context.clone(),
            params: params.clone(),
            input_peak_meter: input_peak_meter.clone(),
            output_peak_meter: output_peak_meter.clone(),
            distortion_types: vec![
                "Saturation".to_string(),
                "Hard clipping".to_string(),
                "Fuzzy rectifier".to_string(),
                "Shockley diode rectifier".to_string(),
                "Dropout".to_string(),
                "Double soft clipper".to_string(),
                "Wavefolder".to_string(),
            ],
        }
        .build(cx);

        ResizeHandle::new(cx);

        // UI
        VStack::new(cx, |cx| {
            // Title
            Label::new(cx, "croaker")
                .font(assets::NOTO_SANS_BOLD)
                .font_size(25.0)
                .height(Pixels(50.0))
                .child_top(Stretch(1.0))
                .child_bottom(Pixels(0.0))
                .bottom(Pixels(15.0));

            // Waveshaper dropdown
            VStack::new(cx, |cx| {
                HStack::new(cx, |cx| {
                    // Dropdown to select waveshaper
                    Dropdown::new(
                        cx,
                        move |cx|
                        // A Label and an Icon
                        HStack::new(cx, move |cx|{
                            Label::new(cx, Data::params.map(|p| p.distortion_type.to_string()));
                        }).class("title"),
                        move |cx| {
                            // List of options
                            List::new(cx, Data::distortion_types, move |cx, idx, item| {
                                VStack::new(cx, move |cx| {
                                    Binding::new(
                                        cx,
                                        Data::params.map(|p| p.distortion_type.to_string()),
                                        move |cx, choice| {
                                            let selected = *item.get(cx) == *choice.get(cx);
                                            Label::new(cx, &item.get(cx))
                                                .width(Stretch(1.0))
                                                .background_color(if selected {
                                                    Color::from("#C28919")
                                                } else {
                                                    Color::transparent()
                                                })
                                                .on_press(move |cx| {
                                                    cx.emit(ParamChangeEvent::DistortionEvent(idx));
                                                    cx.emit(PopupEvent::Close);
                                                });
                                        },
                                    );
                                });
                            });
                        },
                    );
                })
                .class("waveshaper_selector");
            });

            // Knobs
            HStack::new(cx, |cx| {
                // Input gain control
                // Label::new(cx, "Gain").bottom(Pixels(-1.0));
                // ParamSlider::new(cx, Data::params, |params| &params.gain);
                make_knob(cx, params.input_gain.as_ptr(), |params| &params.input_gain);

                // Saturation control
                // Label::new(cx, "Saturation").bottom(Pixels(-1.0));
                // ParamSlider::new(cx, Data::params, |params| &params.saturation);
                make_knob(cx, params.drive.as_ptr(), |params| &params.drive);

                // Dry/wet control
                // Label::new(cx, "Saturation").bottom(Pixels(-1.0));
                // ParamSlider::new(cx, Data::params, |params| &params.saturation);
                make_knob(cx, params.dry_wet_ratio.as_ptr(), |params| {
                    &params.dry_wet_ratio
                });

                // Gain control
                // Label::new(cx, "Gain").bottom(Pixels(-1.0));
                // ParamSlider::new(cx, Data::params, |params| &params.gain);
                make_knob(cx, params.output_gain.as_ptr(), |params| {
                    &params.output_gain
                });
            })
            .class("knobs")
            .bottom(Pixels(10.0));

            HStack::new(cx, |cx| {
                // Input gain label
                Label::new(cx, "Input gain").font_size(15.0);

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
            .top(Pixels(15.0))
            .bottom(Pixels(-40.0));

            // Output gain
            HStack::new(cx, |cx| {
                // Output gain label
                Label::new(cx, "Output gain").font_size(15.0);

                // Output gain meter
                PeakMeter::new(
                    cx,
                    Data::output_peak_meter
                        .map(|peak_meter| util::gain_to_db(peak_meter.load(Ordering::Relaxed))),
                    Some(Duration::from_millis(600)),
                );
            })
            .col_between(Pixels(7.0))
            .bottom(Pixels(10.0));
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
        );

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

        // Units
        Label::new(
            cx,
            Data::params.map(move |params| params_to_param(params).to_string()),
        )
        .width(Pixels(100.));
    })
    .child_space(Stretch(0.2))
}
