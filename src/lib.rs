pub mod compute;
mod directional_bind_group;
pub mod renderer;
pub mod vertex;
use std::cell::Cell;

use std::{rc::Rc, sync::Arc};

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::{compute::Compute, directional_bind_group::Direction, renderer::Renderer};

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    compute: Compute,
    renderer: Renderer,
    direction: Rc<Cell<Direction>>,
    iteration: u32,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        println!("Adapter: {:?}", adapter.get_info());
        //device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();
        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        // IMPROVEMENT TO TUTORIAL: https://sotrh.github.io/learn-wgpu/beginner/tutorial2-surface/#state-new
        // use .is_srgb() instead of .describe().srgb
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // ------ GPU Compute config ------
        let width = 512;
        let height = 512;
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture_a = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture A"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Initialize texture A with some data
        let input_data = generate_input_data(width, height);
        queue.write_texture(
            texture_a.as_image_copy(),
            bytemuck::cast_slice(input_data.as_slice()),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: None,
            },
            texture_size,
        );
        let texture_b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture B"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let txa_view = &texture_a.create_view(&wgpu::TextureViewDescriptor::default());
        let txb_view = &texture_b.create_view(&wgpu::TextureViewDescriptor::default());

        let direction = Rc::new(Cell::new(Direction::Forward));

        let compute = Compute::new(&device, direction.clone(), texture_size, txa_view, txb_view);
        let renderer = Renderer::new(&device, direction.clone(), &config, txa_view, txb_view);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            compute,
            renderer,
            direction,
            iteration: 0,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    fn compute_step(&mut self) {
        self.compute.step(&mut self.device, &self.queue);
        self.iteration += 1;
        let d = self.direction.as_ref();
        match d.get() {
            Direction::Forward => {
                d.set(Direction::Backward);
            }
            Direction::Backward => {
                d.set(Direction::Forward);
            }
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.renderer
            .render(&mut self.device, &self.surface, &self.queue)
    }
}

pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let mut state = State::new(window).await;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                state.compute_step();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                state.window().request_redraw();
            }
            _ => {}
        }
    });
}

fn gaussian(x: f32) -> f32 {
    const SIGMA: f32 = 0.5;
    const SIGMA2: f32 = SIGMA * SIGMA;
    const ROOT_2PI: f32 = 2.5066283;
    (1.0 / (ROOT_2PI * SIGMA)) * (-x * x / (2.0 * SIGMA2)).exp()
}

fn generate_input_data(width: u32, height: u32) -> Vec<f32> {
    use noise::{NoiseFn, Perlin};
    let mut data = vec![0.0; (width * height) as usize];
    const WIDTH: f32 = 1.0;
    const HEIGHT: f32 = 1.0;
    let perlin = Perlin::new(1);
    for i in 0..width {
        for j in 0..height {
            let x = (i as f32 / width as f32) * WIDTH;
            let y = (j as f32 / height as f32) * HEIGHT;
            let x = x - WIDTH / 2.0;
            let y = y - HEIGHT / 2.0;
            let r = (x * x + y * y).sqrt();
            let noise = perlin.get([x as f64 * 10.0, y as f64 * 10.0, 0.0]) as f32;
            data[(i + j * width) as usize] = gaussian(r) + noise;
        }
    }
    data
}
