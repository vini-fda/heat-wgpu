use winit::{
    dpi::{LogicalSize, Size},
    event::*,
    event_loop::{ControlFlow, EventLoop, EventLoopBuilder},
    window::{Window, WindowBuilder, WindowId},
};

use crate::{heat_equation::HeatEquation, renderer::Renderer};

struct App {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    heat_eqn: HeatEquation,
    renderer: Renderer,
}

impl App {
    // Creating some of the wgpu types requires async code
    fn new(window: Window) -> Self {
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
        let (size, surface) = unsafe {
            let size = window.inner_size();

            #[cfg(any(not(target_arch = "wasm32"), target_os = "emscripten"))]
            let surface = instance.create_surface(&window).unwrap();
            #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
            let surface = {
                if let Some(offscreen_canvas_setup) = &offscreen_canvas_setup {
                    log::info!("Creating surface from OffscreenCanvas");
                    instance.create_surface_from_offscreen_canvas(
                        offscreen_canvas_setup.offscreen_canvas.clone(),
                    )
                } else {
                    instance.create_surface(&window)
                }
            }
            .unwrap();

            (size, surface)
        };

        let adapter = pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::default(),
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .unwrap()
        });
        println!("Adapter: {:?}", adapter.get_info());
        println!("Surface: {:?}", surface.get_capabilities(&adapter));
        //device and queue
        let (device, queue) = pollster::block_on(async {
            adapter
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
                .unwrap()
        });
        let mut config = surface
            .get_default_config(&adapter, size.width, size.height)
            .expect("Surface isn't supported by the adapter.");
        config.present_mode = wgpu::PresentMode::Fifo;
        config.format = wgpu::TextureFormat::Bgra8Unorm;
        let surface_view_format = config.format.add_srgb_suffix();
        config.view_formats.push(surface_view_format);
        println!("Config: {:?}", config);
        surface.configure(&device, &config);

        // ------ GPU Compute config ------
        let n = 512;
        let width = n;
        let height = n;
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Heat Equation Texture"),
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
        // Initialize texture with some data
        let input_data = generate_input_data(width, height);
        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(input_data.as_slice()),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: None,
            },
            texture_size,
        );
        let texture_view = &texture.create_view(&wgpu::TextureViewDescriptor::default());

        let alpha = 2e-4;
        let dt = 0.016;

        let compute = HeatEquation::new(&device, alpha, n as usize, dt, &input_data, &texture);
        let renderer = Renderer::new(&device, &config, texture_view);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            heat_eqn: compute,
            renderer,
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

    fn compute_step(&mut self) {
        self.heat_eqn.compute_step(&self.device, &self.queue);
    }

    fn render(&self) -> Result<(), wgpu::SurfaceError> {
        self.renderer
            .render(&self.device, &self.surface, &self.queue)
    }
}

pub fn run() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_min_inner_size(Size::Logical(LogicalSize::new(640.0, 480.0)))
        .with_inner_size(Size::Logical(LogicalSize::new(1280.0, 720.0)))
        .with_title("heat-wgpu")
        .build(&event_loop)
        .unwrap();
    let mut app = App::new(window);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::RedrawEventsCleared => {
                // TODO: in the wgpu example, an async executor is used for polling
                // we need to check how to sync the compute step with the render step
                // spawner.run_until_stalled();

                app.window().request_redraw();
            }
            Event::WindowEvent {
                event:
                    WindowEvent::Resized(size)
                    | WindowEvent::ScaleFactorChanged {
                        new_inner_size: &mut size,
                        ..
                    },
                ..
            } => {
                log::info!("Resizing to {:?}", size);
                app.resize(size);
                app.surface.configure(&app.device, &app.config);
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    if keycode == VirtualKeyCode::Escape {
                        *control_flow = ControlFlow::Exit
                    }
                }
                _ => {
                    // TODO example.update(event);
                }
            },
            Event::RedrawRequested(window_id) if window_id == app.window().id() => {
                app.compute_step();

                match app.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => app.resize(app.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
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
