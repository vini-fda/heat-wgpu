use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use wgpu_profiler::GpuProfiler;

use wgpu_profiler::GpuTimerScopeResult;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::{compute::Compute, directional_bind_group::Direction, renderer::Renderer};

struct App {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: Mutex<wgpu::SurfaceConfiguration>,
    size: Mutex<winit::dpi::PhysicalSize<u32>>,
    window: Window,
    compute: Compute,
    renderer: Renderer,
    profiler: Arc<Mutex<GpuProfiler>>,
    profiler_data: Arc<Mutex<Option<Vec<GpuTimerScopeResult>>>>,
    direction: Arc<RwLock<Direction>>,
    iteration: Arc<RwLock<u32>>,
}

impl App {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
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
                    features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                        | GpuProfiler::ALL_WGPU_TIMER_FEATURES,
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
        let mut config = surface
            .get_default_config(&adapter, size.width, size.height)
            .expect("Surface isn't supported by the adapter.");
        config.present_mode = wgpu::PresentMode::Fifo;
        let surface_view_format = config.format.add_srgb_suffix();
        config.view_formats.push(surface_view_format);
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

        let direction = Arc::new(RwLock::new(Direction::Forward));

        let compute = Compute::new(&device, direction.clone(), texture_size, txa_view, txb_view);
        let renderer = Renderer::new(&device, direction.clone(), &config, txa_view, txb_view);

        let profiler = Arc::new(Mutex::new(GpuProfiler::new(
            4,
            queue.get_timestamp_period(),
            device.features(),
        )));

        Self {
            window,
            surface,
            device,
            queue,
            config: Mutex::new(config),
            size: Mutex::new(size),
            compute,
            renderer,
            profiler,
            profiler_data: Arc::new(Mutex::new(None)),
            direction,
            iteration: Arc::new(RwLock::new(0)),
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            *self.size.lock().unwrap() = new_size;
            let mut config = self.config.lock().unwrap();
            config.width = new_size.width;
            config.height = new_size.height;
            self.surface.configure(&self.device, &config);
        }
    }

    fn compute_step(&self) {
        let mut profiler = self.profiler.lock().unwrap();
        let mut profiler_data = self.profiler_data.lock().unwrap();
        self.compute
            .step(&self.device, &self.queue, &mut profiler, &mut profiler_data);
        let mut iteration = self.iteration.write().unwrap();
        *iteration += 1;
        if *iteration % 2 == 0 {
            *self.direction.write().unwrap() = Direction::Forward;
        } else {
            *self.direction.write().unwrap() = Direction::Backward;
        }
        drop(iteration);
    }

    fn render(&self) -> Result<(), wgpu::SurfaceError> {
        let mut profiler = self.profiler.lock().unwrap();
        let mut profiler_data = self.profiler_data.lock().unwrap();
        self.renderer.render(
            &self.device,
            &self.surface,
            &self.queue,
            &mut profiler,
            &mut profiler_data,
        )
    }
}

pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    // let mut app = App::new(window).await;
    // create app to be used in two threads
    let app = Arc::new(App::new(window).await);

    // Run GPU compute in a separate thread
    {
        use std::thread;
        use std::time::{Duration, Instant};
        let target_frame_duration = Duration::from_secs_f64(1.0 / 60.0);
        let mut previous_frame_time = Instant::now();
        let app = app.clone();
        // Spawn a new thread to run at fixed timestep
        thread::spawn(move || {
            loop {
                // Perform your computation here
                app.compute_step();

                // Calculate the time elapsed since the previous frame
                let elapsed = previous_frame_time.elapsed();
                println!("Elapsed: {:?}", elapsed);

                // Check if we need to introduce a delay to maintain 60 Hz
                if elapsed < target_frame_duration {
                    thread::sleep(target_frame_duration - elapsed);
                }

                // Update the previous frame time
                previous_frame_time = Instant::now();
            }
        });
    }

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
                app.surface
                    .configure(&app.device, &app.config.lock().unwrap());
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
                } => match keycode {
                    VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                    VirtualKeyCode::Space => {
                        let profiler_data = app.profiler_data.lock().unwrap();
                        if let Some(ref data) = *profiler_data {
                            wgpu_profiler::chrometrace::write_chrometrace(
                                std::path::Path::new("trace.json"),
                                data,
                            )
                            .expect("Failed to write trace.json");
                        }
                    }
                    _ => {}
                },
                _ => {
                    // TODO example.update(event);
                }
            },
            Event::RedrawRequested(window_id) if window_id == app.window().id() => {
                //app.compute_step();

                match app.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => app.resize(*app.size.lock().unwrap()),
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
