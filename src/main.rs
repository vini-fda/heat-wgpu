use heat_wgpu::app::App;
use winit::{event_loop::EventLoop, window::WindowBuilder, event::{Event, WindowEvent}, keyboard::{KeyCode, PhysicalKey::Code}};



fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("Event loop creation failed");
    let window = WindowBuilder::new()
        .build(&event_loop)
        .expect("Window builder creation failed");
    let mut app = pollster::block_on(App::new(&window));

    event_loop.run(move |event, event_loop_window_target| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                match &event {
                    WindowEvent::Resized(physical_size) => app.resize(*physical_size),
                    WindowEvent::ScaleFactorChanged { .. } => {
                        app.resize(window.inner_size());
                    }
                    WindowEvent::CloseRequested => event_loop_window_target.exit(),
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state.is_pressed()
                            && matches!(event.physical_key, Code(KeyCode::Escape))
                        {
                            event_loop_window_target.exit();
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        app.update();

                        match app.render(&window) {
                            Ok(_) => {}
                            // Recreate the swap_chain if lost
                            Err(wgpu::SurfaceError::Lost) => app.reacquire_size(),
                            // The system is out of memory, we should probably quit
                            Err(wgpu::SurfaceError::OutOfMemory) => event_loop_window_target.exit(),
                            // All other errors (Outdated, Timeout) should be resolved by the next frame
                            Err(e) => eprintln!("Unhandled error: {:?}", e),
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window.request_redraw()
            }
            _ => {}
        }
    }).unwrap();
}
