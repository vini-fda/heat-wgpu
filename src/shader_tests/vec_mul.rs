#[cfg(test)]
mod tests {
    use wgpu::util::DeviceExt;

    async fn execute_gpu(vec_a: &[f32], vec_b: &[f32]) -> Option<Vec<f32>> {
        // Instantiates instance of WebGPU
        let instance = wgpu::Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        let info = adapter.get_info();
        // skip this on LavaPipe temporarily
        if info.vendor == 0x10005 {
            return None;
        }

        execute_gpu_inner(&device, &queue, vec_a, vec_b).await
    }

    async fn execute_gpu_inner(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vec_a: &[f32],
        vec_b: &[f32],
    ) -> Option<Vec<f32>> {
        // Loads the shader from WGSL
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vec_mul.wgsl").into()),
        });

        // Gets the size in bytes of the buffer.
        let slice_size = vec_a.len() * std::mem::size_of::<u32>();
        let size = slice_size as wgpu::BufferAddress;

        // Instantiates buffer without data.
        // `usage` of buffer specifies how it can be used:
        //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
        //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Instantiates buffer with data (`vec_a`).
        // Usage allowing the buffer to be:
        //   A storage buffer (can be bound within a bind group and thus available to a shader).
        //   The destination of a copy.
        //   The source of a copy.
        let storage_buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer A"),
            contents: bytemuck::cast_slice(vec_a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let storage_buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer B"),
            contents: bytemuck::cast_slice(vec_b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Stores the result of the computation
        let storage_buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Storage Buffer C"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // A bind group defines how buffers are accessed by shaders.
        // It is to WebGPU what a descriptor set is to Vulkan.
        // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

        // A pipeline specifies the operation of a shader

        // Instantiates the pipeline.
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &cs_module,
            entry_point: "main",
        });

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage_buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage_buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: storage_buffer_c.as_entire_binding(),
                },
            ],
        });

        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("compute vector element-wise multiplication");
            cpass.dispatch_workgroups(vec_a.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
        }
        // Sets adds copy operation to command encoder.
        // Will copy data from storage buffer on GPU to staging buffer on CPU.
        encoder.copy_buffer_to_buffer(&storage_buffer_c, 0, &staging_buffer, 0, size);

        // Submits command encoder for processing
        queue.submit(Some(encoder.finish()));

        // Note that we're not calling `.await` here.
        let buffer_slice = staging_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        device.poll(wgpu::Maintain::Wait);

        // Awaits until `buffer_future` can be read from
        if let Some(Ok(())) = receiver.receive().await {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result = bytemuck::cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            staging_buffer.unmap(); // Unmaps buffer from memory
                                    // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                    //   delete myPointer;
                                    //   myPointer = NULL;
                                    // It effectively frees the memory

            // Returns data from buffer
            Some(result)
        } else {
            panic!("failed to run dot product compute on gpu!")
        }
    }

    #[test]
    fn elementwise_multiplication() {
        let vec_a = vec![2.0; 128 * 128];
        let vec_b = vec![3.0; 128 * 128];
        let result;

        #[cfg(not(target_arch = "wasm32"))]
        {
            env_logger::init();
            result = pollster::block_on(async { execute_gpu(&vec_a, &vec_b).await.unwrap() });
        }
        #[cfg(target_arch = "wasm32")]
        {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init().expect("could not initialize logger");
            result = wasm_bindgen_futures::spawn_local(async {
                execute_gpu(&vec_a, &vec_b).await.unwrap()
            });
        }
        assert!(result == vec![6.0; 128 * 128]);
    }
}
