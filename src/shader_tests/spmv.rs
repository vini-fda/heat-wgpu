#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;
    use regex::Regex;
    use wgpu::util::DeviceExt;

    #[derive(Zeroable, Clone, Copy)]
    #[repr(C)]
    struct DIAMatrixParams {
        num_cols: u32,
        num_rows: u32,
        num_diags: u32,
    }

    async fn execute_gpu(
        x: &[f32],
        params: &DIAMatrixParams,
        data: &[f32],
        offsets: &[i32],
    ) -> Option<Vec<f32>> {
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

        execute_gpu_inner(&device, &queue, x, params, data, offsets).await
    }

    async fn execute_gpu_inner(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        x: &[f32],
        params: &DIAMatrixParams,
        data: &[f32],
        offsets: &[i32],
    ) -> Option<Vec<f32>> {
        let shader_input = include_str!("../shaders/spmv.wgsl");
        let pattern = Regex::new(r"\{NUM_OFFSETS\}").unwrap();
        let shader = pattern.replace_all(shader_input, 3.to_string().as_str());

        // Loads the shader from WGSL
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader),
        });

        // Gets the size in bytes of the buffer.
        let slice_size = x.len() * std::mem::size_of::<u32>();
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
            label: Some("Storage Buffer A - input vector"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let matrix_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer B - matrix params"),
            contents: bytemuck::cast_slice(&[params.num_cols, params.num_rows, params.num_diags]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let storage_buffer_data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer B - matrix data"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let storage_buffer_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer B - matrix offsets"),
            contents: bytemuck::cast_slice(offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Stores the result of the computation
        let storage_buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Storage Buffer C - output vector"),
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
                    resource: matrix_param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: storage_buffer_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: storage_buffer_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
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
            cpass.insert_debug_marker("compute sparse matrix-vector multiplication (SpMV)");
            cpass.dispatch_workgroups(x.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
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
    fn spmv() {
        const M: usize = 16;
        let x = vec![2.0; M];
        let params = DIAMatrixParams {
            num_cols: M as u32,
            num_rows: M as u32,
            num_diags: 3,
        };
        let data = vec![
            0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0,
        ];
        let offsets = vec![-1, 0, 1];
        let result;

        #[cfg(not(target_arch = "wasm32"))]
        {
            env_logger::init();
            result = pollster::block_on(async {
                execute_gpu(&x, &params, &data, &offsets).await.unwrap()
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init().expect("could not initialize logger");
            result = wasm_bindgen_futures::spawn_local(async {
                execute_gpu(&x, &params, &data, &offsets).await.unwrap()
            });
        }
        println!("{:?}", result);
        //assert!(result == vec![6.0; 128 * 128]);
    }
}
