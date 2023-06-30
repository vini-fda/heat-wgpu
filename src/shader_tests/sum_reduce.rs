#[cfg(test)]
mod tests {
    use regex::Regex;
    use wgpu::util::DeviceExt;

    async fn execute_gpu(vec_in: &[f32]) -> Option<(Vec<f32>, f32)> {
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

        execute_gpu_inner(&device, &queue, vec_in).await
    }

    async fn execute_gpu_inner(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vec_in: &[f32],
    ) -> Option<(Vec<f32>, f32)> {
        const WORKGROUP_SIZE: u32 = 256;
        let work_size = vec_in.len() as u32;
        let shader_input_1 = include_str!("../shaders/sum_reduce.wgsl");
        let pattern = Regex::new(r"\{WORKGROUP_SIZE\}").unwrap();
        let shader_1 = pattern.replace_all(shader_input_1, WORKGROUP_SIZE.to_string().as_str());

        let shader_input_2 = include_str!("../shaders/sum_reduce_final.wgsl");
        let patterns = vec![(
            Regex::new(r"\{NUM_GROUPS\}").unwrap(),
            work_size / WORKGROUP_SIZE,
        )];
        let shader_2 =
            patterns
                .iter()
                .fold(shader_input_2.to_string(), |acc, (pattern, replacement)| {
                    pattern
                        .replace_all(&acc, replacement.to_string())
                        .to_string()
                });
        // Loads the shader from WGSL
        let cs_module_1 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_1),
        });

        let cs_module_2 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_2.into()),
        });

        // Gets the size in bytes of the buffer.
        let slice_size = vec_in.len() * std::mem::size_of::<u32>();
        let size = slice_size as wgpu::BufferAddress;

        // Instantiates buffer without data.
        // `usage` of buffer specifies how it can be used:
        //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
        //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
        let staging_buffer_1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buffer_2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<f32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Instantiates buffer with data (`vec_a`).
        // Usage allowing the buffer to be:
        //   A storage buffer (can be bound within a bind group and thus available to a shader).
        //   The destination of a copy.
        //   The source of a copy.
        let storage_buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer for Input"),
            contents: bytemuck::cast_slice(vec_in),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Stores the intermediate result of the computation
        let storage_buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Storage Buffer for intermediate output"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Stores the result of the computation
        let storage_buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Storage Buffer for output"),
            size: std::mem::size_of::<f32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // A bind group defines how buffers are accessed by shaders.
        // It is to WebGPU what a descriptor set is to Vulkan.
        // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

        // A pipeline specifies the operation of a shader

        // Instantiates the pipeline.
        let compute_pipeline_1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &cs_module_1,
            entry_point: "main",
        });

        let compute_pipeline_2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &cs_module_2,
            entry_point: "main",
        });

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_layout_1 = compute_pipeline_1.get_bind_group_layout(0);
        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage_buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage_buffer_b.as_entire_binding(),
                },
            ],
        });

        let bind_group_layout_2 = compute_pipeline_2.get_bind_group_layout(0);
        let bind_group_2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout_2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage_buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
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
            cpass.set_pipeline(&compute_pipeline_1);
            cpass.set_bind_group(0, &bind_group_1, &[]);
            cpass.insert_debug_marker("compute vector block sum");
            cpass.dispatch_workgroups(work_size / WORKGROUP_SIZE, 1, 1);
            cpass.set_pipeline(&compute_pipeline_2);
            cpass.set_bind_group(0, &bind_group_2, &[]);
            cpass.insert_debug_marker("compute vector final sum");
            cpass.dispatch_workgroups(work_size, 1, 1);
        }
        // Sets adds copy operation to command encoder.
        // Will copy data from storage buffer on GPU to staging buffer on CPU.
        encoder.copy_buffer_to_buffer(&storage_buffer_b, 0, &staging_buffer_1, 0, size);
        encoder.copy_buffer_to_buffer(
            &storage_buffer_c,
            0,
            &staging_buffer_2,
            0,
            std::mem::size_of::<f32>() as wgpu::BufferAddress,
        );
        // Submits command encoder for processing
        queue.submit(Some(encoder.finish()));

        let result_1 = load_from_buffer(device, &staging_buffer_1).await;
        let result_2: f32;
        let buffer_slice = staging_buffer_2.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        if let Some(Ok(())) = receiver.receive().await {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let r: &[f32] = bytemuck::cast_slice(&data);
            result_2 = r[0];
            drop(data);
        } else {
            panic!("failed to receive buffer");
        }
        Some((result_1, result_2))
    }

    async fn load_from_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
        // Note that we're not calling `.await` here.
        let buffer_slice = buffer.slice(..);
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
            buffer.unmap(); // Unmaps buffer from memory
                            // If you are familiar with C++ these 2 lines can be thought of similarly to:
                            //   delete myPointer;
                            //   myPointer = NULL;
                            // It effectively frees the memory

            // Returns data from buffer
            result
        } else {
            panic!("failed to run dot product compute on gpu!")
        }
    }

    #[test]
    fn sum_reduce() {
        let vec_in = vec![2.0; 128 * 128];
        let (result_1, result_2) =
            pollster::block_on(async { execute_gpu(&vec_in).await.unwrap() });

        println!("result1[0] = {:?}", result_1[0]);
        println!("result1.sum() = {:?}", result_1.iter().sum::<f32>());
        assert!(result_1.iter().sum::<f32>() == 128.0 * 128.0 * 2.0);
        println!("result2 = {:?}", result_2);
        assert!(result_2 == 128.0 * 128.0 * 2.0);
    }
}
