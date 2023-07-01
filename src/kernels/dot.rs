use super::ExecutionStep;

pub struct DotKernel {
    vec_mul: ExecutionStep,
    block_sum_reduce: ExecutionStep,
    sum_reduce: ExecutionStep,
}

impl DotKernel {
    pub fn new(
        device: &wgpu::Device,
        x: &wgpu::Buffer,
        tmp0: &wgpu::Buffer,
        tmp1: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Self {
        // First stage of iteration: tmp = x .* x
        let vec_mul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Element-wise vector multiplication shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vec_mul.wgsl").into()),
        });

        let vec_mul_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group layout for element-wise vector multiplication"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                ],
            });

        let vec_mul_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group for element-wise vector multiplication"),
            layout: &vec_mul_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tmp0.as_entire_binding(),
                },
            ],
        });

        let vec_mul_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Element-wise vector multiplication pipeline layout"),
                bind_group_layouts: &[&vec_mul_bind_group_layout],
                push_constant_ranges: &[],
            });

        let vec_mul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Element-wise vector multiplication pipeline"),
            layout: Some(&vec_mul_pipeline_layout),
            module: &vec_mul_shader,
            entry_point: "main",
        });

        let vec_mul_workgroups = (256, 1, 1);

        // Second stage of iteration: output_vec = block_sum(tmp)
        let block_sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Parallel block sum-reduce shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sum_reduce.wgsl").into()),
        });

        let block_sum_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group layout for parallel block sum-reduce"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                ],
            });

        let block_sum_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group for parallel block sum-reduce"),
            layout: &block_sum_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tmp0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tmp1.as_entire_binding(),
                },
            ],
        });

        let block_sum_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Parallel block sum-reduce pipeline layout"),
                bind_group_layouts: &[&block_sum_bind_group_layout],
                push_constant_ranges: &[],
            });

        let block_sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Parallel block sum-reduce pipeline"),
            layout: Some(&block_sum_pipeline_layout),
            module: &block_sum_shader,
            entry_point: "main",
        });

        let block_sum_workgroups = (256, 1, 1);

        // Third stage of iteration: output = sum(output_vec) (final output as scalar)
        let sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Parallel sum-reduce shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sum_reduce_final.wgsl").into(),
            ),
        });

        let sum_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group layout for parallel sum-reduce"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                ],
            });

        let sum_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group for parallel sum-reduce"),
            layout: &sum_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tmp1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        let sum_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Parallel sum-reduce pipeline layout"),
            bind_group_layouts: &[&sum_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Parallel sum-reduce pipeline"),
            layout: Some(&sum_pipeline_layout),
            module: &sum_shader,
            entry_point: "main",
        });

        let sum_workgroups = (1, 1, 1);

        Self {
            vec_mul: ExecutionStep::new(vec_mul_bind_group, vec_mul_pipeline, vec_mul_workgroups),
            block_sum_reduce: ExecutionStep::new(
                block_sum_bind_group,
                block_sum_pipeline,
                block_sum_workgroups,
            ),
            sum_reduce: ExecutionStep::new(sum_bind_group, sum_pipeline, sum_workgroups),
        }
    }
}
