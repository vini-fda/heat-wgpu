use super::{kernel::Kernel, ExecutionStep};

/// Performs y = x - y
pub struct SAXPYUpdateKernel {
    step: ExecutionStep,
}

impl SAXPYUpdateKernel {
    pub fn new(device: &wgpu::Device, x: &wgpu::Buffer, y: &wgpu::Buffer) -> Self {
        const WORKGROUP_SIZE: u32 = 256;
        let work_size = y.size() as u32 / std::mem::size_of::<f32>() as u32;
        let saxpy_update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SAXPY update shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/saxpy_update.wgsl").into()),
        });

        let saxpy_update_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group layout for SAXPY update"),
                entries: &[
                    // binding 0: y
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    // binding 1: x
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
                ],
            });

        let saxpy_update_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group for SAXPY update"),
            layout: &saxpy_update_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.as_entire_binding(),
                },
            ],
        });

        let saxpy_update_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline layout for SAXPY update"),
                bind_group_layouts: &[&saxpy_update_bind_group_layout],
                push_constant_ranges: &[],
            });

        let saxpy_update_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SAXPY update pipeline"),
                layout: Some(&saxpy_update_pipeline_layout),
                module: &saxpy_update_shader,
                entry_point: "main",
            });

        let workgroups = ((work_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);

        Self {
            step: ExecutionStep::new(saxpy_update_bind_group, saxpy_update_pipeline, workgroups),
        }
    }
}

impl Kernel for SAXPYUpdateKernel {
    fn add_to_pass<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        self.step.add_to_pass(pass);
    }
}
