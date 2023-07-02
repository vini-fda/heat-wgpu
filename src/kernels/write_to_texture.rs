use super::{kernel::Kernel, ExecutionStep};

/// Writes a buffer to a 2d storage texture.
pub struct WriteToTextureKernel {
    step: ExecutionStep,
}

impl WriteToTextureKernel {
    pub fn new(device: &wgpu::Device, x: &wgpu::Buffer, t: &wgpu::Texture) -> Self {
        let write_to_texture_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Write-to-texture shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/write_to_texture.wgsl").into(),
            ),
        });

        let write_to_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group layout for write-to-texture"),
                entries: &[
                    // binding 0: x
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
                    // binding 1: t
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let write_to_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group for write-to-texture"),
            layout: &write_to_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &t.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
            ],
        });

        let write_to_texture_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline layout for write-to-texture"),
                bind_group_layouts: &[&write_to_texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let write_to_texture_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute pipeline for write-to-texture"),
                layout: Some(&write_to_texture_pipeline_layout),
                module: &write_to_texture_shader,
                entry_point: "main",
            });
        let texture_size = t.size();
        let workgroups = (texture_size.width / 8, texture_size.height / 8, 1);

        Self {
            step: ExecutionStep::new(
                write_to_texture_bind_group,
                write_to_texture_pipeline,
                workgroups,
            ),
        }
    }
}

impl Kernel for WriteToTextureKernel {
    fn add_to_pass<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        self.step.add_to_pass(pass);
    }
}
