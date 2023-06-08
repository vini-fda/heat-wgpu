use std::sync::Arc;

use crate::directional_bind_group::{Direction, DirectionalBindGroup};

pub struct Compute {
    bind_group: DirectionalBindGroup,
    pipeline: wgpu::ComputePipeline,
    texture_size: wgpu::Extent3d,
}

impl Compute {
    pub fn new(
        device: &wgpu::Device,
        direction: Arc<Direction>,
        texture_size: wgpu::Extent3d,
        texture_a: &wgpu::TextureView,
        texture_b: &wgpu::TextureView,
    ) -> Self {
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Decay shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/decay.wgsl").into()),
        });

        let compute_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Texture bind group layout for compute shader"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
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

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Decay pipeline layout"),
                bind_group_layouts: &[&compute_texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Decay pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "decay_main",
        });

        // Create bind group for compute shader
        // this bind group specifies the forward direction A -> B
        // i.e. read from A and write to B
        let compute_texture_bind_group_forward =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Texture bind group for computation (forward direction)"),
                layout: &compute_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(texture_a),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(texture_b),
                    },
                ],
            });

        // Create bind group for compute shader
        // this bind group specifies the backward direction B -> A
        // i.e. read from B and write to A
        let compute_texture_bind_group_backward =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Texture bind group for computation (backward direction)"),
                layout: &compute_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(texture_b),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(texture_a),
                    },
                ],
            });
        Self {
            bind_group: DirectionalBindGroup::new(
                direction,
                compute_texture_bind_group_forward,
                compute_texture_bind_group_backward,
            ),
            pipeline: compute_pipeline,
            texture_size,
        }
    }
    pub fn step(&mut self, device: &mut wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder for GPU compute"),
        });
        let (dispatch_width, dispatch_height) = compute_work_group_count(
            (self.texture_size.width, self.texture_size.height),
            (16, 16),
        );
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Decay pass"),
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, self.bind_group.get(), &[]);
        compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        drop(compute_pass);
        queue.submit(std::iter::once(encoder.finish()));
    }
}

fn compute_work_group_count(
    (width, height): (u32, u32),
    (workgroup_width, workgroup_height): (u32, u32),
) -> (u32, u32) {
    let x = (width + workgroup_width - 1) / workgroup_width;
    let y = (height + workgroup_height - 1) / workgroup_height;

    (x, y)
}
