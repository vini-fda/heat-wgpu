use super::ExecutionStep;
use crate::dia_matrix::DIAMatrixDescriptor;

/// Specialized sparse matrix-vector multiplication kernel.
///
/// Describes y = A * x.
pub struct SpMV {
    step: ExecutionStep,
}

impl SpMV {
    pub fn new(
        device: &wgpu::Device,
        a: &DIAMatrixDescriptor,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
    ) -> Self {
        let spmv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sparse matrix-vector multiplication shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/spmv.wgsl").into()),
        });

        let spmv_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group layout for sparse matrix-vector multiplication"),
                entries: &[
                    // binding 0: input_vec
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
                    // binding 1: params (DIAMatrixParams)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    //binding 2: data (matrix data)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    //binding 3: offsets (matrix offsets)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                        count: None,
                    },
                    //binding 4: output_vec
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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

        let spmv_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group for sparse matrix-vector multiplication"),
            layout: &spmv_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: a.data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: a.offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: y.as_entire_binding(),
                },
            ],
        });

        let spmv_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sparse matrix-vector multiplication pipeline layout"),
            bind_group_layouts: &[&spmv_bind_group_layout],
            push_constant_ranges: &[],
        });

        let spmv_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sparse matrix-vector multiplication pipeline"),
            layout: Some(&spmv_pipeline_layout),
            module: &spmv_shader,
            entry_point: "main",
        });

        let workgroups = ((a.num_cols + 63) / 64, 1, 1);

        Self {
            step: ExecutionStep::new(spmv_bind_group, spmv_pipeline, workgroups),
        }
    }
}
