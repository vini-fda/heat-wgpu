use super::{kernel::Kernel, ExecutionStep};
use crate::dia_matrix::DIAMatrixDescriptor;

/// Specialized sparse matrix-vector multiplication kernel.
///
/// Describes y = A * x.
pub struct SpMVKernel {
    step: ExecutionStep,
}

impl SpMVKernel {
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

        let spmv_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sparse matrix-vector multiplication pipeline"),
            layout: None,
            module: &spmv_shader,
            entry_point: "main",
        });

        let spmv_bind_group_layout = spmv_pipeline.get_bind_group_layout(0);

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

        let workgroups = ((a.num_cols + 63) / 64, 1, 1);

        Self {
            step: ExecutionStep::new(spmv_bind_group, spmv_pipeline, workgroups),
        }
    }
}

impl Kernel for SpMVKernel {
    fn add_to_pass<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        self.step.add_to_pass(pass);
    }
}
