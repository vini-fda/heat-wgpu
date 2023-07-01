pub mod dot;
pub mod kernel;
pub mod saxpy_update;
pub mod spmv;

pub struct ExecutionStep {
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    workgroups: (u32, u32, u32),
}

impl ExecutionStep {
    pub fn new(
        bind_group: wgpu::BindGroup,
        pipeline: wgpu::ComputePipeline,
        workgroups: (u32, u32, u32),
    ) -> Self {
        Self {
            bind_group,
            pipeline,
            workgroups,
        }
    }
}
