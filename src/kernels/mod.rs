pub mod dot;
pub mod kernel;
pub mod saxpy_update;
pub mod saxpy_update_div;
pub mod spmv;
pub mod write_to_texture;

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

    pub fn add_to_pass<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.workgroups.0, self.workgroups.1, self.workgroups.2);
    }
}
