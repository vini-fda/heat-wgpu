use regex::Regex;

use super::{kernel::Kernel, ExecutionStep};

pub struct DotKernel {
    vec_mul: ExecutionStep,
    block_sum_reduce: ExecutionStep,
    sum_reduce: ExecutionStep,
}

impl DotKernel {
    pub fn new(
        device: &wgpu::Device,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        tmp0: &wgpu::Buffer,
        tmp1: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Self {
        // First stage of iteration: tmp = x .* y
        let vec_mul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Element-wise vector multiplication shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vec_mul.wgsl").into()),
        });

        let vec_mul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Element-wise vector multiplication pipeline"),
            layout: None,
            module: &vec_mul_shader,
            entry_point: "main",
        });

        let vec_mul_bind_group_layout = vec_mul_pipeline.get_bind_group_layout(0);

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
                    resource: y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tmp0.as_entire_binding(),
                },
            ],
        });

        let vec_mul_workgroups = (256, 1, 1);

        // Second stage of iteration: output_vec = block_sum(tmp)
        let shader_string = include_str!("../shaders/sum_reduce.wgsl");
        const WORKGROUP_SIZE: u32 = 256;
        let pattern = Regex::new(r"\{WORKGROUP_SIZE\}").unwrap();
        let shader_string = pattern.replace_all(shader_string, WORKGROUP_SIZE.to_string().as_str());
        let block_sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Parallel block sum-reduce shader"),
            source: wgpu::ShaderSource::Wgsl(shader_string),
        });

        let block_sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Parallel block sum-reduce pipeline"),
            layout: None,
            module: &block_sum_shader,
            entry_point: "main",
        });

        let block_sum_bind_group_layout = block_sum_pipeline.get_bind_group_layout(0);

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

        let block_sum_workgroups = (WORKGROUP_SIZE, 1, 1);

        // Third stage of iteration: output = sum(output_vec) (final output as scalar)
        let sum_shader_string = include_str!("../shaders/sum_reduce_final.wgsl");
        let pattern = Regex::new(r"\{NUM_GROUPS\}").unwrap();
        let work_size = (x.size() / (std::mem::size_of::<f32>() as u64)) as u32;
        let num_groups = (work_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let sum_shader_string =
            pattern.replace_all(sum_shader_string, num_groups.to_string().as_str());
        let sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Parallel sum-reduce shader"),
            source: wgpu::ShaderSource::Wgsl(sum_shader_string),
        });

        let sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Parallel sum-reduce pipeline"),
            layout: None,
            module: &sum_shader,
            entry_point: "main",
        });

        let sum_bind_group_layout = sum_pipeline.get_bind_group_layout(0);

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

impl Kernel for DotKernel {
    fn add_to_pass<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        self.vec_mul.add_to_pass(pass);
        self.block_sum_reduce.add_to_pass(pass);
        self.sum_reduce.add_to_pass(pass);
    }
}
