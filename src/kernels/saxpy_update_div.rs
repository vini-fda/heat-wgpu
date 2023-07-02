use regex::Regex;

use super::{kernel::Kernel, ExecutionStep};

/// Performs y = (a1/a2) * x OP y
/// where OP is either addition or subtraction.
pub struct SAXPYUpdateDivKernel {
    step: ExecutionStep,
}

#[derive(Copy, Clone)]
pub enum Operation {
    Add,
    Sub,
}

impl SAXPYUpdateDivKernel {
    pub fn new(
        device: &wgpu::Device,
        a1: &wgpu::Buffer,
        a2: &wgpu::Buffer,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        op: Operation,
    ) -> Self {
        const WORKGROUP_SIZE: u64 = 256;
        let work_size = y.size() / std::mem::size_of::<f32>() as u64;
        let shader_source = include_str!("../shaders/saxpy_update_div.wgsl");
        let pattern = Regex::new(r"\{OP\}").unwrap();
        let shader_string = pattern.replace_all(
            shader_source,
            match op {
                Operation::Add => "+",
                Operation::Sub => "-",
            },
        );
        let saxpy_update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SAXPY update div shader"),
            source: wgpu::ShaderSource::Wgsl(shader_string),
        });

        let saxpy_update_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group layout for SAXPY update div"),
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
                    // binding 2: a1
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
                    // binding 3: a2
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: a1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: a2.as_entire_binding(),
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

        let workgroups = (
            ((work_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) as u32,
            1,
            1,
        );

        Self {
            step: ExecutionStep::new(saxpy_update_bind_group, saxpy_update_pipeline, workgroups),
        }
    }
}

impl Kernel for SAXPYUpdateDivKernel {
    fn add_to_pass<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        self.step.add_to_pass(pass);
    }
}
