pub trait Kernel {
    fn add_to_pass<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>);
}
