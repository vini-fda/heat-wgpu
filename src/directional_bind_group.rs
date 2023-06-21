#[derive(Clone, Copy)]
pub enum Direction {
    Forward,
    Backward,
}

pub struct DirectionalBindGroup {
    bind_group_forward: wgpu::BindGroup,
    bind_group_backward: wgpu::BindGroup,
}

impl DirectionalBindGroup {
    pub fn new(bind_group_forward: wgpu::BindGroup, bind_group_backward: wgpu::BindGroup) -> Self {
        Self {
            bind_group_forward,
            bind_group_backward,
        }
    }
    pub fn get(&self, direction: Direction) -> &wgpu::BindGroup {
        match direction {
            Direction::Forward => &self.bind_group_forward,
            Direction::Backward => &self.bind_group_backward,
        }
    }
}
