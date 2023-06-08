use std::sync::Arc;

pub enum Direction {
    Forward,
    Backward,
}

pub struct DirectionalBindGroup {
    direction: Arc<Direction>,
    bind_group_forward: wgpu::BindGroup,
    bind_group_backward: wgpu::BindGroup,
}

impl DirectionalBindGroup {
    pub fn new(
        direction: Arc<Direction>,
        bind_group_forward: wgpu::BindGroup,
        bind_group_backward: wgpu::BindGroup,
    ) -> Self {
        Self {
            direction,
            bind_group_forward,
            bind_group_backward,
        }
    }
    pub fn get(&self) -> &wgpu::BindGroup {
        match self.direction.as_ref() {
            Direction::Forward => &self.bind_group_forward,
            Direction::Backward => &self.bind_group_backward,
        }
    }
}
