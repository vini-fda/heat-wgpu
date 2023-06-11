use std::sync::{Arc, RwLock};

#[derive(Clone, Copy)]
pub enum Direction {
    Forward,
    Backward,
}

pub struct DirectionalBindGroup {
    direction: Arc<RwLock<Direction>>,
    bind_group_forward: wgpu::BindGroup,
    bind_group_backward: wgpu::BindGroup,
}

impl DirectionalBindGroup {
    pub fn new(
        direction: Arc<RwLock<Direction>>,
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
        match *self.direction.read().unwrap() {
            Direction::Forward => &self.bind_group_forward,
            Direction::Backward => &self.bind_group_backward,
        }
    }
}
