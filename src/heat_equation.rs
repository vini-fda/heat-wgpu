use std::rc::Rc;

use wgpu::util::DeviceExt;

use crate::{
    conjugate_gradient::{CGBuffers, CG},
    dia_matrix::DIAMatrixDescriptor,
    kernels::{kernel::Kernel, spmv::SpMVKernel, write_to_texture::WriteToTextureKernel},
};

pub struct HeatEquation {
    cg_forward: CG,                    // CG method for forward mode (tmp -> u_)
    cg_backward: CG,                   // CG method for backward mode (tmp -> u)
    initial_spmv_forward: SpMVKernel,  // Initial SpMV kernel for forward mode
    initial_spmv_backward: SpMVKernel, // Initial SpMV kernel for backward mode
    write_to_texture_forward: WriteToTextureKernel, // Write to texture kernel for forward mode
    write_to_texture_backward: WriteToTextureKernel, // Write to texture kernel for backward mode
    iteration: usize,                  // current iteration
}

impl HeatEquation {
    pub fn new(
        device: &wgpu::Device,
        alpha: f32,
        n: usize,
        dt: f32,
        u0: &[f32],
        texture: &wgpu::Texture,
    ) -> Self {
        let a = Self::a_matrix(device, alpha, n, dt);
        let b = Self::b_matrix(device, alpha, n, dt);
        let u = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("U Vector"),
            contents: bytemuck::cast_slice(u0),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let size_in_bytes = (n * n * std::mem::size_of::<f32>()) as u64;
        let u_ = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("U_ Vector"),
            size: size_in_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tmp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tmp Vector"),
            size: size_in_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let cg_buffers = Rc::new(CGBuffers::new(device, size_in_bytes));
        let cg_forward = CG::new(device, cg_buffers.clone(), &a, &tmp, &u_);
        let cg_backward = CG::new(device, cg_buffers, &a, &tmp, &u);
        let initial_spmv_forward = SpMVKernel::new(device, &b, &u, &tmp);
        let initial_spmv_backward = SpMVKernel::new(device, &b, &u_, &tmp);
        let write_to_texture_forward = WriteToTextureKernel::new(device, &u_, texture);
        let write_to_texture_backward = WriteToTextureKernel::new(device, &u, texture);

        Self {
            cg_forward,
            cg_backward,
            initial_spmv_forward,
            initial_spmv_backward,
            write_to_texture_forward,
            write_to_texture_backward,
            iteration: 0,
        }
    }

    fn a_matrix(device: &wgpu::Device, alpha: f32, n: usize, dt: f32) -> DIAMatrixDescriptor {
        let m = n * n;
        let num_cols = m;
        let num_rows = m;
        let num_diags = 5;
        let h = 1.0 / (n as f32);
        let gamma: f32 = alpha * dt / (2.0 * h * h);
        // describes discretization matrix A
        // obtained from discretizing U(x,y) in row-major order
        let f = |i: usize, j: usize| -> f32 {
            if i == j {
                1.0 + 4.0 * gamma
            } else if (i == j + 1 && i % n != 0)
                || (j == i + 1 && j % n != 0)
                || (i == j + n || j == i + n)
            {
                -gamma
            } else {
                0.0
            }
        };
        let n = n as i32;
        let offsets = vec![-n, -1, 0, 1, n];
        let mut data: Vec<f32> = Vec::with_capacity(num_diags * m);
        for offset in offsets.iter() {
            for i in 0..m {
                let j = i as i32 + offset;
                if j >= 0 && j < m as i32 {
                    data.push(f(i, j as usize));
                } else {
                    data.push(0.0);
                }
            }
        }
        DIAMatrixDescriptor::new(
            device,
            num_cols as u32,
            num_rows as u32,
            num_diags as u32,
            &data,
            &offsets,
        )
    }

    /// Same as `a_matrix`, but gamma has a negative sign
    fn b_matrix(device: &wgpu::Device, alpha: f32, n: usize, dt: f32) -> DIAMatrixDescriptor {
        let m = n * n;
        let num_cols = m;
        let num_rows = m;
        let num_diags = 5;
        let h = 1.0 / (n as f32);
        let gamma: f32 = -alpha * dt / (2.0 * h * h);
        // describes discretization matrix A
        // obtained from discretizing U(x,y) in row-major order
        let f = |i: usize, j: usize| -> f32 {
            if i == j {
                1.0 + 4.0 * gamma
            } else if (i == j + 1 && i % n != 0)
                || (j == i + 1 && j % n != 0)
                || (i == j + n || j == i + n)
            {
                -gamma
            } else {
                0.0
            }
        };
        let n = n as i32;
        let offsets = vec![-n, -1, 0, 1, n];
        let mut data: Vec<f32> = Vec::with_capacity(num_diags * m);
        for offset in offsets.iter() {
            for i in 0..m {
                let j = i as i32 + offset;
                if j >= 0 && j < m as i32 {
                    data.push(f(i, j as usize));
                } else {
                    data.push(0.0);
                }
            }
        }
        DIAMatrixDescriptor::new(
            device,
            num_cols as u32,
            num_rows as u32,
            num_diags as u32,
            &data,
            &offsets,
        )
    }

    pub fn compute_step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // First step: tmp = B * u_old
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Initial SpMV Encoder (tmp = B*U)"),
        });
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Initial SpMV Compute Pass (tmp = B*U)"),
            timestamp_writes: None,
        });
        if self.iteration % 2 == 0 {
            self.initial_spmv_forward.add_to_pass(&mut compute_pass);
        } else {
            self.initial_spmv_backward.add_to_pass(&mut compute_pass);
        };

        drop(compute_pass);
        queue.submit(Some(encoder.finish()));
        // Now we can treat the vector tmp as the "b" in A u_new = b
        // for our Conjugate Gradient solver
        if self.iteration % 2 == 0 {
            self.cg_forward.run(device, queue);
        } else {
            self.cg_backward.run(device, queue);
        }

        // now we need to write u_new to the storage texture
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Write to Texture Encoder"),
        });
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Write to Texture Compute Pass"),
            timestamp_writes: None,
        });
        if self.iteration % 2 == 0 {
            self.write_to_texture_forward.add_to_pass(&mut compute_pass);
        } else {
            self.write_to_texture_backward
                .add_to_pass(&mut compute_pass);
        };

        drop(compute_pass);
        queue.submit(Some(encoder.finish()));
        self.iteration += 1;
    }
}
