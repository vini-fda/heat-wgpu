use crate::{
    conjugate_gradient::CG,
    dia_matrix::DIAMatrixDescriptor,
    kernels::{kernel::Kernel, spmv::SpMVKernel, write_to_texture::WriteToTextureKernel},
};

pub struct HeatEquation {
    alpha: f32,             // thermal diffusivity
    n: usize,               // number of grid points in x and y directions
    dt: f32,                // time step
    a: DIAMatrixDescriptor, // A matrix
    b: DIAMatrixDescriptor, // B matrix
    u: wgpu::Buffer,        // U vector
    u_: wgpu::Buffer,       // U_ vector (we use a double buffer to avoid copying)
    tmp: wgpu::Buffer,      // Temporary buffer for storing initial SpMV result
    texture: wgpu::Texture, // Texture for rendering
    iteration: usize,       // current iteration
}

impl HeatEquation {
    pub fn new(device: &wgpu::Device, alpha: f32, n: usize, dt: f32) -> Self {
        let a = Self::a_matrix(device, alpha, n, dt);
        let b = Self::b_matrix(device, alpha, n, dt);
        let u = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("U Vector"),
            size: (n * n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let u_ = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("U_ Vector"),
            size: (n * n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tmp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tmp Vector"),
            size: (n * n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let texture_size = wgpu::Extent3d {
            width: n as u32,
            height: n as u32,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Heat Equation Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        Self {
            alpha,
            n,
            dt,
            a,
            b,
            u,
            u_,
            tmp,
            texture,
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

    fn compute_step_internal(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        u_new: &wgpu::Buffer,
        u_old: &wgpu::Buffer,
    ) {
        let Self {
            alpha: _,
            n,
            dt: _,
            a,
            b,
            u: _,
            u_: _,
            tmp,
            texture,
            ..
        } = self;
        let size_bytes = (n * n * std::mem::size_of::<f32>()) as u64;
        // First step: tmp = B * u_old
        let initial_spmv = SpMVKernel::new(device, b, u_old, tmp);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Initial SpMV Encoder (tmp = B*U)"),
        });
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Initial SpMV Compute Pass (tmp = B*U)"),
        });
        initial_spmv.add_to_pass(&mut compute_pass);
        drop(compute_pass);
        queue.submit(Some(encoder.finish()));
        // Now we can treat the vector tmp as the "b" in A u_new = b
        // for our Conjugate Gradient solver
        let cg = CG::new(device, size_bytes);
        cg.run(device, queue, a, tmp, u_new);
        // now we need to write u_new to the storage texture
        let write2texture = WriteToTextureKernel::new(device, u_new, texture);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Write to Texture Encoder"),
        });
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Write to Texture Compute Pass"),
        });
        write2texture.add_to_pass(&mut compute_pass);
        drop(compute_pass);
        queue.submit(Some(encoder.finish()));
    }

    fn compute_step(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.iteration % 2 == 0 {
            self.compute_step_internal(device, queue, &self.u_, &self.u)
        } else {
            self.compute_step_internal(device, queue, &self.u, &self.u_)
        }
    }

    fn render_step(&self, device: &wgpu::Device, queue: &wgpu::Queue, surface: &wgpu::Surface) {
        todo!()
    }
}
