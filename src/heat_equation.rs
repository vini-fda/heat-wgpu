use crate::dia_matrix::DIAMatrixDescriptor;

pub struct HeatEquation {
    alpha: f32,             // thermal diffusivity
    n: usize,               // number of grid points in x and y directions
    dt: f32,                // time step
    a: DIAMatrixDescriptor, // A matrix
    b: DIAMatrixDescriptor, // B matrix
    u: wgpu::Buffer,        // U vector
    u_: wgpu::Buffer,       // U_ vector (we use a double buffer to avoid copying)
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
            label: Some("U Vector"),
            size: (n * n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            alpha,
            n,
            dt,
            a,
            b,
            u,
            u_,
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

    fn compute_step(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        todo!()
    }
}
