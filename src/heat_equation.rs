use crate::dia_matrix::DIAMatrixDescriptor;

pub struct HeatEquation {
    alpha: f32, // thermal diffusivity
    n: usize,   // number of grid points in x and y directions
    dt: f32,    // time step
}

impl HeatEquation {
    pub fn new(alpha: f32, n: usize, dt: f32) -> Self {
        Self { alpha, n, dt }
    }

    pub fn a_matrix(&self, device: &wgpu::Device) -> DIAMatrixDescriptor {
        let &Self { alpha, n, dt } = self;

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
    pub fn b_matrix(&self, device: &wgpu::Device) -> DIAMatrixDescriptor {
        let &Self { alpha, n, dt } = self;

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
        DIAMatrixDescriptor::new(device, num_cols as u32, num_rows as u32, num_diags as u32, &data, &offsets)
    }
}
