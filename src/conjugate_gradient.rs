use std::rc::Rc;

use crate::{
    dia_matrix::DIAMatrixDescriptor,
    kernels::{
        dot::DotKernel,
        kernel::Kernel,
        saxpy_update::SAXPYUpdateKernel,
        saxpy_update_div::{Operation, SAXPYUpdateDivKernel},
        spmv::SpMVKernel,
    },
};

/// Specialized data structure for the conjugate gradient method
/// specific for GPU compute.
///
/// This uses the CG method to solve the system of linear equations Ax = b where A is a sparse matrix.
pub struct CG {
    buffers: Rc<CGBuffers>,
    init_stages: Vec<Box<dyn Kernel>>,
    stages: Vec<Box<dyn Kernel>>,
    max_steps: usize,
}

impl CG {
    pub fn new(
        device: &wgpu::Device,
        buffers: Rc<CGBuffers>,
        a: &DIAMatrixDescriptor, // Sparse matrix A
        b: &wgpu::Buffer,        // Vector b
        x: &wgpu::Buffer,        // Vector x initialized with initial guess x_0
    ) -> Self {
        Self {
            buffers: buffers.clone(),
            init_stages: Self::init_stages(device, buffers.as_ref(), a, b, x),
            stages: Self::stages(device, buffers.as_ref(), a, x),
            max_steps: 10,
        }
    }

    fn init_stages(
        device: &wgpu::Device,
        buffers: &CGBuffers,
        a: &DIAMatrixDescriptor,
        b: &wgpu::Buffer,
        x: &wgpu::Buffer,
    ) -> Vec<Box<dyn Kernel>> {
        let CGBuffers { r, .. } = buffers;
        // Initialize r = b - A * x
        let r_init0 = SpMVKernel::new(device, a, x, r);
        let r_init1 = SAXPYUpdateKernel::new(device, b, r);
        vec![Box::new(r_init0), Box::new(r_init1)]
    }

    /// Define the stages for a single iteration of the CG algorithm on a `wgpu::ComputePass`
    fn stages(
        device: &wgpu::Device,
        buffers: &CGBuffers,
        a: &DIAMatrixDescriptor,
        x: &wgpu::Buffer,
    ) -> Vec<Box<dyn Kernel>> {
        let CGBuffers {
            r,
            p,
            q,
            sigma,
            sigma_prime,
            tmp0,
            tmp1,
        } = buffers;
        // Iteration stages
        // First stage of iteration: sigma = dot(r, r)
        let sigma_stage = DotKernel::new(device, r, r, tmp0, tmp1, sigma);

        // Second stage of iteration: q = A * p (Sparse matrix-vector multiplication)
        let q_stage = SpMVKernel::new(device, a, p, q);

        // Third stage of iteration: sigma_prime = dot(p, q)
        let sigma_prime_stage = DotKernel::new(device, p, q, tmp0, tmp1, sigma_prime);

        // Fourth stage of iteration: x = x + (sigma / sigma_prime) * p
        let x_stage = SAXPYUpdateDivKernel::new(device, sigma, sigma_prime, p, x, Operation::Add);

        // Fifth stage of iteration: r = r - (sigma / sigma_prime) * q
        let r_stage = SAXPYUpdateDivKernel::new(device, sigma, sigma_prime, q, r, Operation::Sub);

        // Sixth stage of iteration: sigma_prime = dot(r, r)
        let sigma_prime_stage2 = DotKernel::new(device, r, r, tmp0, tmp1, sigma_prime);

        // Seventh stage of iteration: p = r + (sigma_prime / sigma) * p
        let p_stage = SAXPYUpdateDivKernel::new(device, sigma_prime, sigma, r, p, Operation::Add);

        // create Vec<Box<dyn Kernel>> to iterate over
        vec![
            Box::new(sigma_stage),
            Box::new(q_stage),
            Box::new(sigma_prime_stage),
            Box::new(x_stage),
            Box::new(r_stage),
            Box::new(sigma_prime_stage2),
            Box::new(p_stage),
        ]
    }

    pub fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let CGBuffers { r, p, .. } = self.buffers.as_ref();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CG - initialization"),
        });
        // Initialize r = b - A * x
        let cdescriptor = wgpu::ComputePassDescriptor { label: None };
        let mut compute_pass = encoder.begin_compute_pass(&cdescriptor);
        for stage in self.init_stages.iter() {
            stage.add_to_pass(&mut compute_pass);
        }
        drop(compute_pass);
        encoder.copy_buffer_to_buffer(r, 0, p, 0, r.size());
        queue.submit(Some(encoder.finish()));
        // describes all the stages in a single iteration of the CG algorithm
        for _ in 0..self.max_steps {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CG - iteration step"),
            });
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            for s in self.stages.iter() {
                s.add_to_pass(&mut compute_pass);
            }
            drop(compute_pass);
            queue.submit(Some(encoder.finish()));
        }
    }
}

/// Describes all the intermediate buffers used in the CG algorithm.
///
/// The dimensions of the vectors are the same as the dimensions of the vectors x and b in Ax=b.
#[derive(Debug)]
pub struct CGBuffers {
    r: wgpu::Buffer,           // residual vector
    p: wgpu::Buffer,           // direction vector
    q: wgpu::Buffer,           // A * p
    sigma: wgpu::Buffer,       // scalar
    sigma_prime: wgpu::Buffer, // scalar
    tmp0: wgpu::Buffer,        // scratch vector
    tmp1: wgpu::Buffer,        // scratch vector
}

impl CGBuffers {
    pub fn new(
        device: &wgpu::Device,
        size: wgpu::BufferAddress, // size of the vectors, in bytes
    ) -> Self {
        // before iterating, must set up the r(residual) and p(direction) vectors as GPU buffers
        let r = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("r"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let p = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("p"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Intermediate Buffers for iteration
        let q = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let f32_size = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let sigma = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sigma"),
            size: f32_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sigma_prime = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sigma_prime"),
            size: f32_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tmp0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tmp0"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tmp1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tmp1"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            r,
            p,
            q,
            sigma,
            sigma_prime,
            tmp0,
            tmp1,
        }
    }
}
