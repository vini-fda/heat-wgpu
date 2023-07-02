use wgpu::util::DeviceExt;
/// Represents a sparse matrix in diagonal format.
pub struct DIAMatrixDescriptor {
    pub num_cols: u32,
    pub num_rows: u32,
    pub num_diags: u32,
    pub params: wgpu::Buffer, //num_cols, num_rows, num_diags
    pub data: wgpu::Buffer,
    pub offsets: wgpu::Buffer,
}

impl DIAMatrixDescriptor {
    pub fn new(
        device: &wgpu::Device,
        num_cols: u32,
        num_rows: u32,
        num_diags: u32,
        data: &[f32],
        offsets: &[i32],
    ) -> Self {
        Self {
            num_cols,
            num_rows,
            num_diags,
            params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Matrix Params Buffer"),
                contents: bytemuck::cast_slice(&[num_cols, num_rows, num_diags]),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            data: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Matrix Data Buffer"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            offsets: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Matrix Offsets Buffer"),
                contents: bytemuck::cast_slice(offsets),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        }
    }
}
