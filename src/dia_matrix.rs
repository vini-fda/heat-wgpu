/// Represents a sparse matrix in diagonal format.
pub struct DIAMatrixDescriptor {
    pub num_cols: u32,
    pub num_rows: u32,
    pub num_diags: u32,
    pub params: wgpu::Buffer, //num_cols, num_rows, num_diags
    pub data: wgpu::Buffer,
    pub offsets: wgpu::Buffer,
}
