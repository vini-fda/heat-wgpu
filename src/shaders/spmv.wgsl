@group(0) @binding(0) var<storage, read> input_vec: array<f32>;
@group(0) @binding(1) var<uniform> params: DIAMatrixParams;
@group(0) @binding(2) var<storage, read> data: array<f32>;
@group(0) @binding(3) var<storage, read> offsets: array<i32>;
@group(0) @binding(4) var<storage, write> output_vec: array<f32>;

// Diagonal representation of a matrix A
struct DIAMatrixParams {
    num_cols: u32,
    num_rows: u32,
    num_diags: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var row = global_id.x;
    if (row < params.num_rows) {
        var dot: f32 = 0.0;
        for (var n = 0u; n < params.num_diags; n++) {
            let col = i32(row) + offsets[n];
            let val = data[params.num_rows * n + row];
            if (col >= 0 && col < i32(params.num_cols)) {
                dot += val * input_vec[u32(col)];
            }
        }
        output_vec[row] = dot;
    }
}