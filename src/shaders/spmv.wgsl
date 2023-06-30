@group(0) @binding(0) var<storage, read> input_vec: array<f32>;
@group(0) @binding(1) var<storage, read> input_matrix: DIAMatrix;
@group(0) @binding(2) var<storage, write> output_vec: array<f32>;

// Diagonal representation of a matrix A
struct DIAMatrix {
    data: array<f32>;
    offsets: array<i32>;
    num_cols: u32;
    num_rows: u32;
    num_diags: u32; 
}

@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var row = global_id.x;
    if (row < input_matrix.num_rows) {
        var dot: f32 = 0.0;
        for (var n = 0u; n < input_matrix.num_diags; n++) {
            let col = row + offsets[n];
            let val = input_matrix.data[input_matrix.num_rows * n + row];
            if (col >= 0 && col < input_matrix.num_cols) {
                dot += val * input_vec[col];
            }
        }
        output_vec[row] = dot;
    }
}