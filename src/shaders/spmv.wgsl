@group(0) @binding(0) var input_vec: texture_storage_2d<r32float, read>;
@group(0) @binding(1) var input_matrix: DIAMatrix;
@group(0) @binding(2) var output_vec: texture_storage_2d<r32float, write>;

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
    // the dimensions of input_vec must match the dimensions of output_vec
    let dimensions: vec2<u32> = textureDimensions(input_vec);
    let coords = vec2<i32>(global_id.xy);
    
    u32 row = coords.x + coords.y * dimensions.x;
    if (row < num_rows){
        f32 dot = 0;
        for (u32 n = 0; n < num_diags; n++) {
            u32 col = row + offsets[n];
            f32 val = data [num_rows * n + row];
            if (col >= 0 && col < num_cols) {
                let p = vec2<i32>(col % dimensions.x, col / dimensions.x);
                dot += val * textureLoad(input_vec, p);
            }
        }
        textureStore(output_vec, coords, dot);
    }


    textureStore(output_vec, coords, v4 + k * sum);
}