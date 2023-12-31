@group(0) @binding(0) var input_vec_a: texture_storage_2d<r32float, read>;
@group(0) @binding(1) var input_vec_b: texture_storage_2d<r32float, read>;
@group(0) @binding(2) var<storage, read_write> output: f32;

@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // the dimensions of input_vec_a must match the dimensions of output_vec as well as input_vec_b
    let dimensions: vec2<u32> = textureDimensions(input_vec);
    let coords = vec2<i32>(global_id.xy);
    
    let a = textureLoad(input_vec_a, coords);
    let b = textureLoad(input_vec_b, coords);
    let prev = textureLoad(output_vec, coords);
        
    textureStore(output_vec, coords, prev + a * b);
}