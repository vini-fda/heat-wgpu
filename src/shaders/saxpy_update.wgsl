@group(0) @binding(0) var<storage, read_write> input_vec_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_vec_b: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // perform update a = b - a
    input_vec_a[index] = input_vec_b[index] - input_vec_a[index];
}