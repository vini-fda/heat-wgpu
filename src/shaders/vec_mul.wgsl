@group(0) @binding(0) var<storage, read> input_vec_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_vec_b: array<f32>;
@group(0) @binding(2) var<storage, write> output: array<f32>;

@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // the dimensions of input_vec_a must match the dimensions of output_vec as well as input_vec_b
    let index = global_id.x;

    // output is the element-wise product of input_vec_a and input_vec_b
    output[index] = input_vec_a[index] * input_vec_b[index];
}