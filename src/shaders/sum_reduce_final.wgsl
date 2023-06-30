@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, write> output: f32;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // does the final pass parallel sum reduction of the input array
    // and stores the final result in the output array
    let id = global_id.x;

    if (id == 0u) {
        var sum = 0.0;
        for (var i = 0; i < {NUM_GROUPS}; i += 1) {
            sum += input[i];
        }
        output = sum;
    }
}