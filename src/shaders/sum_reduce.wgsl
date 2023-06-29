@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, write> output: array<f32>;

var<workgroup> sdata: array<f32, 16>;

@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) group_id: vec3<u32>) {
    // does a parallel sum reduction of the input array
    // and stores the block results in the output array

    let tid = local_id.x;
    let i = global_id.x + group_id.x * 16u;

    sdata[tid] = input[i] + input[i + 16u];

    workgroupBarrier();

    for (var s = 8u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        workgroupBarrier();
    }

    if (tid == 0u) {
        output[group_id.x] = sdata[0];
    }
}