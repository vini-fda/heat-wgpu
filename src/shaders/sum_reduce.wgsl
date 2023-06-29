@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, write> output: array<f32>;

var<workgroup> sdata: array<f32, {WORKGROUP_SIZE}>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) group_id: vec3<u32>) {
    // does a parallel sum reduction of the input array
    // and stores the block results in the output array

    let tid = local_id.x;
    let i = global_id.x + group_id.x * {WORKGROUP_SIZE}u;

    sdata[tid] = input[i] + input[i + {WORKGROUP_SIZE}u];

    workgroupBarrier();

    for (var s = {WORKGROUP_SIZE}u / 2u; s > 32u; s >>= 1u) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        workgroupBarrier();
    }

    if (tid < 32u) {
        sdata[tid] += sdata[tid + 32u];
        sdata[tid] += sdata[tid + 16u];
        sdata[tid] += sdata[tid + 8u];
        sdata[tid] += sdata[tid + 4u];
        sdata[tid] += sdata[tid + 2u];
        sdata[tid] += sdata[tid + 1u];
    }

    if (tid == 0u) {
        output[group_id.x] = sdata[0];
    }
}