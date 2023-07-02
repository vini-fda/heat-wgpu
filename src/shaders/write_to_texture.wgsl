@group(0) @binding(0) var<storage, read> input_vec: array<f32>;
@group(0) @binding(1) var texture: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let dimensions: vec2<u32> = textureDimensions(texture);
    let W: u32 = dimensions.x;
    let H: u32 = dimensions.y;

    if (x >= W || y >= H) {
        return;
    }

    let i: u32 = y * W + x;

    textureStore(texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(input_vec[i], 0.0, 0.0, 1.0));
}