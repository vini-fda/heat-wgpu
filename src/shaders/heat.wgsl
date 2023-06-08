@group(0) @binding(0) var input_texture: texture_storage_2d<r32float, read>;
@group(0) @binding(1) var output_texture: texture_storage_2d<r32float, write>;

const stencil = array<f32, 9>(
    1.0, 4.0, 1.0,
    4.0, -20.0, 4.0,
    1.0, 4.0, 1.0
);

const k: f32 = 0.05;

// solves heat equation with 3x3 stencil and a forward Euler approach
// TODO: use a more stable method
@compute @workgroup_size(16, 16)
fn heat_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions: vec2<u32> = textureDimensions(input_texture);
    let coords = vec2<i32>(global_id.xy);

    var sum: vec4<f32> = vec4<f32>(0.0);

    // unrolled loop
    let k0 = coords + vec2<i32>(-1, -1);
    let v0 = textureLoad(input_texture, k0);
    sum += v0 * stencil[0];
    let k1 = coords + vec2<i32>(0, -1);
    let v1 = textureLoad(input_texture, k1);
    sum += v1 * stencil[1];
    let k2 = coords + vec2<i32>(1, -1);
    let v2 = textureLoad(input_texture, k2);
    sum += v2 * stencil[2];
    let k3 = coords + vec2<i32>(-1, 0);
    let v3 = textureLoad(input_texture, k3);
    sum += v3 * stencil[3];
    let k4 = coords;
    let v4 = textureLoad(input_texture, k4);
    sum += v4 * stencil[4];
    let k5 = coords + vec2<i32>(1, 0);
    let v5 = textureLoad(input_texture, k5);
    sum += v5 * stencil[5];
    let k6 = coords + vec2<i32>(-1, 1);
    let v6 = textureLoad(input_texture, k6);
    sum += v6 * stencil[6];
    let k7 = coords + vec2<i32>(0, 1);
    let v7 = textureLoad(input_texture, k7);
    sum += v7 * stencil[7];
    let k8 = coords + vec2<i32>(1, 1);
    let v8 = textureLoad(input_texture, k8);
    sum += v8 * stencil[8];

    textureStore(output_texture, coords, v4 + k * sum);
}