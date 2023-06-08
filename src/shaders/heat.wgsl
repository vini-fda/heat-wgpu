@group(0) @binding(0) var input_texture: texture_storage_2d<r32float, read>;
@group(0) @binding(1) var output_texture: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16)
fn heat_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(input_texture);
    let coords = vec2<i32>(global_id.xy);

    var sum: vec4<f32> = vec4<f32>(0.0);
    for (var dx: i32 = -1; dx <= 1; dx++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            let kernelCoords = coords + vec2<i32>(dx, dy);
            let kernelValue = textureLoad(input_texture, kernelCoords);
            sum += kernelValue;
        }
    }

    let value = sum / 9.0;

    textureStore(output_texture, coords, value * 0.999);
}