// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let value = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let color = vec4<f32>(TurboColormap(value.r), 1.0);
    return color;
}

// Turbo colormap (polynomial approximation)
// reference: https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
// Original LUT: https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
// Authors: Anton Mikhailov (mikhailov@google.com), Ruofei Du (ruofei@google.com)
fn TurboColormap(x: f32) -> vec3<f32> {
  let kRedVec4: vec4<f32> = vec4<f32>(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  let kGreenVec4: vec4<f32> = vec4<f32>(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  let kBlueVec4: vec4<f32> = vec4<f32>(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  let kRedVec2: vec2<f32> = vec2<f32>(-152.94239396, 59.28637943);
  let kGreenVec2: vec2<f32> = vec2<f32>(4.27729857, 2.82956604);
  let kBlueVec2: vec2<f32> = vec2<f32>(-89.90310912, 27.34824973);
  
  let y = clamp(x, 0.0, 1.0);
  var v4: vec4<f32> = vec4<f32>(1.0, y, y * y, y * y * y);
  var v2: vec2<f32> = v4.zw * v4.z;
  return vec3<f32>(
    dot(v4, kRedVec4) + dot(v2, kRedVec2),
    dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
    dot(v4, kBlueVec4) + dot(v2, kBlueVec2)
  );
}