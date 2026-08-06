// Pass one of the SPLIT variant: walk.wgsl plus the running phase stored once per symbol.
// Benchmark 3's store side, 5 blocks/symbol across the pair against the fused path's 7,
// at 4 B/symbol. theta is block-relative and NOT wrapped (under ~10 rad); sums[b] stays
// bit-identical to walk.wgsl's.

@group(0) @binding(0) var<storage, read> cfg: SymCfg;

@group(0) @binding(1) var<storage, read_write> sums: array<f32>;

@group(0) @binding(2) var<storage, read_write> theta: array<f32>;

fn stage(s: u32, idx: u32) -> vec4<f32> {
    let o = s * 5u;

    return draw4(cfg.keys[o], cfg.keys[o + 1u], cfg.keys[o + 2u], cfg.keys[o + 3u], cfg.keys[o + 4u], idx);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= cfg.nb) {
        return;
    }

    let base = b * cfg.bl;
    var s = 0.0;
    for (var j = 0u; j < cfg.bl; j = j + 1u) {
        let k = base + j;
        let gt = stage(3u, k);
        let gl = stage(4u, k);
        s = s + cfg.sd_tx * gt.x - cfg.sd_lo * gl.x;
        theta[k] = s;
    }

    sums[b] = s;
}
