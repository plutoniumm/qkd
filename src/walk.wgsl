// Pass one of the phase random walk: Wiener increments summed within each block. The scan
// itself stays on the host -- a monolithic f32 scan errs 4.0e-2 rad, the size of the
// residual phase noise being measured, against 4.8e-5 rad for the host's f64 scan.

@group(0) @binding(0) var<storage, read> cfg: SymCfg;

@group(0) @binding(1) var<storage, read_write> sums: array<f32>;

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
    }

    sums[b] = s;
}
