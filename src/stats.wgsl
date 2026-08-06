// Pass two of the SPLIT variant: symbols.wgsl with the phase read from memory as
// `wrapf(carry[b] + theta[k])` rather than replayed from stages 3 and 4. Benchmark 3's
// other arm, NOT the production path. Agrees with the fused kernel to f32 rounding only,
// not bit-exactly: wrapping rounds per block.

const WG: u32 = 256u;

@group(0) @binding(0) var<storage, read> cfg: SymCfg;

@group(0) @binding(1) var<storage, read> carry: array<f32>;

@group(0) @binding(2) var<storage, read_write> parts: array<f32>;

@group(0) @binding(3) var<storage, read> theta: array<f32>;

var<workgroup> red: array<f32, 256>;

fn stage(s: u32, idx: u32) -> vec4<f32> {
    let o = s * 5u;

    return draw4(cfg.keys[o], cfg.keys[o + 1u], cfg.keys[o + 2u], cfg.keys[o + 3u], cfg.keys[o + 4u], idx);
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let b = gid.x;
    var acc = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);

    // Out-of-range threads contribute zeros but must NOT return: every barrier
    // below has to be reached by the whole workgroup.
    if (b < cfg.nb) {
        let bl = cfg.bl;
        let base = b * bl;
        let head = carry[b];
        var s_aa = 0.0;
        var s_ai = 0.0;
        var s_yy = 0.0;
        var p_re = 0.0;
        var p_im = 0.0;
        var n_re = 0.0;
        var n_im = 0.0;

        for (var j = 0u; j < bl; j = j + 1u) {
            let k = base + j;

            // The whole difference from symbols.wgsl: one load where the fused kernel
            // re-derives two Threefry blocks.
            let run = wrapf(head + theta[k]);
            let ts = sin(run);
            let tc = cos(run);

            let ga = stage(0u, k);
            let gc = stage(1u, k);
            let gd = stage(2u, k);
            let xa = cfg.ampl * ga.x;
            let pa = cfg.ampl * ga.y;

            let sx = cfg.root_t * (xa * tc - pa * ts) + cfg.ch_sd * gc.x;
            let sp = cfg.root_t * (xa * ts + pa * tc) + cfg.ch_sd * gc.y;
            let yx = cfg.gain * sx + cfg.det_sd * gd.x;
            let yp = cfg.gain * sp + cfg.det_sd * gd.y;

            let ix = yx * tc + yp * ts;
            let ip = -yx * ts + yp * tc;
            s_aa = s_aa + xa * xa + pa * pa;
            s_ai = s_ai + xa * ix + pa * ip;
            s_yy = s_yy + yx * yx + yp * yp;

            let cyc = f32(ticks(cfg.ramp_hi, cfg.ramp_lo, k)) * SCALE;
            let car = TAU * cyc;
            let cs = sin(car);
            let cc = cos(car);
            let px = cfg.pilot_amp * cos(car + run) + cfg.pilot_sd * gd.z;
            let pp = cfg.pilot_amp * sin(car + run) + cfg.pilot_sd * gd.w;
            let dx = px * cc + pp * cs;
            let dp = -px * cs + pp * cc;
            p_re = p_re + dx;
            p_im = p_im + dp;

            let ang = TAU * f32(j) / f32(bl);
            let ns = sin(ang);
            let nc = cos(ang);
            n_re = n_re + dx * nc + dp * ns;
            n_im = n_im - dx * ns + dp * nc;
        }

        acc[0] = s_aa;
        acc[1] = s_ai;
        acc[2] = s_yy;
        acc[3] = p_re * p_re + p_im * p_im;
        acc[4] = n_re * n_re + n_im * n_im;
    }

    // The fused kernel's tree, unchanged, so the comparison isolates per-symbol work.
    let t = lid.x;
    for (var a = 0u; a < 5u; a = a + 1u) {
        red[t] = acc[a];
        workgroupBarrier();
        for (var s = WG / 2u; s > 0u; s = s >> 1u) {
            if (t < s) {
                red[t] = red[t] + red[t + s];
            }

            workgroupBarrier();
        }

        if (t == 0u) {
            parts[wid.x * 5u + a] = red[0];
        }

        workgroupBarrier();
    }
}
