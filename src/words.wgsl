// Raw Threefry blocks, one per counter index, for comparison only: test/gpu.py runs this
// and src/threefry.rs over the same (seed, stage, index, draw) and requires all 128 output
// bits to match. Nothing downstream reads the buffer.

struct WordCfg {
    keys: array<u32, 5>,
    base_lo: u32,
    base_hi: u32,
    draw: u32,
    n: u32,
}

@group(0) @binding(0) var<storage, read> cfg: WordCfg;

@group(0) @binding(1) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) {
        return;
    }

    // 64-bit counter add, so a base past 2^32 exercises the high word.
    let lo = cfg.base_lo + i;
    var hi = cfg.base_hi;
    if (lo < cfg.base_lo) {
        hi = hi + 1u;
    }

    let x = tf_block(cfg.keys[0], cfg.keys[1], cfg.keys[2], cfg.keys[3], cfg.keys[4], lo, hi, cfg.draw);
    let at = i * 4u;
    out[at] = x.x;
    out[at + 1u] = x.y;
    out[at + 2u] = x.z;
    out[at + 3u] = x.w;
}
