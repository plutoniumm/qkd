// Threefry-4x32-20 and Box-Muller, the twin of src/threefry.rs: the draw at i
// is a pure function of (seed, stage, i), bit-identical across backends.
// No f64 in WGSL: f32 permanently, and the 64-bit key schedule runs on the host.
// Prelude, include_str!'d ahead of the other shaders; not a pipeline itself.

const TAU: f32 = 6.2831853071795864769;

// 2^-32, matching SCALE in threefry.rs: the full 32-bit word, not a float
// mantissa, so Box-Muller reaches ~6.66 sigma.
const SCALE: f32 = 1.0 / 4294967296.0;

// No rotate builtin. Every Threefry constant is in [5, 27], so 32u - r stays in range.
fn rot_l(x: u32, r: u32) -> u32 {
    return (x << r) | (x >> (32u - r));
}

// High 32 bits of a 32x32 product via four 16-bit partials: WGSL has no mulhi, no u64.
fn mul_hi(a: u32, b: u32) -> u32 {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);

    return a1 * b1 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
}

// Four Threefry rounds plus one key injection: rounds 4s-4 .. 4s-1 of the 20.
// `ra`/`rb` are the (0,1)/(0,3) and (2,3)/(2,1) rotation constants, as ROT.
fn tf_quad(v: vec4<u32>, ra: vec4<u32>, rb: vec4<u32>, kj: vec4<u32>, s: u32) -> vec4<u32> {
    var x = v;

    x.x = x.x + x.y;
    x.y = rot_l(x.y, ra.x) ^ x.x;
    x.z = x.z + x.w;
    x.w = rot_l(x.w, rb.x) ^ x.z;

    x.x = x.x + x.w;
    x.w = rot_l(x.w, ra.y) ^ x.x;
    x.z = x.z + x.y;
    x.y = rot_l(x.y, rb.y) ^ x.z;

    x.x = x.x + x.y;
    x.y = rot_l(x.y, ra.z) ^ x.x;
    x.z = x.z + x.w;
    x.w = rot_l(x.w, rb.z) ^ x.z;

    x.x = x.x + x.w;
    x.w = rot_l(x.w, ra.w) ^ x.x;
    x.z = x.z + x.y;
    x.y = rot_l(x.y, rb.w) ^ x.z;

    return x + vec4<u32>(kj.x, kj.y, kj.z, kj.w + s);
}

// The 128-bit block at counter (index, draw). The rotation schedule has period 8, so the
// five key-injection groups alternate between two constant sets.
fn tf_block(k0: u32, k1: u32, k2: u32, k3: u32, k4: u32, lo: u32, hi: u32, draw: u32) -> vec4<u32> {
    let ra = vec4<u32>(10u, 11u, 13u, 23u);
    let rb = vec4<u32>(26u, 21u, 27u, 5u);
    let rc = vec4<u32>(6u, 17u, 25u, 18u);
    let rd = vec4<u32>(20u, 11u, 10u, 20u);
    var x = vec4<u32>(lo + k0, hi + k1, draw + k2, k3);

    x = tf_quad(x, ra, rb, vec4<u32>(k1, k2, k3, k4), 1u);
    x = tf_quad(x, rc, rd, vec4<u32>(k2, k3, k4, k0), 2u);
    x = tf_quad(x, ra, rb, vec4<u32>(k3, k4, k0, k1), 3u);
    x = tf_quad(x, rc, rd, vec4<u32>(k4, k0, k1, k2), 4u);
    x = tf_quad(x, ra, rb, vec4<u32>(k0, k1, k2, k3), 5u);

    return x;
}

// Four uniforms in (0, 1); | 1u keeps log(u) finite. f32(x) rounds to 24 bits,
// the only place the GPU path diverges from the CPU one.
fn unif4(x: vec4<u32>) -> vec4<f32> {
    return vec4<f32>(f32(x.x | 1u), f32(x.y | 1u), f32(x.z | 1u), f32(x.w | 1u)) * SCALE;
}

// Two Box-Muller pairs. Fixed consumption, no rejection loop, so the index -> value map
// holds; costs 7% against a ziggurat at f32.
fn norm4(u: vec4<f32>) -> vec4<f32> {
    let r0 = sqrt(-2.0 * log(u.x));
    let a0 = TAU * u.y;
    let r1 = sqrt(-2.0 * log(u.z));
    let a1 = TAU * u.w;

    return vec4<f32>(r0 * cos(a0), r0 * sin(a0), r1 * cos(a1), r1 * sin(a1));
}

fn draw4(k0: u32, k1: u32, k2: u32, k3: u32, k4: u32, idx: u32) -> vec4<f32> {
    return norm4(unif4(tf_block(k0, k1, k2, k3, k4, idx, 0u, 0u)));
}

// Wrap into [-pi, pi]; the host-supplied carry already bounds the argument.
fn wrapf(x: f32) -> f32 {
    return x - TAU * round(x / TAU);
}

// The pilot phase at symbol k, in 2^-32 cycles. frac is a 64-bit fixed-point
// (hi, lo) pair times k in exact u32 modular arithmetic: f32 loses it by 1e7.
fn ticks(hi: u32, lo: u32, k: u32) -> u32 {
    return hi * k + mul_hi(lo, k);
}

// Symbol-pipeline constants. `keys` holds five host-computed key words per
// stage, ordered alice, channel, detect, walk_tx, walk_lo.
struct SymCfg {
    keys: array<u32, 25>,
    bl: u32,
    nb: u32,
    ramp_hi: u32,
    ramp_lo: u32,
    ampl: f32,
    root_t: f32,
    ch_sd: f32,
    gain: f32,
    det_sd: f32,
    pilot_amp: f32,
    pilot_sd: f32,
    sd_tx: f32,
    sd_lo: f32,
}
