// Threefry-4x32-20, Salmon, Moraes, Dror & Shaw (SC'11); Philox needs a 32x32 mulhi WGSL
// cannot express. `Stream::new(seed, stage).normals(k)` is a pure function of (seed, stage, k)
// under any chunking or thread count, bit-exact against threefry.wgsl past counter 2^32.

/// Threefish key-schedule parity constant (Random123 `SKEIN_KS_PARITY`).
const PARITY: u32 = 0x1BD1_1BDA;

/// 2^-32, the full word: u_min = 2^-32 lets Box-Muller reach ~6.66 sigma.
const SCALE: f64 = 1.0 / 4_294_967_296.0;

/// Stage ids, one key per noise source, so adding one shifts no other.
pub(crate) const ALICE: u32 = 1;

pub(crate) const CHANNEL: u32 = 2;

pub(crate) const WALK_TX: u32 = 3;

pub(crate) const WALK_LO: u32 = 4;

pub(crate) const DETECT: u32 = 5;

/// Even round: mix (0,1) then (2,3). Rotation amounts are literals: a variable rotate
/// measures ~3x slower on the block.
macro_rules! even {
    ($x:ident, $ra:literal, $rb:literal) => {
        $x[0] = $x[0].wrapping_add($x[1]);
        $x[1] = $x[1].rotate_left($ra) ^ $x[0];
        $x[2] = $x[2].wrapping_add($x[3]);
        $x[3] = $x[3].rotate_left($rb) ^ $x[2];
    };
}

/// Odd round: mix (0,3) then (2,1).
macro_rules! odd {
    ($x:ident, $ra:literal, $rb:literal) => {
        $x[0] = $x[0].wrapping_add($x[3]);
        $x[3] = $x[3].rotate_left($ra) ^ $x[0];
        $x[2] = $x[2].wrapping_add($x[1]);
        $x[1] = $x[1].rotate_left($rb) ^ $x[2];
    };
}

/// Key injection after every fourth round, schedule word `$s`.
macro_rules! inject {
    ($x:ident, $ks:ident, $s:literal) => {
        $x[0] = $x[0].wrapping_add($ks[$s % 5]);
        $x[1] = $x[1].wrapping_add($ks[($s + 1) % 5]);
        $x[2] = $x[2].wrapping_add($ks[($s + 2) % 5]);
        $x[3] = $x[3].wrapping_add($ks[($s + 3) % 5]).wrapping_add($s as u32);
    };
}

/// SplitMix64 finalizer (Steele et al. 2014).
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

    z ^ (z >> 31)
}

/// One (seed, stage) pair, keyed once, indexed by symbol number.
#[derive(Clone, Copy)]
pub(crate) struct Stream {
    ks: [u32; 5],
}

impl Stream {
    /// Constants from SplittableRandom (Steele, Lea & Flood, OOPSLA 2014): the golden-ratio gamma
    /// spaces the stages, the seeder increment keeps `b` from being `mix(a)`. Every anchor in
    /// `test/` is drawn from this pair.
    pub(crate) fn new(seed: u64, stage: u32) -> Self {
        let a = mix(seed ^ (stage as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let b = mix(a ^ 0xD1B5_4A32_D192_ED03);
        let k = [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32];
        let ks = [
            k[0],
            k[1],
            k[2],
            k[3],
            PARITY ^ k[0] ^ k[1] ^ k[2] ^ k[3],
        ];

        Self { ks }
    }

    /// The five key words; the host runs the schedule, WGSL having no 64-bit integers.
    pub(crate) fn key(&self) -> [u32; 5] {
        self.ks
    }

    /// The 128-bit block at counter (index, draw); `backend.rs` compares the GPU words to it.
    ///
    /// Rotation amounts are the Random123 table `R_32x4_{r}_{0,1}` (threefry.wgsl's `ROT`),
    /// round r taking entry r % 8, first value for the (0,1)/(0,3) pair and second for
    /// (2,3)/(2,1). threefry.wgsl must keep agreeing word for word.
    ///
    /// Four counters in four lanes is a pessimisation, 0.68x in registers and 0.85-0.90x
    /// through `cpu_words`; do not re-propose it. Plain `[u32; 4]` macro-unrolled also beat
    /// hand `core::arch::aarch64`, and is portable across the wheel matrix.
    pub(crate) fn block(&self, index: u64, draw: u32) -> [u32; 4] {
        let ks = self.ks;
        let ctr = [index as u32, (index >> 32) as u32, draw, 0];
        let mut x = [
            ctr[0].wrapping_add(ks[0]),
            ctr[1].wrapping_add(ks[1]),
            ctr[2].wrapping_add(ks[2]),
            ctr[3].wrapping_add(ks[3]),
        ];

        even!(x, 10, 26);
        odd!(x, 11, 21);
        even!(x, 13, 27);
        odd!(x, 23, 5);
        inject!(x, ks, 1);
        even!(x, 6, 20);
        odd!(x, 17, 11);
        even!(x, 25, 10);
        odd!(x, 18, 20);
        inject!(x, ks, 2);
        even!(x, 10, 26);
        odd!(x, 11, 21);
        even!(x, 13, 27);
        odd!(x, 23, 5);
        inject!(x, ks, 3);
        even!(x, 6, 20);
        odd!(x, 17, 11);
        even!(x, 25, 10);
        odd!(x, 18, 20);
        inject!(x, ks, 4);
        even!(x, 10, 26);
        odd!(x, 11, 21);
        even!(x, 13, 27);
        odd!(x, 23, 5);
        inject!(x, ks, 5);

        x
    }

    /// Four uniforms in (0, 1); `| 1` keeps ln(u) finite at one ulp's cost.
    pub(crate) fn uniforms(&self, index: u64) -> [f64; 4] {
        let x = self.block(index, 0);

        [
            ((x[0] | 1) as f64) * SCALE,
            ((x[1] | 1) as f64) * SCALE,
            ((x[2] | 1) as f64) * SCALE,
            ((x[3] | 1) as f64) * SCALE,
        ]
    }

    /// Four standard normals, two Box-Muller pairs. Not ziggurat: a rejection sampler's
    /// variable uniform count breaks the index map threefry.wgsl is compared against.
    pub(crate) fn normals(&self, index: u64) -> [f64; 4] {
        let u = self.uniforms(index);
        let r0 = (-2.0 * u[0].ln()).sqrt();
        let (s0, c0) = (std::f64::consts::TAU * u[1]).sin_cos();
        let r1 = (-2.0 * u[2].ln()).sqrt();
        let (s1, c1) = (std::f64::consts::TAU * u[3]).sin_cos();

        [r0 * c0, r0 * s0, r1 * c1, r1 * s1]
    }
}
