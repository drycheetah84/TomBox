//! RDNA2 → HLSL translator (stage 3 of the shader pipeline).
//!
//! Model: one HLSL invocation == one RDNA2 *lane*. Every VGPR/SGPR is a `uint` holding the
//! raw 32-bit pattern (floats via asfloat/asuint), so no type inference is needed. Lane
//! masks (exec, vcc, compare results in SGPRs) are represented by their bit for *this*
//! lane (0/1). Descriptor-table pointers in user SGPRs are tracked symbolically: an
//! `s_load_dwordxN` from a never-written SGPR is a descriptor fetch from a root table, and
//! the way the descriptor is consumed (s_buffer_load → CBV, MIMG resource → SRV, MIMG
//! sampler → sampler) classifies the table. Slots map 1:1 to HLSL registers because the
//! title's root signature (UE layout) uses one SRV/CBV/sampler table per stage from
//! register 0.
//!
//! Scope: forward-only control flow (no loops yet). Anything not understood returns `Err`
//! naming the instruction — the caller must fall back; nothing is guessed.

use crate::meta::SigElement;
use crate::rdna2::{disassemble, Enc, Insn, Opnd};
use crate::{ShaderExtract, Stage};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;

pub struct Translated {
    pub hlsl: String,
    pub entry: &'static str,
    /// FXC/D3DCompile target profile.
    pub profile: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sym {
    Table(u8),
    Desc { table: u8, off: u32, k: u32 },
    /// Dword `k` of one of several descriptors (control-flow join of different `s_load`s,
    /// e.g. a loop that picks texture i of N). `set` indexes `Emitter::multi_sets`; which
    /// candidate is live is tracked at run time in a `tb_dsel<reg>` variable.
    Multi { set: u16, k: u32 },
    /// A PC-relative pointer into the shader's own code section (`s_getpc_b64` + offsets),
    /// used to read constant tables embedded after the program.
    Code,
}

/// An image op whose texture and/or sampler register holds one of several descriptors.
/// Each candidate is `(descriptor id, resolved resource)`; the id is what the run-time
/// selector `tb_dsel<reg>` holds.
#[derive(Clone, Debug)]
struct MSel {
    tex_reg: u16,
    samp_reg: u16,
    tex: Vec<(u32, Res)>,
    samp: Vec<(u32, Res)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Res {
    Cb(u32),
    Tex(u32, u32), // slot, dim
    Samp(u32),
    Uav(u32, u32), // slot, dim
    Buf(u32, bool), // slot, is_uav (structured buffer)
    TBuf(u32, bool), // slot, is_uav (typed buffer)
    CbDyn(u32),      // constant buffer read through MUBUF (per-lane offset)
    CodeTbl,         // s_load from the code section (embedded constant table)
}

const XYZW: [&str; 4] = ["x", "y", "z", "w"];

/// RDNA2 cube-map helpers. v_cube{ma,sc,tc,id}_f32 follow the ISA definition (major axis ×2,
/// face-local S/T, face index 0..5 = +X,-X,+Y,-Y,+Z,-Z). Hardware cube sampling takes
/// (s, t, face) with s,t = sc/|ma| + 1.5 (so each is in [1, 2]); `tb_cube_dir` inverts that
/// to the direction vector HLSL TextureCube sampling expects.
const CUBE_HELPERS: &str = r#"uint tb_perm_byte(uint hi, uint lo, uint sel) { if (sel >= 13u) return 0xffu; if (sel == 12u) return 0u; if (sel == 11u) return (hi >> 31) * 0xffu; if (sel == 10u) return ((hi >> 15) & 1u) * 0xffu; if (sel == 9u) return (lo >> 31) * 0xffu; if (sel == 8u) return ((lo >> 15) & 1u) * 0xffu; return sel >= 4u ? ((hi >> ((sel - 4u) * 8u)) & 0xffu) : ((lo >> (sel * 8u)) & 0xffu); }
uint tb_perm(uint hi, uint lo, uint sel) { return tb_perm_byte(hi, lo, sel & 0xffu) | (tb_perm_byte(hi, lo, (sel >> 8) & 0xffu) << 8) | (tb_perm_byte(hi, lo, (sel >> 16) & 0xffu) << 16) | (tb_perm_byte(hi, lo, sel >> 24) << 24); }
bool tb_cmp_class(uint b, uint mask) { uint e = (b >> 23) & 0xffu; uint m = b & 0x7fffffu; bool neg = (b >> 31) != 0u; uint c; if (e == 0xffu) { c = (m != 0u) ? (((m & 0x400000u) != 0u) ? 1u : 0u) : (neg ? 2u : 9u); } else if (e == 0u) { c = (m == 0u) ? (neg ? 5u : 6u) : (neg ? 4u : 7u); } else { c = neg ? 3u : 8u; } return ((mask >> c) & 1u) != 0u; }uint tb_bfe_i32(uint a, uint b) { uint o = b & 31u, w = (b >> 16) & 127u; if (w == 0u) return 0u; if (w + o >= 32u) return asuint(asint(a) >> o); return asuint(asint(a << (32u - w - o)) >> (32u - w)); }uint tb_mulhi_u32(uint a, uint b) { uint a0 = a & 0xffffu, a1 = a >> 16, b0 = b & 0xffffu, b1 = b >> 16; uint p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1; uint mid = (p00 >> 16) + (p01 & 0xffffu) + (p10 & 0xffffu); return p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16); }
uint tb_mulhi_i32(uint a, uint b) { uint h = tb_mulhi_u32(a, b); if (asint(a) < 0) h -= b; if (asint(b) < 0) h -= a; return h; }
float tb_cubema(float x, float y, float z) { float ax = abs(x), ay = abs(y), az = abs(z); if (az >= ax && az >= ay) return 2.0 * z; if (ay >= ax) return 2.0 * y; return 2.0 * x; }
float tb_cubesc(float x, float y, float z) { float ax = abs(x), ay = abs(y), az = abs(z); if (az >= ax && az >= ay) return z < 0.0 ? -x : x; if (ay >= ax) return x; return x < 0.0 ? z : -z; }
float tb_cubetc(float x, float y, float z) { float ax = abs(x), ay = abs(y), az = abs(z); if (az >= ax && az >= ay) return -y; if (ay >= ax) return y < 0.0 ? -z : z; return -y; }
float tb_cubeid(float x, float y, float z) { float ax = abs(x), ay = abs(y), az = abs(z); if (az >= ax && az >= ay) return z < 0.0 ? 5.0 : 4.0; if (ay >= ax) return y < 0.0 ? 3.0 : 2.0; return x < 0.0 ? 1.0 : 0.0; }
float3 tb_cube_dir(float s, float t, float face) { float u = (s - 1.5) * 2.0, v = (t - 1.5) * 2.0; int f = (int)round(face); if (f == 0) return float3(1.0, -v, -u); if (f == 1) return float3(-1.0, -v, u); if (f == 2) return float3(u, 1.0, v); if (f == 3) return float3(u, -1.0, -v); if (f == 4) return float3(u, -v, 1.0); return float3(-u, -v, -1.0); }
float tb_mull(float a, float b) { return (a == 0.0 || b == 0.0) ? 0.0 : a * b; }
"#;

fn name(o: &Opnd) -> Result<String, String> {
    Ok(match o {
        Opnd::Sgpr(n) => format!("s{n}"),
        Opnd::Vgpr(n) => format!("v{n}"),
        Opnd::VccLo => "vcc_lo".into(),
        Opnd::VccHi => "vcc_hi".into(),
        Opnd::ExecLo => "exec_lo".into(),
        Opnd::ExecHi => "exec_hi".into(),
        Opnd::M0 => "m0".into(),
        Opnd::Scc => "scc".into(),
        Opnd::Null => "nul".into(),
        Opnd::Temp(n) => format!("tb_s{n}"),
        other => return Err(format!("operand {other:?} is not assignable")),
    })
}

/// Operand as a `uint` rvalue expression.
fn r(o: &Opnd) -> Result<String, String> {
    Ok(match o {
        Opnd::ConstInt(v) => format!("{:#x}u", *v as i32 as u32),
        Opnd::ConstF32(f) => format!("{:#x}u", f.to_bits()),
        Opnd::Literal(v) => format!("{v:#x}u"),
        Opnd::Raw(c) => return Err(format!("unmodelled operand code {c}")),
        other => name(other)?,
    })
}

fn f(o: &Opnd) -> Result<String, String> {
    Ok(format!("asfloat({})", r(o)?))
}

/// Scalar register number of an operand (SGPRs, plus vcc_lo/vcc_hi = 106/107, which SMEM
/// loads can target directly: `s_buffer_load_dwordx2 vcc_lo, ...`).
fn sreg_num(o: &Opnd) -> Option<u8> {
    match o {
        Opnd::Sgpr(n) => Some(*n),
        Opnd::VccLo => Some(106),
        Opnd::VccHi => Some(107),
        _ => None,
    }
}

fn sreg_opnd(n: u8) -> Opnd {
    match n {
        106 => Opnd::VccLo,
        107 => Opnd::VccHi,
        n => Opnd::Sgpr(n),
    }
}

/// High half of a 64-bit scalar operand.
fn hi(o: &Opnd) -> Opnd {
    match o {
        Opnd::Sgpr(n) => Opnd::Sgpr(n + 1),
        Opnd::VccLo => Opnd::VccHi,
        Opnd::ExecLo => Opnd::ExecHi,
        Opnd::ConstInt(v) => Opnd::ConstInt(if *v < 0 { -1 } else { 0 }),
        Opnd::Null => Opnd::Null,
        _ => Opnd::ConstInt(0),
    }
}

/// Name of the SGPR used as an SMEM dynamic offset (SOFFSET field).
fn ssrc_name(code: u32) -> Result<String, String> {
    if code <= 105 { Ok(format!("s{code}")) } else { Err(format!("unsupported SOFFSET register code {code}")) }
}

fn sig_type(e: &SigElement) -> (String, usize) {
    // Component count = number of mask bits (elements can be packed into a shared register,
    // e.g. TEXCOORD4.xy + TEXCOORD5.zw, so the highest bit is not the count).
    let comps = (e.mask.count_ones() as usize).max(1);
    let base = match e.comp_type {
        1 => "uint",
        2 => "int",
        _ => "float",
    };
    (if comps == 1 { base.to_string() } else { format!("{base}{comps}") }, comps)
}

/// Map a hardware (register, component) to (signature element index, component within
/// that element). RDNA2 attributes / parameter exports are whole registers; a register may
/// hold several packed signature elements.
fn sig_lookup(sig: &[SigElement], register: u32, chan: usize) -> Option<(usize, usize)> {
    sig.iter().enumerate().find_map(|(idx, e)| {
        if e.sysval == 0 && e.register == register && (e.mask >> chan) & 1 != 0 {
            Some((idx, (e.mask & ((1u8 << chan) - 1)).count_ones() as usize))
        } else {
            None
        }
    })
}

/// Distinct registers of the given elements, in signature order.
fn distinct_regs<'s>(it: impl Iterator<Item = &'s SigElement>) -> Vec<u32> {
    let mut v: Vec<u32> = Vec::new();
    for e in it {
        if !v.contains(&e.register) {
            v.push(e.register);
        }
    }
    v
}

fn semantic(e: &SigElement) -> String {
    format!("{}{}", e.name, e.index)
}

struct Emitter<'a> {
    ex: &'a ShaderExtract,
    insns: Vec<Insn>,
    res: HashMap<usize, Vec<Res>>, // per-instruction resolved resources
    cb_size: BTreeMap<u32, u32>,   // slot -> dwords
    tex: BTreeMap<u32, u32>,       // slot -> dim
    samp: BTreeSet<u32>,
    samp_cmp: BTreeSet<u32>,
    uav: BTreeMap<u32, u32>,       // slot -> dim
    sbuf: BTreeMap<(bool, u32), u32>, // (is_uav, slot) -> element dwords used
    tbuf: BTreeSet<(bool, u32)>,      // typed buffers (is_uav, slot)
    tbuf_uint: BTreeSet<u32>,         // typed UAV slots whose element must be uint (atomics)
    /// Inferred integer element type of typed buffers: (is_uav, slot) -> 1 uint, 2 sint.
    tbuf_ty: BTreeMap<(bool, u32), u8>,
    /// Inferred integer texel type of textures read with texel fetches / written with image
    /// stores: (is_uav, slot) -> 1 uint, 2 sint. Absent = float (the default).
    img_ty: BTreeMap<(bool, u32), u8>,
    /// Pixel-shader colour-export epilogues (see `find_export_epilogs`): pack instruction
    /// index -> capture id, and export instruction index -> the capture ids feeding its
    /// two packed operands (usize::MAX = operand not written).
    epilog_pack: HashMap<usize, usize>,
    epilog_exp: HashMap<usize, (usize, usize)>,
    /// Index of the instruction being emitted.
    cur_idx: usize,
    /// Structured buffers read with scalar loads (`s_buffer_load` through an SRV/UAV
    /// descriptor): they need the byte-offset helper.
    sbuf_flat: BTreeSet<(bool, u32)>,
    is_cs: bool,
    /// Nesting depth of emitted HLSL loops at the current instruction.
    loop_depth: u32,
    is_gs: bool,
    /// Compute shader uses LDS (groupshared memory).
    uses_lds: bool,
    /// Shader reads constant tables embedded in its code section.
    uses_code: bool,
    /// The program uses register-relative moves: VGPRs live in the `tb_v` array.
    vreg_array: bool,
    /// Interned candidate lists `(table, offset)` for `Sym::Multi`.
    multi_sets: Vec<Vec<(u8, u32)>>,
    /// Image ops whose texture and/or sampler is one of several descriptors.
    /// UAV texture slots used with image atomics (declared `<uint>`).
    uav_uint: BTreeSet<u32>,
    /// The shader uses wave intrinsics (needs shader model 6: DXC only).
    uses_wave: bool,
    msel: HashMap<usize, MSel>,
    /// Buffer ops (MUBUF, s_buffer_load) whose descriptor register holds one of several
    /// descriptors: (register key, candidates as (descriptor id, resource)).
    bsel: HashMap<usize, (u16, Vec<(u32, Res)>)>,
    /// Descriptor loads `(base register, descriptor id)` and base-dword moves `(dst, src)`
    /// per instruction, for maintaining the run-time `tb_dsel<reg>` selectors.
    dsel_loads: HashMap<usize, (u16, u32)>,
    dsel_moves: HashMap<usize, (u16, u16)>,
    /// Registers read before written (entry ABI), from `live_in`.
    live: Vec<String>,
    /// Pixel shader needs its position within the 2x2 quad (ds_swizzle).
    uses_quad: bool,
    regs: BTreeSet<String>,
    body: String,
    tmp: usize,
    /// PS attribute index -> input register (registers the shader actually reads).
    ps_attrs: Vec<u32>,
    /// VS parameter-export index -> output register.
    out_params: Vec<u32>,
}

/// SGPR spills through vector lanes.
///
/// When a shader runs out of scalar registers the console compiler parks scalars (observed:
/// texture/sampler descriptor dwords) in individual lanes of an otherwise unused VGPR with
/// `v_writelane_b32 vN, sX, <lane>` and brings them back with `v_readlane_b32 sY, vN, <lane>`
/// (VOP3 0x361 / 0x360; fixture `fortnite_lobby_desc.dxbc`). The translator models one lane
/// per VGPR, so each (VGPR, constant lane) slot is lowered to a function-scope translator
/// temporary and the two instructions become plain `s_mov_b32`. The descriptor dataflow
/// tracks those temporaries like registers (keys 1000+), so descriptors survive a spill.
///
/// Refused (explicit error): a non-constant lane index, a read of a slot that was never
/// written, or a spill VGPR that is also used as an ordinary vector.
/// Synthetic opcode of the wave-reduction instruction produced by `lower_wave_reductions`.
const OP_WAVE_REDUCE: u32 = 0x7ff0;

/// Whole-wave reductions (HLSL `WaveActiveMin` / `WaveActiveMax`; observed in
/// CocFlattenMainCS, 4f9670de0be37288). The compiled form is:
///
/// ```text
///   s_nand_saveexec_b64 sX, 0        ; save exec, enable every lane
///   v_mov_b32     vI, <identity>
///   v_cndmask_b32 vR, vI, vIn, sX    ; inactive lanes contribute the identity
///   v_min_f32     vR, <dpp>, vR      ; DPP row/quad permutes (src0 = 233 / 250)
///   ...  vop3<0x378>, v_min_f32, v_readlane_b32 (lane 32), v_min_f32 ...
///   v_readlane_b32 sRes, vR, 0       ; the result, a scalar
///   s_mov_b64     exec_lo, sX        ; restore exec
/// ```
///
/// A per-lane translation cannot express the cross-lane data movement, but the net effect
/// is exactly a wave reduction, so the region is replaced by: `sX = exec` (exec itself is
/// left alone), the identity move, and one synthetic instruction
/// `sRes = WaveActive{Min,Max}(lane active ? vIn : identity)`. Needs shader model 6 (DXC).
///
/// Inferred, not documented: operand codes 233/250 as DPP with a trailing control dword and
/// SOP1 0x29 as nand-saveexec are taken from the public GFX10 encoding and are consistent
/// with this shader's instruction stream; the meaning of VOP3 0x378 is not needed.
/// Anything in the region that does not fit this shape is refused.
fn lower_wave_reductions(insns: &mut [Insn]) -> Result<(), String> {
    let is_dpp = |i: &Insn| i.enc == Enc::Vop2 && matches!(i.src.first(), Some(Opnd::Raw(233 | 234 | 250)));
    let mut at = 0;
    while let Some(first) = (at..insns.len()).find(|&i| is_dpp(&insns[i])) {
        let off = insns[first].offset;
        let bad = |what: &str| format!("wave reduction at {off:#x}: {what}");
        let red = insns[first].mnemonic.clone();
        if !matches!(red.as_str(), "v_min_f32" | "v_max_f32") {
            return Err(bad(&format!("unsupported reduction op {red}")));
        }
        let Some(Opnd::Vgpr(vr)) = insns[first].dst.first().copied() else { return Err(bad("no accumulator")) };
        // Prologue: nand_saveexec, identity move, cndmask.
        if first < 3 {
            return Err(bad("no prologue"));
        }
        let (a, mv, cm) = (first - 3, first - 2, first - 1);
        let saved = match (&insns[a].enc, insns[a].op, insns[a].dst.first(), insns[a].src.first()) {
            (Enc::Sop1, 0x29, Some(Opnd::Sgpr(x)), Some(Opnd::ConstInt(0))) => *x,
            _ => return Err(bad("prologue does not start with s_nand_saveexec_b64 sX, 0")),
        };
        let (ident_reg, ident) = match (insns[mv].mnemonic.as_str(), insns[mv].dst.first(), insns[mv].src.first()) {
            ("v_mov_b32", Some(Opnd::Vgpr(v)), Some(Opnd::Literal(k))) => (*v, *k),
            _ => return Err(bad("identity is not a literal move")),
        };
        let c = &insns[cm];
        let vin = match (c.mnemonic.as_str(), c.dst.first(), c.src.first(), c.src.get(1), c.src.get(2)) {
            ("v_cndmask_b32", Some(Opnd::Vgpr(d)), Some(Opnd::Vgpr(i)), Some(Opnd::Vgpr(x)), Some(Opnd::Sgpr(m)))
                if *d == vr && *i == ident_reg && *m == saved =>
            {
                *x
            }
            _ => return Err(bad("unexpected select before the reduction")),
        };
        // Body up to the exec restore.
        let mut end = None;
        let mut last_read = None;
        for i in first..insns.len() {
            let x = &insns[i];
            if x.mnemonic == "s_mov_b64" && x.dst.first() == Some(&Opnd::ExecLo) && x.src.first() == Some(&Opnd::Sgpr(saved)) {
                end = Some(i);
                break;
            }
            let ok = (x.enc == Enc::Vop2 && x.mnemonic == red && x.dst.first() == Some(&Opnd::Vgpr(vr)))
                || (x.enc == Enc::Vop3 && x.op == 0x378 && x.src.first() == Some(&Opnd::Vgpr(vr)))
                || (x.enc == Enc::Vop3 && x.op == 0x360 && x.src.first() == Some(&Opnd::Vgpr(vr)));
            if !ok {
                return Err(bad(&format!("unexpected {} at {:#x} inside the reduction", x.mnemonic, x.offset)));
            }
            if x.enc == Enc::Vop3 && x.op == 0x360 {
                last_read = Some(i);
            }
        }
        let (Some(end), Some(last)) = (end, last_read) else { return Err(bad("no exec restore / result read")) };
        if insns[last].src.get(1) != Some(&Opnd::ConstInt(0)) {
            return Err(bad("result is not read from lane 0"));
        }
        let res = match insns[last].dst.first() {
            Some(Opnd::Vgpr(n)) | Some(Opnd::Sgpr(n)) => *n,
            _ => return Err(bad("bad result register")),
        };
        // Rewrite.
        insns[a].mnemonic = "s_mov_b64".into();
        insns[a].op = 4;
        insns[a].src = vec![Opnd::ExecLo];
        for i in cm..end {
            let x = &mut insns[i];
            x.enc = Enc::Vop1;
            x.op = 0;
            x.mnemonic = "v_nop".into();
            x.dst = vec![Opnd::Vgpr(vr)];
            x.src = vec![];
            x.aux = Default::default();
        }
        let x = &mut insns[last];
        x.enc = Enc::Vop3;
        x.op = OP_WAVE_REDUCE;
        x.mnemonic = if red == "v_min_f32" { "tb_wave_min_f32".into() } else { "tb_wave_max_f32".into() };
        x.dst = vec![Opnd::Sgpr(res)];
        x.src = vec![Opnd::Vgpr(vin), Opnd::Sgpr(saved)];
        x.aux.imm = ident as i64;
        at = end + 1;
    }
    Ok(())
}

/// Translator temporaries from this index up are lane-spill slots (declared up front).
const LANE_TEMP_BASE: u16 = 10_000;

fn lower_lane_spills(insns: &mut [Insn]) -> Result<(), String> {
    const READLANE: u32 = 0x360;
    const WRITELANE: u32 = 0x361;
    let is_lane = |i: &Insn| i.enc == Enc::Vop3 && (i.op == READLANE || i.op == WRITELANE);
    if !insns.iter().any(is_lane) {
        return Ok(());
    }
    let mut slots: BTreeMap<(u8, i64), Opnd> = BTreeMap::new();
    let mut spill_vregs: BTreeSet<u8> = BTreeSet::new();
    for ins in insns.iter_mut() {
        if !is_lane(ins) {
            continue;
        }
        let at = ins.offset;
        let lane = match ins.src.get(1) {
            Some(Opnd::ConstInt(l)) if (0..64).contains(l) => *l,
            other => return Err(format!("lane op at {at:#x}: non-constant lane operand {other:?}")),
        };
        if ins.op == WRITELANE {
            let Some(Opnd::Vgpr(v)) = ins.dst.first().cloned() else {
                return Err(format!("v_writelane at {at:#x}: destination is not a VGPR"));
            };
            let next = slots.len();
            let slot = match slots.get(&(v, lane)) {
                Some(s) => *s,
                None => {
                    let s = Opnd::Temp(LANE_TEMP_BASE + next as u16);
                    slots.insert((v, lane), s);
                    s
                }
            };
            spill_vregs.insert(v);
            let value = ins.src.first().cloned().ok_or_else(|| format!("v_writelane at {at:#x}: no source"))?;
            ins.dst = vec![slot];
            ins.src = vec![value];
        } else {
            // The destination field of v_readlane names an SGPR (decoded generically as a VGPR).
            let d = match ins.dst.first() {
                Some(Opnd::Vgpr(n)) | Some(Opnd::Sgpr(n)) => *n,
                other => return Err(format!("v_readlane at {at:#x}: unexpected destination {other:?}")),
            };
            let Some(Opnd::Vgpr(v)) = ins.src.first().cloned() else {
                return Err(format!("v_readlane at {at:#x}: source is not a VGPR"));
            };
            let Some(&slot) = slots.get(&(v, lane)) else {
                return Err(format!("v_readlane at {at:#x}: v{v} lane {lane} read before any v_writelane"));
            };
            ins.dst = vec![Opnd::Sgpr(d)];
            ins.src = vec![slot];
        }
        ins.enc = Enc::Sop1;
        ins.op = 0x03;
        ins.mnemonic = "s_mov_b32".into();
    }
    for ins in insns.iter() {
        if ins.mnemonic == "v_nop" {
            continue; // the two header words at the start of every program decode as v_nop
        }
        for o in ins.dst.iter().chain(ins.src.iter()) {
            if let Opnd::Vgpr(v) = o {
                if spill_vregs.contains(v) {
                    return Err(format!("v{v} is used both as a lane spill area and as a vector at {:#x}", ins.offset));
                }
            }
        }
    }
    Ok(())
}

pub fn translate(ex: &ShaderExtract) -> Result<Translated, String> {
    let (profile, is_ps) = match ex.stage {
        Stage::Vertex => ("vs_5_0", false),
        Stage::Pixel => ("ps_5_0", true),
        Stage::Compute => ("cs_5_0", false),
        Stage::Geometry => ("gs_5_0", false),
        other => return Err(format!("stage {other:?} not supported yet")),
    };
    let mut insns = disassemble(&ex.isa);
    lower_wave_reductions(&mut insns)?;
    lower_lane_spills(&mut insns)?;
    let mut em = Emitter {
        loop_depth: 0,
        ex,
        insns,
        res: HashMap::new(),
        cb_size: BTreeMap::new(),
        tex: BTreeMap::new(),
        samp: BTreeSet::new(),
        samp_cmp: BTreeSet::new(),
        uav: BTreeMap::new(),
        sbuf: BTreeMap::new(),
        tbuf: BTreeSet::new(),
        tbuf_uint: BTreeSet::new(),
        tbuf_ty: BTreeMap::new(),
        img_ty: BTreeMap::new(),
        epilog_pack: HashMap::new(),
        epilog_exp: HashMap::new(),
        cur_idx: 0,
        sbuf_flat: BTreeSet::new(),
        is_cs: ex.stage == Stage::Compute,
        is_gs: ex.stage == Stage::Geometry,
        uses_lds: false,
        uses_code: false,
        vreg_array: false,
        multi_sets: Vec::new(),
        uav_uint: BTreeSet::new(),
        uses_wave: false,
        msel: HashMap::new(),
        bsel: HashMap::new(),
        dsel_loads: HashMap::new(),
        dsel_moves: HashMap::new(),
        live: Vec::new(),
        uses_quad: false,
        regs: BTreeSet::new(),
        body: String::new(),
        tmp: 0,
        // PS attribute slot k = the k-th input *register* that carries an interpolant and is
        // read. A register counts as read if any element packed into it is read — including
        // a system value sharing it (observed: PARTICLE_RANDOM unread + SV_IsFrontFace read
        // in the same register still occupies an attribute slot; d26fd88ff8712f9e).
        ps_attrs: distinct_regs(ex.input_sig.iter().filter(|e| {
            e.sysval == 0 && ex.input_sig.iter().any(|o| o.register == e.register && o.rw_mask != 0)
        })),
        out_params: distinct_regs(ex.output_sig.iter().filter(|e| e.sysval == 0)),
    };
    em.vreg_array = em.insns.iter().any(|i| i.mnemonic.starts_with("v_movrel"));
    // Lane-spill slots beyond the tracked range are function-scope temporaries.
    for ins in &em.insns {
        for o in ins.dst.iter().chain(ins.src.iter()) {
            if let Opnd::Temp(n) = o {
                if *n >= LANE_TEMP_BASE {
                    em.regs.insert(format!("tb_s{n}"));
                }
            }
        }
    }
    em.resolve_resources()?;
    if is_ps {
        em.find_export_epilogs();
    }
    em.infer_tbuf_types();
    let live_in = em.live_in();
    em.live = live_in.clone();
    em.emit_body(is_ps)?;
    let mut hlsl = em.finish(is_ps)?;
    hlsl.insert_str(0, &format!("// entry-live registers (read before written): {}
", live_in.join(" ")));
    // ps_5_0 forces UAV slots to start after the render-target outputs (FXC X4509);
    // SM5.1 has no such rule, and D3D12 accepts SM5.1 bytecode.
    let profile = if is_ps && !em.uav.is_empty() { "ps_5_1" } else { profile };
    Ok(Translated { hlsl, entry: "main", profile })
}

impl<'a> Emitter<'a> {
    /// Root-signature range type (0 SRV, 1 UAV, 2 CBV, 3 Sampler) of the descriptor table
    /// whose pointer lives in user SGPR `sgpr` at shader entry. The FSSM register table
    /// maps user SGPR (base + i) to root parameter i, with base 8 for graphics stages and
    /// 4 for compute (observed: PS s8/s9/s10 = params 0/1/2, VS s12 = param 4, CS s5/s7 =
    /// params 1/3). `None` when the root signature is unavailable or the SGPR is not a table.
    fn table_kind(&self, sgpr: u8) -> Option<u32> {
        // Root descriptor-table pointers arrive in consecutive user SGPRs: from s4 in compute
        // shaders and s8 in the other stages. Vertex shaders compiled to feed a geometry
        // shader ("MainForGS") start with `s_swappc_b64 s12, s12` and receive them from s19
        // 7 registers after the swapped PC register pair instead (observed: `s_swappc_b64
        // s12, s12` -> SRV table s19, CBV s20 in a55400ccab602283; `s_swappc_b64 s4, s4` ->
        // SRV s11, CBV s12 in 4d318a496eebf6d3).
        let chained = self.insns.iter().take(16).find(|i| i.mnemonic == "s_swappc_b64").and_then(|i| match i.dst.first() {
            Some(Opnd::Sgpr(n)) => Some(*n as usize + 7),
            _ => None,
        });
        // Geometry shaders: SRV table in s22, CBV table in s23 (6a6bdf123db8b5b0).
        let base = if self.is_cs { 4 } else if self.ex.stage == Stage::Geometry { 22 } else { chained.unwrap_or(8) };
        let idx = (sgpr as usize).checked_sub(base)?;
        match &self.ex.root_sig.as_ref()?.params.get(idx)?.kind {
            crate::meta::RootParamKind::Table(r) if r.len() == 1 && r[0].base_register == 0 => Some(r[0].range_type),
            _ => None,
        }
    }

    /// Classify descriptor loads and bind them to HLSL registers.
    ///
    /// Code is not laid out in execution order (blocks are reached through backward
    /// branches), so the symbolic SGPR state is computed as a forward dataflow fixpoint over
    /// the control-flow graph: state = (known symbolic values, set of written SGPRs), join =
    /// keep only values every predecessor agrees on / union of written sets. A final pass
    /// then walks each block with its entry state and records the resources.
    fn resolve_resources(&mut self) -> Result<(), String> {
        type State = (HashMap<u16, Sym>, BTreeSet<u16>);
        let n = self.insns.len();
        if n == 0 {
            return Ok(());
        }
        let off2idx: HashMap<usize, usize> = self.insns.iter().enumerate().map(|(i, x)| (x.offset, i)).collect();
        // Block leaders and per-instruction branch targets.
        let mut leaders: BTreeSet<usize> = BTreeSet::new();
        leaders.insert(0);
        let mut tgt: HashMap<usize, usize> = HashMap::new();
        for (i, ins) in self.insns.iter().enumerate() {
            if ins.enc != Enc::Sopp {
                continue;
            }
            if Self::branch_cond(ins.op).is_some() {
                let t = ins.offset as i64 + 4 + ins.aux.imm * 4;
                if let Some(&ti) = off2idx.get(&(t as usize)) {
                    tgt.insert(i, ti);
                    leaders.insert(ti);
                }
                leaders.insert(i + 1);
            } else if ins.op == 1 {
                leaders.insert(i + 1);
            }
        }
        let starts: Vec<usize> = leaders.into_iter().filter(|&s| s < n).collect();
        let block_of = |i: usize| starts.partition_point(|&s| s <= i) - 1;
        let ranges: Vec<std::ops::Range<usize>> =
            (0..starts.len()).map(|b| starts[b]..starts.get(b + 1).copied().unwrap_or(n)).collect();
        let succs: Vec<Vec<usize>> = ranges
            .iter()
            .map(|r| {
                let last = r.end - 1;
                let ins = &self.insns[last];
                let mut v = Vec::new();
                let is_sopp = ins.enc == Enc::Sopp;
                if let Some(&t) = tgt.get(&last) {
                    v.push(block_of(t));
                }
                let falls = !(is_sopp && (ins.op == 1 || ins.op == 2));
                if falls && r.end < n {
                    v.push(block_of(r.end));
                }
                v
            })
            .collect();

        // Fixpoint (analysis only: recordings are rolled back afterwards).
        let saved = (
            self.res.clone(),
            self.cb_size.clone(),
            self.tex.clone(),
            self.samp.clone(),
            self.samp_cmp.clone(),
            self.uav.clone(),
            self.sbuf.clone(),
            self.tbuf.clone(),
            self.tbuf_uint.clone(),
        );
        let mut first_err: Option<String> = None;
        let mut states: Vec<Option<State>> = vec![None; ranges.len()];
        states[0] = Some((HashMap::new(), BTreeSet::new()));
        let mut work = vec![0usize];
        let mut budget = 20_000usize;
        while let Some(b) = work.pop() {
            if budget == 0 {
                return Err("descriptor dataflow did not converge".into());
            }
            budget -= 1;
            let (mut sym, mut written) = states[b].clone().unwrap();
            if let Err(e) = self.res_block(&mut sym, &mut written, ranges[b].clone()) {
                // Unknown effect: forget everything and treat every SGPR as written. Keep
                // the first failure — it is the root cause if the recording pass fails too.
                first_err.get_or_insert(e);
                sym.clear();
                written = (0u16..=105).collect();
            }
            for &s in &succs[b] {
                let changed = match &mut states[s] {
                    slot @ None => {
                        *slot = Some((sym.clone(), written.clone()));
                        true
                    }
                    Some((osym, owr)) => {
                        let before = (osym.clone(), owr.len());
                        let keys: Vec<u16> = osym.keys().copied().collect();
                        for k in keys {
                            let a = osym[&k];
                            match sym.get(&k).copied() {
                                Some(b) if a == b => {}
                                Some(b) => match self.join_desc(a, b) {
                                    Some(j) => {
                                        osym.insert(k, j);
                                    }
                                    None => {
                                        osym.remove(&k);
                                    }
                                },
                                // Known on one path only. A descriptor stays (see below);
                                // anything else is dropped.
                                None if matches!(a, Sym::Desc { .. } | Sym::Multi { .. } | Sym::Table(_)) => {}
                                None => {
                                    osym.remove(&k);
                                }
                            }
                        }
                        // Optimistic join for descriptors: a register that holds a descriptor
                        // on one incoming path and nothing known on the other is treated as
                        // holding the descriptor. Compiled code loads descriptors inside
                        // blocks that are skipped only when no lane is active
                        // (`s_cbranch_execz`), then uses them after the join; on the skipped
                        // path the result is unused. A later use as a descriptor would be
                        // invalid anyway if the register held something else.
                        for (k, v) in sym.iter() {
                            if matches!(v, Sym::Desc { .. } | Sym::Multi { .. } | Sym::Table(_)) && !osym.contains_key(k) {
                                osym.insert(*k, *v);
                            }
                        }
                        owr.extend(written.iter().copied());
                        before != (osym.clone(), owr.len())
                    }
                };
                if changed && !work.contains(&s) {
                    work.push(s);
                }
            }
        }
        (self.res, self.cb_size, self.tex, self.samp, self.samp_cmp, self.uav, self.sbuf, self.tbuf, self.tbuf_uint) = saved;

        // TB_DFDEBUG=<sgpr>: print that register's symbol at every block entry (diagnostics).
        if let Some(reg) = std::env::var("TB_DFDEBUG").ok().and_then(|v| v.parse::<u16>().ok()) {
            for (b, r) in ranges.iter().enumerate() {
                let s = states[b].as_ref().map(|(sym, wr)| (sym.get(&reg).copied(), wr.contains(&reg)));
                eprintln!("block {b} @{:#x}..: s{reg} = {s:?} succs={:?}", self.insns[r.start].offset, succs[b]);
            }
            eprintln!("first analysis error: {first_err:?}");
        }
        self.msel.clear();
        self.bsel.clear();
        self.dsel_loads.clear();
        self.dsel_moves.clear();
        // Recording pass with the converged entry states.
        for (b, r) in ranges.iter().enumerate() {
            if let Some((mut sym, mut written)) = states[b].clone() {
                self.res_block(&mut sym, &mut written, r.clone()).map_err(|e| first_err.clone().unwrap_or(e))?;
            }
        }
        Ok(())
    }

    /// Element type of each typed buffer, inferred from how the shader uses the values it
    /// loads from (or produces for) the buffer. The buffer view's format is not part of the
    /// shader: the hardware delivers floats for float/normalised formats and integers for
    /// integer formats, and the compiled code simply goes on to treat the registers as one
    /// or the other. HLSL needs the element type declared, so it is recovered from the
    /// first typed uses: conversions from / comparisons as / arithmetic on integers or
    /// floats, and use as a memory index. Bitwise logic and moves are followed, not counted.
    /// Slots with no evidence, or more float than integer evidence, stay float.
    fn infer_tbuf_types(&mut self) {
        // 0 float, 1 uint, 2 sint
        fn class(mn: &str) -> Option<u8> {
            let m = mn.strip_suffix("_e64").unwrap_or(mn);
            if let Some(rest) = m.strip_prefix("v_cvt_") {
                let last = rest.rsplit('_').next().unwrap_or("");
                return Some(if last.starts_with('f') { 0 } else if last.starts_with('i') { 2 } else { 1 });
            }
            if m.starts_with("v_cmp") {
                return if m.contains("_f") { Some(0) } else if m.contains("_i") { Some(2) } else if m.contains("_u") { Some(1) } else { None };
            }
            const SINT: [&str; 7] = ["v_ashr", "v_max_i", "v_min_i", "v_bfe_i", "v_mul_i32", "v_mad_i32", "v_mul_hi_i"];
            const UINT: [&str; 14] = [
                "v_lshl", "v_lshr", "v_bfe_u", "v_mul_lo", "v_mul_hi_u", "v_mul_u32", "v_mad_u32", "v_add_nc", "v_sub_nc", "v_subrev_nc", "v_add_co", "v_sub_co",
                "v_max_u", "v_min_u",
            ];
            if SINT.iter().any(|p| m.starts_with(p)) {
                return Some(2);
            }
            if UINT.iter().any(|p| m.starts_with(p)) {
                return Some(1);
            }
            if m.starts_with("v_") && (m.contains("_f32") || m.contains("_f16") || m.contains("_f64")) {
                return Some(0);
            }
            None
        }
        // Bitwise logic with a small constant (`x & 1`, `x & 0xf`, `x ^ 1`) is integer work:
        // the masks applied to float bit patterns are sign / exponent / mantissa masks, all
        // far larger.
        fn small_mask(u: &Insn) -> bool {
            ["v_and_b32", "v_or_b32", "v_xor_b32"].iter().any(|p| u.mnemonic.starts_with(p))
                && u.src.iter().any(|o| match o {
                    Opnd::ConstInt(c) => *c > 0 && *c <= 64,
                    Opnd::Literal(l) => *l > 0 && *l <= 0xffff,
                    _ => false,
                })
        }
        fn follows(mn: &str) -> bool {
            ["v_mov_b32", "v_and_b32", "v_or_b32", "v_xor_b32", "v_cndmask_b32", "v_not_b32"].iter().any(|p| mn.starts_with(p))
        }
        // Keys are (is image, is_uav, slot): typed buffers and textures are told apart.
        let mut votes: BTreeMap<(bool, bool, u32), [u32; 3]> = BTreeMap::new();
        // (stored-to resource, loaded-from resource): a store of registers loaded from another
        // typed resource says both hold the same kind of element.
        let mut links: Vec<((bool, bool, u32), (bool, bool, u32))> = Vec::new();
        // Textures that are sampled or gathered hold filterable (float) data.
        let mut sampled: BTreeSet<u32> = BTreeSet::new();
        let n = self.insns.len();
        // The typed resource an instruction reads or writes, with the data registers.
        let typed = |em: &Self, i: usize| -> Option<((bool, bool, u32), BTreeSet<u8>, bool, bool)> {
            let ins = &em.insns[i];
            let Some(&Opnd::Vgpr(vd)) = ins.dst.first() else { return None };
            match em.res.get(&i).and_then(|v| v.first()).copied()? {
                Res::TBuf(slot, is_uav) => {
                    let comps = match ins.mnemonic.rsplit('_').next() {
                        Some("x") => 1u8,
                        Some("xy") => 2,
                        Some("xyz") => 3,
                        Some("xyzw") => 4,
                        _ => return None,
                    };
                    let regs = (0..comps).map(|k| vd.wrapping_add(k)).collect();
                    Some(((false, is_uav, slot), regs, ins.mnemonic.starts_with("buffer_load"), ins.mnemonic.starts_with("buffer_store")))
                }
                r @ (Res::Tex(..) | Res::Uav(..)) => {
                    let (slot, is_uav) = match r {
                        Res::Tex(s, _) => (s, false),
                        Res::Uav(s, _) => (s, true),
                        _ => unreachable!(),
                    };
                    let comps = (ins.aux.mask & 0xf).count_ones() as u8;
                    let regs = (0..comps).map(|k| vd.wrapping_add(k)).collect();
                    let load = matches!(ins.mnemonic.as_str(), "image_load" | "image_load_mip");
                    Some(((true, is_uav, slot), regs, load, ins.mnemonic == "image_store"))
                }
                _ => None,
            }
        };
        for i in 0..n {
            if let Some(Res::Tex(slot, _)) = self.res.get(&i).and_then(|v| v.first()).copied() {
                let m = &self.insns[i].mnemonic;
                if m.starts_with("image_sample") || m.starts_with("image_gather") {
                    sampled.insert(slot);
                }
            }
            let Some((key, regs, is_load, is_store)) = typed(self, i) else { continue };
            let v = votes.entry(key).or_insert([0; 3]);
            if is_load {
                // Forward: how are the loaded registers (and plain copies / masks of them) used?
                let mut live = regs;
                for j in i + 1..n {
                    if live.is_empty() {
                        break;
                    }
                    let u = &self.insns[j];
                    // The scan is linear: past an unconditional branch (or the program end)
                    // the following code belongs to another path.
                    if u.enc == Enc::Sopp && (u.op == 1 || u.op == 2) {
                        break;
                    }
                    let uses = u.src.iter().any(|o| matches!(o, Opnd::Vgpr(r) if live.contains(r)));
                    let mem_index = matches!(u.enc, Enc::Mubuf | Enc::Mimg | Enc::Ds) && matches!(u.src.first(), Some(Opnd::Vgpr(r)) if live.contains(r));
                    if mem_index && u.enc == Enc::Mimg && (u.mnemonic.starts_with("image_sample") || u.mnemonic.starts_with("image_gather")) {
                        // Sampling coordinates are floats; only texel fetches index by integer.
                        v[0] += 1;
                    } else if mem_index && !(u.mnemonic.contains("store") && matches!(u.enc, Enc::Mubuf)) {
                        v[1] += 1;
                    } else if uses && !self.epilog_pack.contains_key(&j) {
                        // (The export tail packs whatever type the output has: no evidence.)
                        if let Some(c) = class(&u.mnemonic) {
                            v[c as usize] += 1;
                        } else if small_mask(u) {
                            v[1] += 1;
                        }
                    }
                    let carry = uses && follows(&u.mnemonic);
                    for d in &u.dst {
                        if let Opnd::Vgpr(r) = d {
                            if carry {
                                live.insert(*r);
                            } else if !matches!(u.enc, Enc::Mubuf | Enc::Ds | Enc::Exp) || u.mnemonic.contains("load") || u.mnemonic.contains("read") {
                                live.remove(r);
                            }
                        }
                    }
                }
            } else if is_store {
                // Backward: what produced each stored register?
                for r in regs {
                    for j in (0..i).rev() {
                        let d = &self.insns[j];
                        if d.dst.iter().any(|o| matches!(o, Opnd::Vgpr(x) if *x == r)) {
                            if let Some((k2, _, true, _)) = typed(self, j) {
                                links.push((key, k2));
                                break;
                            }
                            // A conversion's result type is its first type token.
                            let c = match d.mnemonic.strip_prefix("v_cvt_") {
                                Some(rest) => Some(if rest.starts_with('f') { 0 } else if rest.contains("i32") || rest.contains("i16") { 2 } else { 1 }),
                                None => class(&d.mnemonic),
                            };
                            if let Some(c) = c {
                                v[c as usize] += 1;
                            }
                            break;
                        }
                    }
                }
            }
        }
        let mut ty: BTreeMap<(bool, bool, u32), u8> = BTreeMap::new();
        for (k, v) in &votes {
            let ints = v[1] + v[2];
            if ints > v[0] {
                ty.insert(*k, if v[2] > 0 { 2 } else { 1 });
            }
        }
        for (to, from) in links {
            if votes.get(&to).map(|v| v.iter().all(|&x| x == 0)).unwrap_or(true) {
                if let Some(&t) = ty.get(&from) {
                    ty.insert(to, t);
                }
            }
        }
        for ((is_img, is_uav, slot), t) in ty {
            if !is_img {
                self.tbuf_ty.insert((is_uav, slot), t);
            } else if is_uav {
                // Textures used with image atomics already have their own uint declaration.
                if !self.uav_uint.contains(&slot) {
                    self.img_ty.insert((true, slot), t);
                }
            } else if !sampled.contains(&slot) {
                self.img_ty.insert((false, slot), t);
            }
        }
    }

    /// Recognise the colour-export epilogue of a pixel shader.
    ///
    /// Every colour export in the title's pixel shaders has the same shape, whatever the
    /// HLSL output type:
    ///
    /// ```text
    /// s_nop ; s_nop ; v_cvt_pkrtz_f16_f32 vA, x, y ; v_cvt_pkrtz_f16_f32 vB, z, w ; exp mrtN vA, vB (compressed)
    /// ```
    ///
    /// including shaders whose output is `uint4` (the lighting-channel copy writes an R16_UINT
    /// target this way, where a float-to-half conversion of the integer would always give 0).
    /// The export format belongs to the render target, which the shader compiler does not
    /// know, so this tail can only be a default that the console's pipeline creation replaces
    /// per target format (the two leading no-ops are spare room for it). That replacement is
    /// inferred, not observed. What the shader itself computes is the four unpacked operands
    /// (x, y, z, w), so those are exported directly: as integers for an integer output
    /// element, as full floats otherwise (the render-target write then converts to the
    /// target's own format, as a per-format epilogue would).
    ///
    /// Records the pack instructions whose operands must be captured and, per export, which
    /// captures feed it. An export is left alone unless both packed operands come from pack
    /// instructions directly before it with nothing but no-ops in between and no branch
    /// landing inside the sequence.
    fn find_export_epilogs(&mut self) {
        let mut off2idx: HashMap<usize, usize> = HashMap::new();
        for (i, ins) in self.insns.iter().enumerate() {
            off2idx.insert(ins.offset, i);
        }
        let mut targets: BTreeSet<usize> = BTreeSet::new();
        for ins in &self.insns {
            if ins.enc == Enc::Sopp && Self::branch_cond(ins.op).is_some() {
                let t = ins.offset as i64 + 4 + ins.aux.imm * 4;
                if t >= 0 {
                    if let Some(&ti) = off2idx.get(&(t as usize)) {
                        targets.insert(ti);
                    }
                }
            }
        }
        let mut next_id = 0usize;
        for idx in 0..self.insns.len() {
            let ins = &self.insns[idx];
            if ins.enc != Enc::Exp || ins.aux.imm2 & 1 == 0 || ins.aux.imm > 7 {
                continue;
            }
            let (Some(&Opnd::Vgpr(r0)), Some(&Opnd::Vgpr(r1))) = (ins.src.first(), ins.src.get(1)) else { continue };
            let en = ins.aux.mask;
            let (need0, need1) = (en & 0x3 != 0, en & 0xc != 0);
            let (mut p0, mut p1) = (None, None);
            let mut first = idx;
            for j in (idx.saturating_sub(8)..idx).rev() {
                let u = &self.insns[j];
                if matches!(u.mnemonic.as_str(), "s_nop" | "s_waitcnt") {
                    continue;
                }
                if u.mnemonic != "v_cvt_pkrtz_f16_f32" {
                    break;
                }
                let Some(&Opnd::Vgpr(d)) = u.dst.first() else { break };
                if d == r0 && p0.is_none() {
                    p0 = Some(j);
                    first = j;
                }
                if d == r1 && p1.is_none() {
                    p1 = Some(j);
                    first = j;
                }
            }
            if (need0 && p0.is_none()) || (need1 && p1.is_none()) {
                continue;
            }
            if targets.range(first + 1..=idx).next().is_some() {
                continue;
            }
            let mut id_of = |em: &mut Self, p: Option<usize>, need: bool| -> usize {
                match p {
                    Some(j) if need => *em.epilog_pack.entry(j).or_insert_with(|| {
                        next_id += 1;
                        next_id - 1
                    }),
                    _ => usize::MAX,
                }
            };
            let a = id_of(self, p0, need0);
            let b = id_of(self, p1, need1);
            self.epilog_exp.insert(idx, (a, b));
        }
    }

    /// Declared element size, in dwords, of a structured buffer.
    ///
    /// The stride is a property of the buffer view, not of the shader: on the console the
    /// hardware multiplies the element index by the descriptor's stride. D3D12 instead takes
    /// the stride from the HLSL element type (observed: a translation declaring a 12-byte
    /// element for the 16-byte primitive-data buffer read the wrong rows), so the declared
    /// element has to be the real one. It is not recoverable in general. What the PC build
    /// of the same game shows for 801 buffers bound at the same register: stride 16 for
    /// every buffer the Xbox code touches with extents of 8..28 bytes, 64 for extents of 64.
    /// Hence: extents of 8..28 bytes -> 16-byte elements (larger offsets address following
    /// elements), anything larger rounded up to a multiple of 16. A buffer only ever read or
    /// written one dword at offset 0 keeps a 4-byte element: ScatterCopyCS reads its index
    /// buffer that way and its view really has stride 4 (forcing 16 scrambled the per-object
    /// data it uploads); no 16-byte buffer in the comparison was touched like that.
    /// A heuristic (TECHNICAL_DEBT).
    fn sbuf_words(&self, key: (bool, u32)) -> u32 {
        let raw = self.sbuf.get(&key).copied().unwrap_or(1).max(1);
        match raw {
            1 => 1,
            2..=7 => 4,
            _ => raw.div_ceil(4) * 4,
        }
    }

    /// Resource a MUBUF instruction addresses when its descriptor is `slot` of `table`
    /// (also records declaration requirements).
    fn mubuf_res(&mut self, ins: &Insn, slot: u32, table: u8) -> Result<Res, String> {
        let is_uav = match self.table_kind(table) {
            Some(0) => false,
            Some(1) => true,
            Some(2) => {
                // Per-lane indexed read of a constant buffer (e.g. cbuffer arrays indexed by
                // a varying value): size the declaration to the maximum.
                self.cb_size.insert(slot, 4096 * 4);
                return Ok(Res::CbDyn(slot));
            }
            other => return Err(format!("mubuf at {:#x}: buffer in table of kind {other:?}", ins.offset)),
        };
        if ins.mnemonic.contains("_atomic_") {
            if !is_uav {
                return Err(format!("mubuf at {:#x}: atomic on a non-UAV buffer", ins.offset));
            }
            // PSV0 tells typed (6) from structured (8) UAVs. Atomics need a uint element, so
            // a typed UAV used atomically is declared RWBuffer<uint>.
            let structured = self.ex.resources.iter().any(|p| p.res_type == 8 && p.lower <= slot && slot <= p.upper);
            if structured {
                let e = self.sbuf.entry((true, slot)).or_insert(0);
                *e = (*e).max(ins.aux.imm as u32 / 4 + 1);
                return Ok(Res::Buf(slot, true));
            }
            self.tbuf.insert((true, slot));
            self.tbuf_uint.insert(slot);
            return Ok(Res::TBuf(slot, true));
        }
        if ins.mnemonic.contains("_format_") {
            // Typed buffer (Buffer<T> / RWBuffer<T>), addressed by element index.
            self.tbuf.insert((is_uav, slot));
            return Ok(Res::TBuf(slot, is_uav));
        }
        let ndw = Self::mubuf_dwords(&ins.mnemonic).ok_or_else(|| format!("mubuf op {} unsupported", ins.mnemonic))?;
        let end = ins.aux.imm as u32 / 4 + ndw;
        let e = self.sbuf.entry((is_uav, slot)).or_insert(0);
        *e = (*e).max(end);
        Ok(Res::Buf(slot, is_uav))
    }

    /// Candidate descriptors `(table, offset)` of a symbol that is dword 0 of a descriptor.
    fn desc_candidates(&self, s: Option<&Sym>) -> Option<Vec<(u8, u32)>> {
        match s {
            Some(Sym::Desc { table, off, k: 0 }) => Some(vec![(*table, *off)]),
            Some(Sym::Multi { set, k: 0 }) => self.multi_sets.get(*set as usize).cloned(),
            _ => None,
        }
    }

    /// Join of two different descriptor symbols at a control-flow merge: the union of
    /// their candidates when both are the same dword of some descriptor, else nothing.
    fn join_desc(&mut self, a: Sym, b: Sym) -> Option<Sym> {
        let parts = |s: Sym, sets: &Vec<Vec<(u8, u32)>>| -> Option<(Vec<(u8, u32)>, u32)> {
            match s {
                Sym::Desc { table, off, k } => Some((vec![(table, off)], k)),
                Sym::Multi { set, k } => sets.get(set as usize).map(|v| (v.clone(), k)),
                _ => None,
            }
        };
        let (mut ca, ka) = parts(a, &self.multi_sets)?;
        let (cb, kb) = parts(b, &self.multi_sets)?;
        if ka != kb {
            return None;
        }
        ca.extend(cb);
        ca.sort();
        ca.dedup();
        if ca.len() > 16 {
            return None; // not a small "one of N" selection
        }
        let set = match self.multi_sets.iter().position(|v| *v == ca) {
            Some(i) => i,
            None => {
                self.multi_sets.push(ca);
                self.multi_sets.len() - 1
            }
        };
        Some(Sym::Multi { set: set as u16, k: ka })
    }

    /// Transfer function for one basic block (also records resources / reports errors).
    fn res_block(
        &mut self,
        sym: &mut HashMap<u16, Sym>,
        written: &mut BTreeSet<u16>,
        range: std::ops::Range<usize>,
    ) -> Result<(), String> {
        let insns = self.insns.clone();
        let dbg_reg = std::env::var("TB_DFDEBUG").ok().and_then(|v| v.parse::<u16>().ok());
        let mut dbg_prev: Option<Sym> = dbg_reg.and_then(|r| sym.get(&r).copied());
        for idx in range {
            if let Some(r) = dbg_reg {
                let cur = sym.get(&r).copied();
                if cur != dbg_prev && idx > 0 {
                    eprintln!("  s{r}: {dbg_prev:?} -> {cur:?} by {:#x} {}", insns[idx - 1].offset, insns[idx - 1].mnemonic);
                }
                dbg_prev = cur;
            }
            let ins = &insns[idx];
            // Dataflow key of a scalar location: SGPR number, or 1000+ for a lane-spill slot.
            let sg = |o: &Opnd| match o {
                Opnd::Sgpr(n) => Some(*n as u16),
                Opnd::Temp(n) if *n >= LANE_TEMP_BASE => Some(1000 + (*n - LANE_TEMP_BASE)),
                _ => None,
            };
            match ins.enc {
                Enc::Sop1 if ins.mnemonic == "s_mov_b32" => {
                    if let Some(d) = ins.dst.first().and_then(sg) {
                        match ins.src.first().and_then(sg).and_then(|s| sym.get(&s).cloned()) {
                            Some(v) => {
                                sym.insert(d, v);
                            }
                            None => {
                                // A never-written user SGPR copied before use is still a table pointer.
                                let s = ins.src.first().and_then(sg);
                                match s {
                                    Some(s) if !written.contains(&s) => {
                                        sym.insert(d, Sym::Table(s as u8));
                                    }
                                    _ => {
                                        sym.remove(&d);
                                    }
                                }
                            }
                        }
                        written.insert(d);
                        if let Some(s) = ins.src.first().and_then(sg) {
                            if matches!(sym.get(&d), Some(Sym::Desc { k: 0, .. }) | Some(Sym::Multi { k: 0, .. })) {
                                self.dsel_moves.insert(idx, (d, s));
                            }
                        }
                    }
                }
                Enc::Sop1 if ins.mnemonic == "s_mov_b64" => {
                    // 64-bit moves copy descriptors two dwords at a time
                    // (`s_mov_b64 s44, s84` + `s_mov_b64 s46, s86` moves a 4-dword descriptor).
                    if let Some(d) = ins.dst.first().and_then(sg) {
                        let s = ins.src.first().and_then(sg);
                        for k in 0..2u16 {
                            match s.and_then(|s| sym.get(&(s + k)).cloned()) {
                                Some(v) => {
                                    if k == 0 && matches!(v, Sym::Desc { k: 0, .. } | Sym::Multi { k: 0, .. }) {
                                        self.dsel_moves.insert(idx, (d, s.unwrap_or(d)));
                                    }
                                    sym.insert(d + k, v);
                                }
                                None => match s {
                                    // A never-written user SGPR is a root table pointer (see
                                    // s_mov_b32); a 64-bit copy carries it in the low half.
                                    Some(s) if k == 0 && !written.contains(&s) => {
                                        sym.insert(d, Sym::Table(s as u8));
                                    }
                                    _ => {
                                        sym.remove(&(d + k));
                                    }
                                },
                            }
                            written.insert(d + k);
                        }
                    }
                }
                // s_setreg_* program a hardware mode register; the encoding's register
                // field is not an SGPR destination.
                Enc::Sopk if ins.mnemonic.starts_with("s_setreg") => {}
                // A code pointer taken through a VGPR: `v = index*4 + sCode` followed by
                // `v_readfirstlane sBase, v` and `s_load [sBase]` is a constant-table lookup
                // with a run-time index (ColorDeficiencyMain, cc45bda725a0f774). VGPRs are
                // tracked under keys 2000+ for exactly this.
                Enc::Vop2 | Enc::Vop3
                    if matches!(ins.mnemonic.as_str(), "v_add_co_u32" | "v_add_nc_u32")
                        && ins.src.iter().any(|o| sg(o).map(|s| sym.get(&s) == Some(&Sym::Code)).unwrap_or(false)) =>
                {
                    if let Some(Opnd::Vgpr(v)) = ins.dst.first() {
                        sym.insert(2000 + *v as u16, Sym::Code);
                    }
                }
                Enc::Vop1 if ins.mnemonic == "v_readfirstlane_b32" => {
                    let code = matches!(ins.src.first(), Some(Opnd::Vgpr(v)) if sym.get(&(2000 + *v as u16)) == Some(&Sym::Code));
                    if let Some(d) = ins.dst.first().and_then(sg) {
                        if code {
                            sym.insert(d, Sym::Code);
                        } else {
                            sym.remove(&d);
                        }
                        written.insert(d);
                    }
                }
                Enc::Sop1 if ins.mnemonic == "s_getpc_b64" => {
                    if let Some(d) = ins.dst.first().and_then(sg) {
                        sym.insert(d, Sym::Code);
                        sym.remove(&(d + 1));
                        written.insert(d);
                        written.insert(d + 1);
                    }
                }
                Enc::Sop2 if matches!(ins.mnemonic.as_str(), "s_add_u32" | "s_add_i32") => {
                    // Pointer arithmetic keeps a code pointer a code pointer.
                    let is_code = ins.src.iter().any(|o| sg(o).map(|s| sym.get(&s) == Some(&Sym::Code)).unwrap_or(false));
                    if let Some(d) = ins.dst.first().and_then(sg) {
                        if is_code {
                            sym.insert(d, Sym::Code);
                        } else {
                            sym.remove(&d);
                        }
                        written.insert(d);
                    }
                }
                Enc::Smem if ins.op <= 4 && ins.src.first().and_then(sg).map(|b| sym.get(&b) == Some(&Sym::Code)).unwrap_or(false) => {
                    // Read from an embedded constant table.
                    let n = 1u32 << ins.op;
                    let d = ins.dst.first().and_then(sg).ok_or("s_load: bad dst")?;
                    if (ins.raw.get(1).copied().unwrap_or(0) >> 25) != 0x7d {
                        return Err(format!("s_load at {:#x}: code-table read with dynamic soffset unsupported", ins.offset));
                    }
                    self.uses_code = true;
                    self.res.insert(idx, vec![Res::CodeTbl]);
                    for k in 0..n {
                        sym.remove(&(d + k as u16));
                        written.insert(d + k as u16);
                    }
                }
                Enc::Smem if ins.op <= 4 => {
                    let n = 1u32 << ins.op;
                    let b = ins.src.first().and_then(sg).ok_or("s_load: bad base")?;
                    let d = ins.dst.first().and_then(sg).ok_or("s_load: bad dst")?;
                    let table = match sym.get(&b) {
                        Some(Sym::Table(t)) => *t,
                        None if !written.contains(&b) => b as u8,
                        other => {
                            return Err(format!(
                                "s_load at {:#x}: base s{b} is not a root table pointer ({other:?})",
                                ins.offset
                            ))
                        }
                    };
                    if (ins.raw.get(1).copied().unwrap_or(0) >> 25) != 0x7d {
                        return Err(format!("s_load at {:#x}: dynamic soffset unsupported", ins.offset));
                    }
                    for k in 0..n {
                        sym.insert(d + k as u16, Sym::Desc { table, off: ins.aux.imm as u32, k });
                        self.dsel_loads.insert(idx, (d, ((table as u32) << 24) | ins.aux.imm as u32));
                        written.insert(d + k as u16);
                    }
                }
                Enc::Smem => {
                    let n = 1u32 << (ins.op - 8);
                    let b = ins.src.first().and_then(sg).ok_or("s_buffer_load: bad base")?;
                    let d = ins.dst.first().and_then(sreg_num).map(u16::from).ok_or("s_buffer_load: bad dst")?;
                    // CBV descriptors sit on a 32-byte stride (cb1 is loaded from table offset
                    // 0x20 in MainOnePassPointLightPS).
                    let cands = self.desc_candidates(sym.get(&b)).ok_or_else(|| {
                        format!("s_buffer_load at {:#x}: base s{b} is not a buffer descriptor ({:?})", ins.offset, sym.get(&b))
                    })?;
                    // A dynamic SOFFSET register indexes the buffer at run time: size the
                    // cbuffer declaration to the D3D12 maximum (64 KiB = 4096 vec4).
                    let dynamic = (ins.raw.get(1).copied().unwrap_or(0) >> 25) != 0x7d;
                    let end = if dynamic { 4096 * 4 } else { ins.aux.imm as u32 / 4 + n };
                    let mut out: Vec<(u32, Res)> = Vec::new();
                    // A scalar load reads whatever buffer the descriptor names. From the CBV
                    // table that is a constant buffer; from the SRV / UAV table it is a
                    // (structured) buffer addressed by byte offset, e.g. the sky-light
                    // spherical-harmonics buffer read by ReflectionEnvironmentSkyLighting.
                    // Only the unambiguous case is handled; with several candidate
                    // descriptors the load is still treated as a constant buffer (debt).
                    let buffer_kind = if cands.len() == 1 { self.table_kind(cands[0].0).filter(|k| *k <= 1) } else { None };
                    for (table, off) in &cands {
                        let slot = off / 32;
                        if let Some(k) = buffer_kind {
                            let key = (k == 1, slot);
                            let e = self.sbuf.entry(key).or_insert(0);
                            *e = (*e).max(4);
                            self.sbuf_flat.insert(key);
                            out.push((((*table as u32) << 24) | off, Res::Buf(slot, k == 1)));
                            continue;
                        }
                        let e = self.cb_size.entry(slot).or_insert(0);
                        *e = (*e).max(end);
                        out.push((((*table as u32) << 24) | off, Res::Cb(slot)));
                    }
                    self.res.insert(idx, vec![out[0].1]);
                    if out.len() > 1 {
                        self.bsel.insert(idx, (b, out));
                    }
                    for k in 0..n {
                        sym.remove(&(d + k as u16));
                        written.insert(d + k as u16);
                    }
                }
                // buffer_gl0_inv / buffer_gl1_inv (MUBUF 0x71 / 0x72) invalidate the GPU data caches: no
                // resource, no data, and no PC equivalent.
                Enc::Mubuf if ins.op == 0x71 || ins.op == 0x72 => {}
                Enc::Mubuf => {
                    let rs = ins.src.get(1).and_then(sg).ok_or("mubuf: bad srsrc")?;
                    let cands = self.desc_candidates(sym.get(&rs)).ok_or_else(|| {
                        format!("mubuf at {:#x}: s{rs} not a buffer descriptor ({:?})", ins.offset, sym.get(&rs))
                    })?;
                    let mut out: Vec<(u32, Res)> = Vec::new();
                    for (table, off) in &cands {
                        out.push((((*table as u32) << 24) | off, self.mubuf_res(ins, off / 32, *table)?));
                    }
                    self.res.insert(idx, vec![out[0].1]);
                    if out.len() > 1 {
                        self.bsel.insert(idx, (rs, out));
                    }
                }
                Enc::Mimg => {
                    let n = ins.src.len();
                    let rs = ins.src.get(n - 2).and_then(sg).ok_or("mimg: bad srsrc")?;
                    let ss = ins.src.get(n - 1).and_then(sg).ok_or("mimg: bad ssamp")?;
                    let dim = ins.aux.imm as u32;
                    let tex_c = self.desc_candidates(sym.get(&rs)).ok_or_else(|| {
                        format!("mimg at {:#x}: s{rs} not an image descriptor ({:?})", ins.offset, sym.get(&rs))
                    })?;
                    let mut tex_res: Vec<(u32, Res)> = Vec::new();
                    for (ttable, off) in &tex_c {
                        let tslot = off / 32;
                        let is_uav = self.table_kind(*ttable) == Some(1);
                        if is_uav && ins.mnemonic.starts_with("image_atomic_") {
                            self.uav_uint.insert(tslot);
                        }
                        let map = if is_uav { &mut self.uav } else { &mut self.tex };
                        if let Some(prev) = map.insert(tslot, dim) {
                            if prev != dim {
                                return Err(format!("image slot {tslot} used with two dimensions"));
                            }
                        }
                        tex_res.push((((*ttable as u32) << 24) | off, if is_uav { Res::Uav(tslot, dim) } else { Res::Tex(tslot, dim) }));
                    }
                    let mut samp_res: Vec<(u32, Res)> = Vec::new();
                    if ins.mnemonic.starts_with("image_sample") || ins.mnemonic.starts_with("image_gather") {
                        let samp_c = self.desc_candidates(sym.get(&ss)).ok_or_else(|| {
                            format!("mimg at {:#x}: s{ss} not a sampler descriptor ({:?})", ins.offset, sym.get(&ss))
                        })?;
                        for (stable, off) in &samp_c {
                            let sslot = off / 16;
                            // Depth-compare variants (opcode bit 3) need a comparison sampler.
                            if ins.op & 8 != 0 {
                                self.samp_cmp.insert(sslot);
                            } else {
                                self.samp.insert(sslot);
                            }
                            samp_res.push((((*stable as u32) << 24) | off, Res::Samp(sslot)));
                        }
                    }
                    let mut v = vec![tex_res[0].1];
                    if let Some(s) = samp_res.first() {
                        v.push(s.1);
                    }
                    if tex_res.len() > 1 || samp_res.len() > 1 {
                        self.msel.insert(idx, MSel { tex_reg: rs, samp_reg: ss, tex: tex_res, samp: samp_res });
                    }
                    self.res.insert(idx, v);
                }

                _ => {
                    for d in &ins.dst {
                        if let Opnd::Vgpr(v) = d {
                            sym.remove(&(2000 + *v as u16));
                        }
                        if let Some(n) = sg(d) {
                            sym.remove(&n);
                            written.insert(n);
                            if ins.mnemonic.ends_with("_b64") {
                                sym.remove(&(n + 1));
                                written.insert(n + 1);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Registers read before any write in program order — i.e. the shader's entry ABI
    /// (interpolation barycentrics, fragment position, user-data SGPRs, thread ids...).
    fn live_in(&self) -> Vec<String> {
        let mut written: BTreeSet<String> = BTreeSet::new();
        let mut live: BTreeSet<(u8, u32)> = BTreeSet::new();
        for ins in &self.insns {
            if ins.mnemonic == "v_nop" {
                continue;
            }
            for o in &ins.src {
                if let Opnd::Sgpr(n) | Opnd::Vgpr(n) = o {
                    let nm = name(o).unwrap_or_default();
                    if !written.contains(&nm) {
                        live.insert((if matches!(o, Opnd::Vgpr(_)) { 1 } else { 0 }, *n as u32));
                    }
                }
            }
            // Multi-register destinations: SMEM loads, MIMG results, 64-bit scalar ops.
            let extra = match ins.enc {
                Enc::Smem => (1u32 << (ins.op & 7)) - 1,
                Enc::Mimg => (ins.aux.mask.count_ones()).saturating_sub(1),
                _ if ins.mnemonic.ends_with("_b64") => 1,
                _ => 0,
            };
            for o in &ins.dst {
                if let Ok(nm) = name(o) {
                    written.insert(nm);
                }
                match o {
                    Opnd::Sgpr(n) => (1..=extra).for_each(|k| { written.insert(format!("s{}", *n as u32 + k)); }),
                    Opnd::Vgpr(n) => (1..=extra).for_each(|k| { written.insert(format!("v{}", *n as u32 + k)); }),
                    _ => {}
                }
            }
        }
        live.iter().map(|(k, n)| format!("{}{}", if *k == 1 { "v" } else { "s" }, n)).collect()
    }

    fn line(&mut self, s: &str) {
        self.body.push_str("    ");
        self.body.push_str(s);
        self.body.push('\n');
    }

    fn touch(&mut self, o: &Opnd) {
        if let Opnd::Sgpr(_) | Opnd::Vgpr(_) = o {
            if let Ok(n) = name(o) {
                self.regs.insert(n);
            }
        }
    }

    /// Guarded write to a VGPR (inactive lanes keep their value).
    fn wv(&mut self, dst: &Opnd, expr: &str) -> Result<(), String> {
        self.touch(dst);
        let d = name(dst)?;
        self.line(&format!("if (exec_lo & 1u) {d} = {expr};"));
        Ok(())
    }

    fn ws(&mut self, dst: &Opnd, expr: &str) -> Result<(), String> {
        self.touch(dst);
        let d = name(dst)?;
        self.line(&format!("{d} = {expr};"));
        Ok(())
    }

    /// Condition under which a SOPP branch opcode is taken (None = not a branch).
    fn branch_cond(op: u32) -> Option<&'static str> {
        Some(match op {
            2 => "true",
            4 => "scc == 0u",
            5 => "scc != 0u",
            6 => "(vcc_lo & 1u) == 0u",
            7 => "(vcc_lo & 1u) != 0u",
            8 => "(exec_lo & 1u) == 0u",
            9 => "(exec_lo & 1u) != 0u",
            _ => return None,
        })
    }

    /// Control flow.
    ///
    /// * Forward branches use skip flags: `jT` is set when a branch to instruction T is
    ///   taken, and every instruction before T that lies after such a branch is wrapped in
    ///   `if (!(jT || ...))`. This handles arbitrary forward-only CFGs (if/else, breaks).
    /// * A backward branch at B to header L becomes `do { ... } while (cond)` around
    ///   [L..=B]. Flags for labels inside the body are reset each iteration; flags for
    ///   labels past B (loop exits) terminate the loop and keep skipping until their label.
    ///   Loops must be properly nested, have one back-edge per header, and must not be
    ///   entered from outside except through the header — otherwise `Err`.
    fn emit_body(&mut self, is_ps: bool) -> Result<(), String> {
        let mut off2idx: HashMap<usize, usize> = self.insns.iter().enumerate().map(|(i, x)| (x.offset, i)).collect();
        // A branch to the address just past the last instruction leaves the program (same as
        // reaching the final s_endpgm).
        if let Some(l) = self.insns.last() {
            off2idx.insert(l.offset + l.dwords * 4, self.insns.len());
        }
        let prog_end = self.insns.last().map_or(0, |l| l.offset + l.dwords * 4);
        let mut target: HashMap<usize, usize> = HashMap::new(); // forward: branch idx -> target idx
        let mut back: HashMap<usize, usize> = HashMap::new(); // backward: branch idx B -> header idx L
        let n_insns = self.insns.len();
        for (i, ins) in self.insns.iter().enumerate() {
            if ins.enc == Enc::Sopp && ins.op == 1 && i + 1 < n_insns {
                target.insert(i, n_insns); // early s_endpgm -> end of program
                continue;
            }
            if ins.enc == Enc::Sopp && Self::branch_cond(ins.op).is_some() {
                let t = ins.offset as i64 + 4 + ins.aux.imm * 4;
                let ti = if t >= 0 && t as usize >= prog_end {
                        self.insns.len() // at or beyond the end of the program (s_code_end padding): exit
                    } else {
                        *off2idx
                            .get(&(t as usize))
                            .ok_or_else(|| format!("branch at {:#x} targets mid-instruction {t:#x}", ins.offset))?
                    };
                if ti <= i {
                    back.insert(i, ti);
                } else {
                    target.insert(i, ti);
                }
            }
        }
        // Validate loop structure.
        // Several back-edges to one header = `continue`s. The last one (highest address)
        // closes the loop; each earlier one becomes a forward jump to that closing branch
        // which also forces another iteration (see the `|| j{B}` at the loop bottom).
        let mut heads: HashMap<usize, usize> = HashMap::new(); // L -> B
        for (&b, &l) in &back {
            let e = heads.entry(l).or_insert(b);
            *e = (*e).max(b);
        }
        let continues: Vec<(usize, usize)> =
            back.iter().filter(|(&b, &l)| heads[&l] != b).map(|(&b, &l)| (b, heads[&l])).collect();
        for (b, closing) in continues {
            back.remove(&b);
            target.insert(b, closing);
        }
        let loops: Vec<(usize, usize)> = heads.iter().map(|(&l, &b)| (l, b)).collect();
        for &(l1, b1) in &loops {
            for &(l2, b2) in &loops {
                let disjoint = b1 < l2 || b2 < l1;
                let nested = (l1 <= l2 && b2 <= b1) || (l2 <= l1 && b1 <= b2);
                if !(disjoint || nested) {
                    // Interleaved loops cannot be expressed as nested do/while: fall back
                    // to the general block dispatcher.
                    return self.emit_body_dispatch(is_ps);
                }
            }
            // A forward branch from before the header into the body (the usual rotated
            // `while`: jump straight to the condition) is handled by the skip flags: the
            // flag stays set on the first iteration, so everything before the label is
            // skipped, and in-body flags are cleared at the bottom of each iteration.
        }
        let labels: BTreeSet<usize> = target.values().copied().collect();
        for l in &labels {
            self.line(&format!("bool j{l} = false;"));
        }
        if !loops.is_empty() {
            self.line("uint tb_guard = 0u;");
            for &(_, b) in &loops {
                self.line(&format!("bool tb_c{b} = false;"));
            }
        }

        let insns = self.insns.clone();
        let mut open = false;
        let mut cur: Vec<usize> = vec![];
        for (i, ins) in insns.iter().enumerate() {
            if let Some(&b) = heads.get(&i) {
                if open {
                    self.line("}");
                    open = false;
                }
                cur.clear();
                self.line("do {");
                self.loop_depth += 1;
                self.line(&format!("tb_c{b} = false;"));
            }
            let active: Vec<usize> =
                labels.iter().copied().filter(|&t| t > i && target.iter().any(|(&b, &bt)| bt == t && b < i)).collect();
            if active != cur || (open && heads.contains_key(&i)) {
                if open {
                    self.line("}");
                    open = false;
                }
                if !active.is_empty() {
                    let c: Vec<String> = active.iter().map(|t| format!("j{t}")).collect();
                    self.line(&format!("if (!({})) {{", c.join(" || ")));
                    open = true;
                }
                cur = active;
            }
            for o in ins.src.iter().chain(ins.dst.iter()) {
                self.touch(o);
            }
            self.emit_insn(i, ins, is_ps, target.get(&i).copied())
                .map_err(|e| format!("{e} [{} @ {:#x}]", ins.mnemonic, ins.offset))?;
            if back.contains_key(&i) {
                // Evaluate the back-edge condition inside the guard (a skipped branch is not taken).
                let cond = Self::branch_cond(ins.op).unwrap_or("false");
                self.line(&format!("tb_c{i} = ({cond});"));
                if labels.contains(&i) {
                    // A `continue` jumped here: iterate again regardless of the condition.
                    self.line(&format!("tb_c{i} = tb_c{i} || j{i};"));
                }
                if open {
                    self.line("}");
                    open = false;
                }
                cur.clear();
                // Clear the flags of labels inside this loop body for the next iteration.
                let head = back[&i];
                for t in labels.iter().filter(|&&t| t > head && t <= i) {
                    self.line(&format!("j{t} = false;"));
                }
                let exits: Vec<String> = labels.iter().filter(|&&t| t > i).map(|t| format!("j{t}")).collect();
                let exit = if exits.is_empty() { String::new() } else { format!(" && !({})", exits.join(" || ")) };
                self.line(&format!("}} while (tb_c{i}{exit} && ++tb_guard < 65536u);"));
                self.loop_depth = self.loop_depth.saturating_sub(1);
            }
        }
        if open {
            self.line("}");
        }
        Ok(())
    }

    /// General control flow: a `while` + `switch` dispatcher over basic blocks, used when
    /// the CFG is not reducible to nested loops. Each block ends by setting the next block
    /// id (`tb_pc`); the program ends when it reaches the id one past the last instruction.
    /// (Implicit-derivative sampling cannot live inside this loop — FXC rejects it, and the
    /// shader then fails to compile rather than silently changing meaning.)
    fn emit_body_dispatch(&mut self, is_ps: bool) -> Result<(), String> {
        let n = self.insns.len();
        let mut off2idx: HashMap<usize, usize> = self.insns.iter().enumerate().map(|(i, x)| (x.offset, i)).collect();
        if let Some(l) = self.insns.last() {
            off2idx.insert(l.offset + l.dwords * 4, n);
        }
        let prog_end = self.insns.last().map_or(0, |l| l.offset + l.dwords * 4);
        let mut leaders: BTreeSet<usize> = BTreeSet::new();
        leaders.insert(0);
        let mut tgt: HashMap<usize, usize> = HashMap::new();
        for (i, ins) in self.insns.iter().enumerate() {
            if ins.enc != Enc::Sopp {
                continue;
            }
            if Self::branch_cond(ins.op).is_some() {
                let t = ins.offset as i64 + 4 + ins.aux.imm * 4;
                let ti = if t >= 0 && t as usize >= prog_end {
                        self.insns.len() // at or beyond the end of the program (s_code_end padding): exit
                    } else {
                        *off2idx
                            .get(&(t as usize))
                            .ok_or_else(|| format!("branch at {:#x} targets mid-instruction {t:#x}", ins.offset))?
                    };
                tgt.insert(i, ti);
                leaders.insert(ti);
                leaders.insert(i + 1);
            } else if ins.op == 1 {
                leaders.insert(i + 1);
            }
        }
        let starts: Vec<usize> = leaders.into_iter().filter(|&s| s < n).collect();
        self.loop_depth = 0;
        self.line("uint tb_pc = 0u; uint tb_guard = 0u;");
        self.line(&format!("[loop] while (tb_pc != {n}u && tb_guard++ < 1000000u) {{"));
        self.loop_depth += 1;
        // Flat `if (tb_sel == K)` tests on a snapshot of the block id (FXC mishandles a
        // `switch` whose selector is assigned inside its cases).
        self.line("uint tb_sel = tb_pc; tb_pc = 0xffffffffu;");
        let insns = self.insns.clone();
        for (b, &start) in starts.iter().enumerate() {
            let end = starts.get(b + 1).copied().unwrap_or(n);
            self.line(&format!("if (tb_sel == {start}u) {{"));
            for i in start..end {
                let ins = &insns[i];
                for o in ins.src.iter().chain(ins.dst.iter()) {
                    self.touch(o);
                }
                let is_ctl = ins.enc == Enc::Sopp && (Self::branch_cond(ins.op).is_some() || ins.op == 1);
                if !is_ctl {
                    self.emit_insn(i, ins, is_ps, None)
                        .map_err(|e| format!("{e} [{} @ {:#x}]", ins.mnemonic, ins.offset))?;
                }
            }
            let last = &insns[end - 1];
            let next = if last.enc == Enc::Sopp && last.op == 1 {
                format!("{n}u")
            } else if let (Some(c), Some(&t)) = (
                if last.enc == Enc::Sopp { Self::branch_cond(last.op) } else { None },
                tgt.get(&(end - 1)),
            ) {
                if last.op == 2 { format!("{t}u") } else { format!("({c}) ? {t}u : {end}u") }
            } else {
                format!("{end}u")
            };
            self.line(&format!("tb_pc = {next}; }}"));
        }
        // No block matched (cannot happen for a well-formed program): stop.
        self.line(&format!("if (tb_pc == 0xffffffffu) tb_pc = {n}u;"));
        self.line("}");
        self.loop_depth = self.loop_depth.saturating_sub(1);
        Ok(())
    }

    /// Float source with VOP3 abs/neg modifiers applied.
    fn fm(&self, ins: &Insn, i: usize) -> Result<String, String> {
        let mut e = f(&ins.src[i])?;
        if ins.enc == Enc::Vop3 {
            if (ins.aux.mask >> i) & 1 != 0 {
                e = format!("abs({e})");
            }
            if (ins.aux.mask >> (3 + i)) & 1 != 0 {
                e = format!("(-{e})");
            }
        }
        Ok(e)
    }

    /// Wrap a float result with VOP3 omod/clamp and convert to uint bits.
    fn fres(&self, ins: &Insn, e: String) -> String {
        let mut e = e;
        if ins.enc == Enc::Vop3 {
            e = match ins.aux.imm2 {
                1 => format!("(({e}) * 2.0)"),
                2 => format!("(({e}) * 4.0)"),
                3 => format!("(({e}) * 0.5)"),
                _ => e,
            };
            if ins.aux.imm != 0 {
                e = format!("saturate({e})");
            }
        }
        format!("asuint({e})")
    }

    fn cmp_expr(m: &str, a: &Opnd, b: &Opnd) -> Result<String, String> {
        // m like "v_cmp_lt_f32" / "s_cmp_eq_i32"
        let parts: Vec<&str> = m.split('_').collect();
        let (cc, ty) = (parts[2], parts[3]);
        let (x, y) = match ty {
            "f32" => (f(a)?, f(b)?),
            "f16" => {
                let h = |o: &Opnd| -> Result<String, String> {
                    Ok(match o {
                        Opnd::ConstF32(_) => f(o)?,
                        _ => format!("f16tof32(({}) & 0xffffu)", r(o)?),
                    })
                };
                (h(a)?, h(b)?)
            }
            "i32" => (format!("asint({})", r(a)?), format!("asint({})", r(b)?)),
            "u32" => (r(a)?, r(b)?),
            other => return Err(format!("compare type {other} unsupported")),
        };
        Ok(match cc {
            "f" => "false".into(),
            "lt" => format!("({x} < {y})"),
            "eq" => format!("({x} == {y})"),
            "le" => format!("({x} <= {y})"),
            "gt" => format!("({x} > {y})"),
            "lg" | "ne" => format!("({x} != {y})"),
            "ge" => format!("({x} >= {y})"),
            "o" => format!("({x} == {x} && {y} == {y})"),
            "u" => format!("({x} != {x} || {y} != {y})"),
            "nge" => format!("(!({x} >= {y}))"),
            "nlg" => format!("(!({x} != {y}))"),
            "ngt" => format!("(!({x} > {y}))"),
            "nle" => format!("(!({x} <= {y}))"),
            "neq" => format!("(!({x} == {y}))"),
            "nlt" => format!("(!({x} < {y}))"),
            "tru" | "t" => "true".into(),
            other => return Err(format!("compare cond {other} unsupported")),
        })
    }

    fn emit_insn(&mut self, idx: usize, ins: &Insn, is_ps: bool, br: Option<usize>) -> Result<(), String> {
        let m = ins.mnemonic.as_str();
        let s = &ins.src;
        let d = ins.dst.first().cloned();
        self.line(&format!("// {:#06x} {m}", ins.offset));
        self.cur_idx = idx;
        if let Some(&id) = self.epilog_pack.get(&idx) {
            // Export epilogue: keep the unpacked operands (see `find_export_epilogs`).
            for (k, sfx) in ["a", "b"].iter().enumerate() {
                let o = *s.get(k).ok_or("pack without two operands")?;
                self.touch(&o);
                self.regs.insert(format!("tb_pk{id}{sfx}"));
                self.line(&format!("tb_pk{id}{sfx} = {};", r(&o)?));
            }
        }
        // Run-time descriptor selectors (only for shaders with a "one of N" image op).
        if let Some((reg, cands)) = self.bsel.remove(&idx) {
            // One guarded copy of the op per candidate buffer.
            self.regs.insert(format!("tb_dsel{reg}"));
            let mut first = true;
            for (id, res) in &cands {
                self.line(&format!("{}if (tb_dsel{reg} == {id:#x}u) {{", if first { "" } else { "else " }));
                first = false;
                self.res.insert(idx, vec![*res]);
                self.emit_insn(idx, ins, is_ps, br)?;
                self.line("}");
            }
            self.bsel.insert(idx, (reg, cands));
            return Ok(());
        }
        if !self.msel.is_empty() || !self.bsel.is_empty() {
            if let Some(&(d, id)) = self.dsel_loads.get(&idx) {
                self.regs.insert(format!("tb_dsel{d}"));
                self.line(&format!("tb_dsel{d} = {id:#x}u;"));
            }
            if let Some(&(d, s)) = self.dsel_moves.get(&idx) {
                self.regs.insert(format!("tb_dsel{d}"));
                self.regs.insert(format!("tb_dsel{s}"));
                self.line(&format!("tb_dsel{d} = tb_dsel{s};"));
            }
        }
        if ins.enc == Enc::Mimg {
            if let Some(sel) = self.msel.get(&idx).cloned() {
                // One guarded copy of the op per (texture, sampler) candidate pair.
                self.regs.insert(format!("tb_dsel{}", sel.tex_reg));
                self.regs.insert(format!("tb_dsel{}", sel.samp_reg));
                let samps: Vec<Option<(u32, Res)>> =
                    if sel.samp.is_empty() { vec![None] } else { sel.samp.iter().copied().map(Some).collect() };
                let mut first = true;
                for (tid, tres) in &sel.tex {
                    for s in &samps {
                        let mut cond = vec![];
                        if sel.tex.len() > 1 {
                            cond.push(format!("tb_dsel{} == {tid:#x}u", sel.tex_reg));
                        }
                        let mut v = vec![*tres];
                        if let Some((sid, sres)) = s {
                            if sel.samp.len() > 1 {
                                cond.push(format!("tb_dsel{} == {sid:#x}u", sel.samp_reg));
                            }
                            v.push(*sres);
                        }
                        self.line(&format!("{}if ({}) {{", if first { "" } else { "else " }, cond.join(" && ")));
                        first = false;
                        self.res.insert(idx, v);
                        self.emit_mimg(idx, ins)?;
                        self.line("}");
                    }
                }
                return Ok(());
            }
        }
        match ins.enc {
            Enc::Sopp if ins.op == 0x10 => self.emit_sendmsg(ins),
            Enc::Sopp if ins.op == 0x0a && self.is_cs => {
                self.line("GroupMemoryBarrierWithGroupSync();");
                Ok(())
            }
            Enc::Sopp => {
                let cond = match ins.op {
                    2 => Some("true".to_string()),
                    4 => Some("scc == 0u".into()),
                    5 => Some("scc != 0u".into()),
                    6 => Some("(vcc_lo & 1u) == 0u".into()),
                    7 => Some("(vcc_lo & 1u) != 0u".into()),
                    8 => Some("(exec_lo & 1u) == 0u".into()),
                    9 => Some("(exec_lo & 1u) != 0u".into()),
                    // A non-final s_endpgm ends the program on this path: jump to the end label.
                    1 if br.is_some() => Some("true".to_string()),
                    0 | 1 | 0x0a | 0x0c | 0x1f | 0x20 | 0x21 | 0x23 | 0x24 | 0x25 => None,
                    _ => return Err("unsupported SOPP".into()),
                };
                if let (Some(c), Some(t)) = (cond, br) {
                    self.line(&format!("if ({c}) j{t} = true;"));
                }
                Ok(())
            }
            Enc::Vop2 | Enc::Vop1 | Enc::Vop3 | Enc::Vopc => self.emit_vec(ins),
            Enc::Sop2 => {
                let d = d.ok_or("no dst")?;
                let (a, b) = (r(&s[0])?, r(&s[1])?);
                if m == "s_bfm_b64" || m == "s_cselect_b64" || m == "s_lshl_b64" {
                    let (lo, hi_e) = if m == "s_lshl_b64" {
                        let ah = r(&hi(&s[0]))?;
                        (
                            format!("((({b} & 63u) >= 32u) ? 0u : ({a} << ({b} & 31u)))"),
                            format!(
                                "((({b} & 63u) == 0u) ? {ah} : ((({b} & 63u) >= 32u) ? ({a} << (({b} & 63u) - 32u)) : (({ah} << ({b} & 31u)) | ({a} >> (32u - ({b} & 31u))))))"
                            ),
                        )
                    } else if m == "s_bfm_b64" {
                        // Bitfield mask ((1 << a) - 1) << b. Lane model keeps the low word;
                        // counts >= 32 saturate it.
                        (format!("((({a} & 63u) >= 32u ? 0xffffffffu : ((1u << ({a} & 31u)) - 1u)) << ({b} & 31u))"), "0u".to_string())
                    } else {
                        (
                            format!("((scc != 0u) ? {a} : {b})"),
                            format!("((scc != 0u) ? {} : {})", r(&hi(&s[0]))?, r(&hi(&s[1]))?),
                        )
                    };
                    let t = self.tmp;
                    self.tmp += 1;
                    self.line(&format!("uint t{t}l = {lo}; uint t{t}h = {hi_e};"));
                    self.ws(&d, &format!("t{t}l"))?;
                    let dh = hi(&d);
                    self.touch(&dh);
                    return self.ws(&dh, &format!("t{t}h"));
                }
                let is64 = m.ends_with("_b64");
                let bit = |op: &str, a: &str, b: &str| -> Option<String> {
                    Some(match op {
                        "and" => format!("({a} & {b})"),
                        "or" => format!("({a} | {b})"),
                        "xor" => format!("({a} ^ {b})"),
                        "andn2" => format!("({a} & ~{b})"),
                        "orn2" => format!("({a} | ~{b})"),
                        "nand" => format!("(~({a} & {b}))"),
                        "nor" => format!("(~({a} | {b}))"),
                        "xnor" => format!("(~({a} ^ {b}))"),
                        _ => return None,
                    })
                };
                let opname = m.trim_start_matches("s_").trim_end_matches("_b32").trim_end_matches("_b64");
                if let Some(lo) = bit(opname, &a, &b) {
                    if is64 {
                        self.touch(&hi(&s[0]));
                        self.touch(&hi(&s[1]));
                        let (ah, bh) = (r(&hi(&s[0]))?, r(&hi(&s[1]))?);
                        let hi_e = bit(opname, &ah, &bh).unwrap();
                        let t = self.tmp;
                        self.tmp += 1;
                        self.line(&format!("uint t{t}l = {lo}; uint t{t}h = {hi_e};"));
                        self.ws(&d, &format!("t{t}l"))?;
                        let dh = hi(&d);
                        self.touch(&dh);
                        self.ws(&dh, &format!("t{t}h"))?;
                        self.line(&format!("scc = ((t{t}l | t{t}h) != 0u) ? 1u : 0u;"));
                    } else {
                        self.ws(&d, &lo)?;
                        self.line(&format!("scc = ({} != 0u) ? 1u : 0u;", name(&d)?));
                    }
                    return Ok(());
                }
                let e = match m {
                    "s_add_u32" | "s_add_i32" => format!("({a} + {b})"),
                    "s_sub_u32" | "s_sub_i32" => format!("({a} - {b})"),
                    "s_addc_u32" => format!("({a} + {b} + (scc & 1u))"),
                    "s_subb_u32" => format!("({a} - {b} - (scc & 1u))"),
                    "s_mul_i32" => format!("({a} * {b})"),
                    "s_mul_hi_u32" => format!("tb_mulhi_u32({a}, {b})"),
                    "s_mul_hi_i32" => format!("tb_mulhi_i32({a}, {b})"),
                    "s_lshl_b32" => format!("({a} << ({b} & 31u))"),
                    "s_lshr_b32" => format!("({a} >> ({b} & 31u))"),
                    "s_ashr_i32" => format!("asuint(asint({a}) >> ({b} & 31u))"),
                    "s_min_u32" => format!("min({a}, {b})"),
                    "s_max_u32" => format!("max({a}, {b})"),
                    "s_min_i32" => format!("asuint(min(asint({a}), asint({b})))"),
                    "s_max_i32" => format!("asuint(max(asint({a}), asint({b})))"),
                    "s_cselect_b32" => format!("((scc != 0u) ? {a} : {b})"),
                    "s_bfe_u32" => format!("(({a} >> ({b} & 31u)) & ((1u << (({b} >> 16) & 31u)) - 1u))"),
                    "s_bfe_i32" => format!("tb_bfe_i32({a}, {b})"),
                    "s_bfm_b32" => format!("(((1u << ({a} & 31u)) - 1u) << ({b} & 31u))"),
                    "s_lshl1_add_u32" => format!("(({a} << 1) + {b})"),
                    "s_lshl2_add_u32" => format!("(({a} << 2) + {b})"),
                    "s_lshl3_add_u32" => format!("(({a} << 3) + {b})"),
                    "s_lshl4_add_u32" => format!("(({a} << 4) + {b})"),
                    _ => return Err("unsupported SOP2".into()),
                };
                let scc = match m {
                    "s_add_u32" => Some(format!("(({a} + {b}) < {a}) ? 1u : 0u")),
                    "s_addc_u32" => Some(format!("((({a} + {b}) < {a}) || (({a} + {b} + (scc & 1u)) < ({a} + {b}))) ? 1u : 0u")),
                    "s_subb_u32" => Some(format!("(({b} > {a}) || ((scc & 1u) > ({a} - {b}))) ? 1u : 0u")),
                    "s_sub_u32" => Some(format!("({b} > {a}) ? 1u : 0u")),
                    "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" | "s_bfe_u32" => Some("__NZ__".into()),
                    _ => None,
                };
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!("uint t{t} = {e};"));
                match scc {
                    Some(x) if x == "__NZ__" => self.line(&format!("scc = (t{t} != 0u) ? 1u : 0u;")),
                    Some(x) => self.line(&format!("scc = {x};")),
                    None => {}
                }
                self.ws(&d, &format!("t{t}"))
            }
            Enc::Sop1 => {
                let d = d.ok_or("no dst")?;
                let a = r(&s[0])?;
                match m {
                    "s_mov_b32" | "s_wqm_b32" => self.ws(&d, &a),
                    // D[S0[4:0]] = 0 / 1
                    "s_bitset0_b32" => {
                        self.touch(&d);
                        let cur = r(&d)?;
                        self.ws(&d, &format!("({cur} & ~(1u << (({a}) & 31u)))"))
                    }
                    "s_bitset1_b32" => {
                        self.touch(&d);
                        let cur = r(&d)?;
                        self.ws(&d, &format!("({cur} | (1u << (({a}) & 31u)))"))
                    }
                    "s_mov_b64" | "s_wqm_b64" => {
                        self.touch(&hi(&s[0]));
                        let ah = r(&hi(&s[0]))?;
                        let t = self.tmp;
                        self.tmp += 1;
                        self.line(&format!("uint t{t}l = {a}; uint t{t}h = {ah};"));
                        self.ws(&d, &format!("t{t}l"))?;
                        let dh = hi(&d);
                        self.touch(&dh);
                        self.ws(&dh, &format!("t{t}h"))
                    }
                    "s_not_b32" => {
                        self.ws(&d, &format!("(~{a})"))?;
                        self.line(&format!("scc = ({} != 0u) ? 1u : 0u;", name(&d)?));
                        Ok(())
                    }
                    "s_cmov_b32" => self.ws(&d, &format!("((scc != 0u) ? {a} : {})", name(&d)?)),
                    "s_brev_b32" => self.ws(&d, &format!("reversebits({a})")),
                    "s_bcnt1_i32_b64" => {
                        // Population count of a 64-bit value — typically the number of active
                        // lanes (wave-level allocation). One invocation = one lane, so a lane
                        // mask counts as 0 or 1.
                        let ah = r(&hi(&s[0]))?;
                        self.ws(&d, &format!("(countbits({a}) + countbits({ah}))"))?;
                        self.line(&format!("scc = ({} != 0u) ? 1u : 0u;", name(&d)?));
                        Ok(())
                    }
                    "s_bcnt1_i32_b32" => {
                        self.ws(&d, &format!("countbits({a})"))?;
                        self.line(&format!("scc = ({} != 0u) ? 1u : 0u;", name(&d)?));
                        Ok(())
                    }
                    "s_ff1_i32_b32" => self.ws(&d, &format!("firstbitlow({a})")),
                    "s_sext_i32_i8" => self.ws(&d, &format!("asuint(asint({a} << 24) >> 24)")),
                    "s_sext_i32_i16" => self.ws(&d, &format!("asuint(asint({a} << 16) >> 16)")),
                    "s_abs_i32" => {
                        self.ws(&d, &format!("asuint(abs(asint({a})))"))?;
                        self.line(&format!("scc = ({} != 0u) ? 1u : 0u;", name(&d)?));
                        Ok(())
                    }
                    "s_getpc_b64" => {
                        // Address of the next instruction, relative to the program start.
                        self.ws(&d, &format!("{:#x}u", ins.offset as u32 + 4))?;
                        let dh = hi(&d);
                        self.touch(&dh);
                        self.ws(&dh, "0u")
                    }
                    "s_swappc_b64" => {
                        // Vertex-fetch shader call: inputs are materialised from the D3D input layout.
                        self.line("// (fetch-shader call: vertex attributes come from the input layout)");
                        Ok(())
                    }
                    _ if m.contains("saveexec") => {
                        let op = m.trim_start_matches("s_").split("_saveexec").next().unwrap_or("");
                        let e = match op {
                            "and" => format!("({a} & exec_lo)"),
                            "or" => format!("({a} | exec_lo)"),
                            "xor" => format!("({a} ^ exec_lo)"),
                            "andn2" => format!("({a} & ~exec_lo)"),
                            "orn2" => format!("({a} | ~exec_lo)"),
                            "andn1" => format!("(~{a} & exec_lo)"),
                            _ => return Err("unsupported saveexec".into()),
                        };
                        let t = self.tmp;
                        self.tmp += 1;
                        // Lane model: only bit 0 is meaningful.
                        self.line(&format!("uint t{t} = {e} & 1u;"));
                        self.ws(&d, "exec_lo")?;
                        if m.ends_with("_b64") {
                            let dh = hi(&d);
                            self.touch(&dh);
                            self.ws(&dh, "0u")?;
                        }
                        self.line(&format!("exec_lo = t{t}; scc = t{t};"));
                        Ok(())
                    }
                    // Return from a vertex shader chained in front of a geometry shader: the
                    // program ends here (it is followed directly by s_endpgm).
                    "s_setpc_b64" if idx + 2 >= self.insns.len() => {
                        self.line("// (return to the chained stage)");
                        Ok(())
                    }
                    _ => Err("unsupported SOP1".into()),
                }
            }
            Enc::Sopk => {
                let d = d.ok_or("no dst")?;
                let imm = format!("{:#x}u", ins.aux.imm as i32 as u32);
                match m {
                    "s_movk_i32" => self.ws(&d, &imm),
                    // Hardware mode-register writes (rounding / denormal control) and the
                    // counter waits have no HLSL-visible effect.
                    "s_setreg_imm32_b32" | "s_setreg_b32" | "s_waitcnt_vscnt" | "s_waitcnt_vmcnt"
                    | "s_waitcnt_expcnt" | "s_waitcnt_lgkmcnt" => Ok(()),
                    "s_addk_i32" => self.ws(&d, &format!("({} + {imm})", name(&d)?)),
                    "s_mulk_i32" => self.ws(&d, &format!("({} * {imm})", name(&d)?)),
                    _ if m.starts_with("s_cmpk_") => {
                        let fake = m.replace("s_cmpk_", "s_cmp_");
                        let c = Self::cmp_expr(&fake, &d, &Opnd::ConstInt(ins.aux.imm))?;
                        self.line(&format!("scc = {c} ? 1u : 0u;"));
                        Ok(())
                    }
                    _ => Err("unsupported SOPK".into()),
                }
            }
            Enc::Sopc => {
                let c = match m {
                    "s_bitcmp0_b32" => format!("((({} >> ({} & 31u)) & 1u) == 0u)", r(&s[0])?, r(&s[1])?),
                    "s_bitcmp1_b32" => format!("((({} >> ({} & 31u)) & 1u) != 0u)", r(&s[0])?, r(&s[1])?),
                    "s_cmp_eq_u64" => format!(
                        "({} == {} && {} == {})",
                        r(&s[0])?,
                        r(&s[1])?,
                        r(&hi(&s[0]))?,
                        r(&hi(&s[1]))?
                    ),
                    "s_cmp_lg_u64" => format!(
                        "({} != {} || {} != {})",
                        r(&s[0])?,
                        r(&s[1])?,
                        r(&hi(&s[0]))?,
                        r(&hi(&s[1]))?
                    ),
                    _ => Self::cmp_expr(m, &s[0], &s[1])?,
                };
                self.line(&format!("scc = {c} ? 1u : 0u;"));
                Ok(())
            }
            Enc::Smem => {
                if ins.op <= 4 {
                    if let Some(Res::CodeTbl) = self.res.get(&idx).and_then(|v| v.first()) {
                        // Constant table embedded in the code section: byte address = pointer
                        // (relative to the start of the program) + immediate.
                        let n = 1u32 << ins.op;
                        let base = sreg_num(&d.ok_or("no dst")?).ok_or("bad dst")?;
                        let ndw = ((self.ex.isa_data.len().saturating_sub(self.ex.isa.len())) / 4).max(1);
                        let t = self.tmp;
                        self.tmp += 1;
                        self.line(&format!("uint t{t} = ({} + {:#x}u) >> 2;", r(&s[0])?, ins.aux.imm as u32));
                        for k in 0..n {
                            self.line(&format!("uint t{t}_{k} = tb_code[(t{t} + {k}u - {}u) % {ndw}u];", self.ex.isa.len() / 4));
                        }
                        for k in 0..n {
                            let dst = sreg_opnd(base + k as u8);
                            self.touch(&dst);
                            self.ws(&dst, &format!("t{t}_{k}"))?;
                        }
                        return Ok(());
                    }
                    self.line("// (descriptor fetch from root table — resolved statically)");
                    return Ok(());
                }
                let n = 1u32 << (ins.op - 8);
                let base = sreg_num(&d.ok_or("no dst")?).ok_or("bad dst")?;
                let soff = ins.raw.get(1).copied().unwrap_or(0) >> 25;
                let slot = match self.res.get(&idx).and_then(|v| v.first()).copied() {
                    Some(Res::Cb(s)) => s,
                    Some(Res::Buf(slot, is_uav)) => {
                        // Structured buffer read by byte offset (see `tb_sbw_*`).
                        let nm = if is_uav { "ubuf" } else { "sbuf" };
                        let dynamic = if soff != 0x7d { format!(" + {}", ssrc_name(soff)?) } else { String::new() };
                        let t = self.tmp;
                        self.tmp += 1;
                        self.line(&format!("uint t{t} = {:#x}u{dynamic};", ins.aux.imm as u32));
                        for k in 0..n {
                            self.line(&format!("uint t{t}_{k} = tb_sbw_{nm}{slot}(t{t} + {}u);", 4 * k));
                        }
                        for k in 0..n {
                            let dst = sreg_opnd(base + k as u8);
                            self.touch(&dst);
                            self.ws(&dst, &format!("t{t}_{k}"))?;
                        }
                        return Ok(());
                    }
                    _ => return Err("unresolved constant buffer".into()),
                };
                if soff != 0x7d {
                    // Dynamic byte offset = imm + SGPR[soffset].
                    let so = ssrc_name(soff)?;
                    let t = self.tmp;
                    self.tmp += 1;
                    self.line(&format!("uint t{t} = ({:#x}u + {so}) >> 2;", ins.aux.imm as u32));
                    // Read all words before writing (the offset register may be a destination).
                    for k in 0..n {
                        self.line(&format!("uint t{t}_{k} = cb{slot}[((t{t} + {k}u) >> 2) & 4095u][(t{t} + {k}u) & 3u];"));
                    }
                    for k in 0..n {
                        let dst = sreg_opnd(base + k as u8);
                        self.touch(&dst);
                        self.ws(&dst, &format!("t{t}_{k}"))?;
                    }
                    return Ok(());
                }
                let w0 = ins.aux.imm as u32 / 4;
                for k in 0..n {
                    let w = w0 + k;
                    let dst = sreg_opnd(base + k as u8);
                    self.touch(&dst);
                    self.ws(&dst, &format!("cb{slot}[{}].{}", w / 4, XYZW[(w % 4) as usize]))?;
                }
                Ok(())
            }
            Enc::Mimg => self.emit_mimg(idx, ins),
            Enc::Mubuf if ins.op == 0x71 || ins.op == 0x72 => {
                self.line("// (cache flush: no-op)");
                Ok(())
            }
            Enc::Mubuf => self.emit_mubuf(idx, ins),
            Enc::Flat if m.starts_with("flat_load_dword") => {
                // Per-lane load through a 64-bit address. Shaders use this for lookups in
                // constant tables embedded in their own code section (address derived from
                // `s_getpc_b64`), so it is only accepted when the program takes its PC.
                if ins.aux.mask == 1 {
                    return Err("scratch-memory load unsupported".into());
                }
                if !self.insns.iter().any(|i| i.mnemonic == "s_getpc_b64") {
                    return Err("global/flat load in a shader that never takes its PC".into());
                }
                self.uses_code = true;
                let n: u8 = match m {
                    "flat_load_dword" => 1,
                    "flat_load_dwordx2" => 2,
                    "flat_load_dwordx3" => 3,
                    _ => 4,
                };
                let extra = match &s[2] {
                    Opnd::Null => String::new(),
                    o => format!(" + {}", r(o)?),
                };
                let ndw = ((self.ex.isa_data.len().saturating_sub(self.ex.isa.len())) / 4).max(1);
                let vd = match d {
                    Some(Opnd::Vgpr(v)) => v,
                    _ => return Err("bad vdst".into()),
                };
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!("uint t{t} = ({} + {:#x}u{extra}) >> 2;", r(&s[0])?, ins.aux.imm as i32 as u32));
                for k in 0..n {
                    self.line(&format!("uint t{t}_{k} = tb_code[(t{t} + {k}u - {}u) % {ndw}u];", self.ex.isa.len() / 4));
                }
                for k in 0..n {
                    self.wv(&Opnd::Vgpr(vd + k), &format!("t{t}_{k}"))?;
                }
                Ok(())
            }
            Enc::Ds
                if !self.is_gs
                    && !self.is_cs
                    && self.ex.stage == Stage::Vertex
                    && matches!(m, "ds_write_b32" | "ds_write2_b32" | "ds_write_b64" | "ds_write2_b64") =>
            {
                // A vertex shader chained in front of a geometry shader hands its outputs over
                // through LDS instead of parameter exports: the vertex record is the output
                // registers back to back, 4 dwords each (observed in a55400ccab602283: dwords
                // 0..3 = TEXCOORD6.xyzw, 4..6 = TEXCOORD8.xyz, matching the output signature).
                // The address register is this vertex's record base and is not needed.
                let off = ins.aux.imm as u32;
                let vg = |o: &Opnd, k: u8| match o {
                    Opnd::Vgpr(n) => Ok(Opnd::Vgpr(n + k)),
                    _ => Err("LDS output write with a non-VGPR source".to_string()),
                };
                // (byte offset, source register) per written dword.
                let writes: Vec<(u32, Opnd)> = match m {
                    "ds_write_b32" => vec![(off & 0xffff, s[1])],
                    "ds_write2_b32" => vec![((off & 0xff) * 4, s[1]), (((off >> 8) & 0xff) * 4, s[2])],
                    "ds_write_b64" => vec![(off & 0xffff, s[1]), ((off & 0xffff) + 4, vg(&s[1], 1)?)],
                    _ => vec![
                        ((off & 0xff) * 8, s[1]),
                        ((off & 0xff) * 8 + 4, vg(&s[1], 1)?),
                        (((off >> 8) & 0xff) * 8, s[2]),
                        (((off >> 8) & 0xff) * 8 + 4, vg(&s[2], 1)?),
                    ],
                };
                let osig = self.ex.output_sig.clone();
                for (byte, src) in writes {
                    let dw = (byte / 4) as usize;
                    // The record is indexed by output *register* (4 dwords each), system
                    // values included: in 008bccdcf9dcecf9 register 1 is SV_Position and is
                    // written at dwords 4..7.
                    let reg = (dw / 4) as u32;
                    let comp = dw % 4;
                    self.touch(&src);
                    let mut hit = false;
                    for (ei, e) in osig.iter().enumerate() {
                        if e.register == reg && (e.mask >> comp) & 1 != 0 {
                            let idx = (0..comp).filter(|c| (e.mask >> c) & 1 != 0).count();
                            let (ty, n) = sig_type(e);
                            let field = if n == 1 { format!("o.o{ei}") } else { format!("o.o{ei}.{}", XYZW[idx]) };
                            let val = if ty.starts_with("float") { f(&src)? } else if ty.starts_with("int") { format!("asint({})", r(&src)?) } else { r(&src)? };
                            self.line(&format!("if (exec_lo & 1u) {field} = {val};"));
                            hit = true;
                        }
                    }
                    if !hit {
                        self.line(&format!("// (LDS output dword {dw} has no output-signature component: ignored)"));
                    }
                }
                Ok(())
            }
            Enc::Ds if self.is_gs && matches!(m, "ds_read_b32" | "ds_read2_b32" | "ds_write_b32" | "ds_write2_b32") => {
                self.emit_gs_lds(ins)
            }
            Enc::Ds
                if self.is_cs
                    && matches!(
                        m,
                        "ds_read_b32" | "ds_read2_b32" | "ds_read2st64_b32" | "ds_write_b32" | "ds_write2_b32" | "ds_write2st64_b32"
                    ) =>
            {
                // Compute: LDS is the thread group's shared memory → an HLSL `groupshared`
                // array addressed by byte address / 4 (32 KiB, the D3D groupshared limit).
                self.uses_lds = true;
                let off = ins.aux.imm as u32;
                // Two-address forms scale their 8-bit offsets by 4 bytes ("2") or by
                // 64 dwords = 256 bytes ("2st64"); express both in units of 4 bytes.
                let scale = if m.contains("st64") { 64 } else { 1 };
                let (off0, off1) = ((off & 0xff) * scale, ((off >> 8) & 0xff) * scale);
                let addr = r(&s[0])?;
                let at = |o: u32| format!("tb_lds[(({addr} + {o:#x}u) >> 2) & 8191u]");
                let vd = match d {
                    Some(Opnd::Vgpr(v)) => v,
                    _ => return Err("bad vdst".into()),
                };
                let t = self.tmp;
                self.tmp += 1;
                match m {
                    "ds_read_b32" => {
                        self.line(&format!("uint t{t} = {};", at(off & 0xffff)));
                        self.wv(&Opnd::Vgpr(vd), &format!("t{t}"))
                    }
                    "ds_read2_b32" | "ds_read2st64_b32" => {
                        self.line(&format!("uint t{t}a = {}; uint t{t}b = {};", at(off0 * 4), at(off1 * 4)));
                        self.wv(&Opnd::Vgpr(vd), &format!("t{t}a"))?;
                        self.wv(&Opnd::Vgpr(vd + 1), &format!("t{t}b"))
                    }
                    "ds_write_b32" => {
                        self.line(&format!("if (exec_lo & 1u) {} = {};", at(off & 0xffff), r(&s[1])?));
                        Ok(())
                    }
                    _ => {
                        self.line(&format!(
                            "if (exec_lo & 1u) {{ {} = {}; {} = {}; }}",
                            at(off0 * 4),
                            r(&s[1])?,
                            at(off1 * 4),
                            r(&s[2])?
                        ));
                        Ok(())
                    }
                }
            }
            Enc::Ds if self.is_cs && matches!(m, "ds_write_b16" | "ds_write_b8" | "ds_read_u16" | "ds_read_i16" | "ds_read_u8" | "ds_read_i8") => {
                // Sub-word accesses to group-shared memory: tb_lds is addressed in dwords, so
                // read-modify-write the containing dword.
                self.uses_lds = true;
                let off = ins.aux.imm as u32 & 0xffff;
                let bits = if m.ends_with("16") { 16u32 } else { 8 };
                let mask = (1u32 << bits) - 1;
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!("uint t{t}a = ({}) + {off:#x}u; uint t{t}s = (t{t}a & 3u) * 8u;", r(&s[0])?));
                let word = format!("tb_lds[(t{t}a >> 2) & 8191u]");
                if m.starts_with("ds_write") {
                    self.line(&format!(
                        "if (exec_lo & 1u) {word} = ({word} & ~({mask:#x}u << t{t}s)) | ((({}) & {mask:#x}u) << t{t}s);",
                        r(&s[1])?
                    ));
                    return Ok(());
                }
                let v = format!("(({word} >> t{t}s) & {mask:#x}u)");
                let e = if m.contains("_i") { format!("asuint(asint({v} << {}) >> {})", 32 - bits, 32 - bits) } else { v };
                return self.wv(&d.ok_or("no dst")?, &e);
            }
            Enc::Ds
                if self.is_cs
                    && matches!(
                        m.replace("_rtn", "").as_str(),
                        "ds_add_u32" | "ds_min_u32" | "ds_max_u32" | "ds_and_b32" | "ds_or_b32" | "ds_xor_b32"
                    ) =>
            {
                // Atomic read-modify-write on group-shared memory (the `_rtn` forms also
                // return the old value) → the matching HLSL Interlocked* intrinsic.
                self.uses_lds = true;
                let func = match m.replace("_rtn", "").as_str() {
                    "ds_add_u32" => "InterlockedAdd",
                    "ds_min_u32" => "InterlockedMin",
                    "ds_max_u32" => "InterlockedMax",
                    "ds_and_b32" => "InterlockedAnd",
                    "ds_or_b32" => "InterlockedOr",
                    _ => "InterlockedXor",
                };
                let off = ins.aux.imm as u32 & 0xffff;
                let addr = r(&s[0])?;
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!(
                    "uint t{t} = 0u; if (exec_lo & 1u) {{ {func}(tb_lds[(({addr} + {off:#x}u) >> 2) & 8191u], {}, t{t}); }}",
                    r(&s[1])?
                ));
                if m.contains("_rtn") {
                    self.wv(&d.ok_or("no dst")?, &format!("t{t}"))?;
                }
                Ok(())
            }
            Enc::Ds if m == "ds_swizzle_b32" => {
                // Quad swizzle (QDMode): each lane of a 2x2 pixel quad reads the lane selected
                // by a 2-bit field of the 8-bit mask — how RDNA2 pixel shaders fetch neighbour
                // values for derivatives. Per pixel: value at quad position (sx, sy) =
                // v + (sx - px) * ddx_fine(v) + (sy - py) * ddy_fine(v), with (px, py) this
                // pixel's position in its quad. Treats the register as float (derivative use).
                if !is_ps {
                    return Err("ds_swizzle outside a pixel shader".into());
                }
                let off = ins.aux.imm as u32;
                if off & 0x8000 == 0 {
                    return Err("ds_swizzle without QDMode unsupported".into());
                }
                self.uses_quad = true;
                let src = f(&s[0])?;
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!("float t{t} = {src}; uint t{t}s = ({:#x}u >> (tb_quad * 2u)) & 3u;", off & 0xff));
                self.wv(
                    &d.ok_or("no dst")?,
                    &format!(
                        "asuint(t{t} + ((float)(t{t}s & 1u) - (float)(tb_quad & 1u)) * ddx_fine(t{t}) + ((float)(t{t}s >> 1) - (float)(tb_quad >> 1)) * ddy_fine(t{t}))"
                    ),
                )
            }
            Enc::Vinterp => {
                if !is_ps {
                    return Err("v_interp outside a pixel shader".into());
                }
                let attr = ins.aux.imm as usize;
                let chan = ins.aux.imm2 as usize;
                let reg = *self.ps_attrs.get(attr).ok_or_else(|| format!("PS attr{attr} has no signature register"))?;
                let val = if let Some((ei, k)) = sig_lookup(&self.ex.input_sig, reg, chan) {
                    let acc = format!("i.i{ei}.{}", XYZW[k]);
                    if self.ex.input_sig[ei].comp_type == 3 { format!("asuint({acc})") } else { format!("(uint)({acc})") }
                } else if chan == 3 {
                    "0x3f800000u".to_string()
                } else {
                    "0u".to_string()
                };
                // p1 computes a partial term in hardware; the full interpolated value is
                // already available per-pixel here, so both steps assign it.
                self.wv(&d.ok_or("no dst")?, &val)
            }
            Enc::Exp => self.emit_exp(ins, is_ps),
            _ => Err("unsupported encoding".into()),
        }
    }

    fn emit_vec(&mut self, ins: &Insn) -> Result<(), String> {
        let mut m = ins.mnemonic.clone();
        let s = &ins.src;
        let d = ins.dst.first().cloned().ok_or("no dst")?;
        if m == "v_nop" {
            return Ok(());
        }
        if ins.op == OP_WAVE_REDUCE && ins.enc == Enc::Vop3 {
            // See `lower_wave_reductions`. Every invocation computes the same value; lanes
            // that were inactive contribute the identity.
            self.uses_wave = true;
            let f = if m == "tb_wave_min_f32" { "WaveActiveMin" } else { "WaveActiveMax" };
            let e = format!(
                "asuint({f}((({} & 1u) != 0u) ? asfloat({}) : asfloat({:#x}u)))",
                r(&s[1])?,
                r(&s[0])?,
                ins.aux.imm as u32
            );
            return self.ws(&d, &e);
        }
        // ---- half-float arithmetic: operands are f16 in the low 16 bits (inline float
        // constants denote the half-float of that value); the result is written as f16.
        // VOP3 forms carry per-source abs/neg modifiers and a result clamp/output modifier,
        // exactly as for f32 (see `fm` / `fres`).
        let h16 = |o: &Opnd| -> Result<String, String> {
            let mut e = match o {
                Opnd::ConstF32(_) => f(o)?,
                _ => format!("f16tof32(({}) & 0xffffu)", r(o)?),
            };
            if ins.enc == Enc::Vop3 {
                if let Some(i) = s.iter().position(|x| std::ptr::eq(x, o)) {
                    if (ins.aux.mask >> i) & 1 != 0 {
                        e = format!("abs({e})");
                    }
                    if (ins.aux.mask >> (3 + i)) & 1 != 0 {
                        e = format!("(-{e})");
                    }
                }
            }
            Ok(e)
        };
        let hres = |e: String| -> String {
            let mut e = e;
            if ins.enc == Enc::Vop3 {
                e = match ins.aux.imm2 {
                    1 => format!("(({e}) * 2.0)"),
                    2 => format!("(({e}) * 4.0)"),
                    3 => format!("(({e}) * 0.5)"),
                    _ => e,
                };
                if ins.aux.imm != 0 {
                    e = format!("saturate({e})");
                }
            }
            format!("f32tof16({e})")
        };
        if matches!(m.as_str(), "v_add_f16" | "v_sub_f16" | "v_subrev_f16" | "v_mul_f16" | "v_max_f16" | "v_min_f16") {
            let (a, b) = (h16(&s[0])?, h16(&s[1])?);
            let e = match m.as_str() {
                "v_add_f16" => format!("{a} + {b}"),
                "v_sub_f16" => format!("{a} - {b}"),
                "v_subrev_f16" => format!("{b} - {a}"),
                "v_mul_f16" => format!("{a} * {b}"),
                "v_max_f16" => format!("max({a}, {b})"),
                _ => format!("min({a}, {b})"),
            };
            return self.wv(&d, &hres(e));
        }
        if matches!(m.as_str(), "v_rcp_f16" | "v_sqrt_f16" | "v_rsq_f16" | "v_floor_f16" | "v_ceil_f16" | "v_trunc_f16" | "v_rndne_f16" | "v_fract_f16") {
            let a = h16(&s[0])?;
            let e = match m.as_str() {
                "v_rcp_f16" => format!("1.0 / {a}"),
                "v_sqrt_f16" => format!("sqrt({a})"),
                "v_rsq_f16" => format!("rsqrt({a})"),
                "v_floor_f16" => format!("floor({a})"),
                "v_ceil_f16" => format!("ceil({a})"),
                "v_trunc_f16" => format!("trunc({a})"),
                "v_rndne_f16" => format!("round({a})"),
                _ => format!("frac({a})"),
            };
            return self.wv(&d, &hres(e));
        }
        if m == "v_cvt_f16_u16" {
            return self.wv(&d, &format!("f32tof16((float)(({}) & 0xffffu))", r(&s[0])?));
        }
        if m == "v_cvt_f16_i16" {
            return self.wv(&d, &format!("f32tof16((float)(asint(({}) << 16) >> 16))", r(&s[0])?));
        }
        if m == "v_cvt_u16_f16" {
            return self.wv(&d, &format!("((uint)clamp({}, 0.0, 65535.0))", h16(&s[0])?));
        }
        if m == "v_cvt_i16_f16" {
            return self.wv(&d, &format!("(asuint((int)clamp({}, -32768.0, 32767.0)) & 0xffffu)", h16(&s[0])?));
        }
        if matches!(m.as_str(), "v_madak_f16" | "v_madmk_f16") {
            // The trailing literal dword carries the half-float constant K in its low bits.
            let k = format!("f16tof32({:#x}u)", ins.raw.last().copied().unwrap_or(0) & 0xffff);
            let (a, b) = (h16(&s[0])?, h16(&s[1])?);
            let e = if m == "v_madak_f16" { format!("{a} * {b} + {k}") } else { format!("{a} * {k} + {b}") };
            return self.wv(&d, &hres(e));
        }
        if m == "v_mac_f16" {
            self.touch(&d);
            let e = format!("{} * {} + {}", h16(&s[0])?, h16(&s[1])?, h16(&d)?);
            return self.wv(&d, &hres(e));
        }
        if m == "v_fma_f16" {
            let e = format!("{} * {} + {}", h16(&s[0])?, h16(&s[1])?, h16(&s[2])?);
            return self.wv(&d, &hres(e));
        }
        if matches!(m.as_str(), "v_pk_fma_f16" | "v_pk_add_f16" | "v_pk_mul_f16" | "v_pk_min_f16" | "v_pk_max_f16") {
            // Packed half-float ops (VOP3P): two f16 lanes per register. For the low result,
            // OPSEL bit i (dword0[13:11]) picks the low/high half of source i and NEG bit i
            // (dword1[31:29]) negates it; for the high result OPSEL_HI (dword1[28:27],
            // dword0[14]) and NEG_HI (dword0[10:8]) do the same.
            let (w0, w1) = (ins.raw[0], ins.raw.get(1).copied().unwrap_or(0));
            let nsrc = if m == "v_pk_fma_f16" { 3 } else { 2 };
            let sel = [(w0 >> 11) & 7, ((w1 >> 27) & 3) | (((w0 >> 14) & 1) << 2)];
            let neg = [(w1 >> 29) & 7, (w0 >> 8) & 7];
            let mut halves = Vec::new();
            for h in 0..2 {
                let mut v = Vec::new();
                for i in 0..nsrc {
                    let x = match &s[i] {
                        o @ Opnd::ConstF32(_) => format!("f32tof16({})", f(o)?),
                        o => r(o)?,
                    };
                    let mut e = if (sel[h] >> i) & 1 != 0 { format!("f16tof32(({x}) >> 16)") } else { format!("f16tof32(({x}) & 0xffffu)") };
                    if (neg[h] >> i) & 1 != 0 {
                        e = format!("(-{e})");
                    }
                    v.push(e);
                }
                halves.push(match m.as_str() {
                    "v_pk_fma_f16" => format!("{} * {} + {}", v[0], v[1], v[2]),
                    "v_pk_add_f16" => format!("{} + {}", v[0], v[1]),
                    "v_pk_mul_f16" => format!("{} * {}", v[0], v[1]),
                    "v_pk_min_f16" => format!("min({}, {})", v[0], v[1]),
                    _ => format!("max({}, {})", v[0], v[1]),
                });
            }
            return self.wv(&d, &format!("(f32tof16({}) | (f32tof16({}) << 16))", halves[0], halves[1]));
        }
        if matches!(m.as_str(), "v_min3_f16" | "v_max3_f16" | "v_med3_f16") {
            let (a, b, c) = (h16(&s[0])?, h16(&s[1])?, h16(&s[2])?);
            let e = match m.as_str() {
                "v_min3_f16" => format!("min(min({a}, {b}), {c})"),
                "v_max3_f16" => format!("max(max({a}, {b}), {c})"),
                _ => format!("max(min({a}, {b}), min(max({a}, {b}), {c}))"),
            };
            return self.wv(&d, &hres(e));
        }
        if m == "v_bcnt_u32_b32" {
            return self.wv(&d, &format!("(countbits({}) + {})", r(&s[0])?, r(&s[1])?));
        }
        if m == "v_bfm_b32" {
            return self.wv(&d, &format!("(((1u << (({}) & 31u)) - 1u) << (({}) & 31u))", r(&s[0])?, r(&s[1])?));
        }
        if m == "v_perm_b32" {
            return self.wv(&d, &format!("tb_perm({}, {}, {})", r(&s[0])?, r(&s[1])?, r(&s[2])?));
        }
        if m == "v_bfe_i32" {
            // Sign-extended bit field: offset = S1[4:0], width = S2[4:0].
            let (a, o, w) = (r(&s[0])?, r(&s[1])?, r(&s[2])?);
            return self.wv(&d, &format!(
                "((({w}) & 31u) == 0u ? 0u : asuint(asint(({a}) << ((32u - (({w}) & 31u) - (({o}) & 31u)) & 31u)) >> ((32u - (({w}) & 31u)) & 31u)))"));
        }
        if matches!(m.as_str(), "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32") {
            // Register-relative moves: the source and/or destination VGPR number is offset by
            // M0 at run time (local arrays kept in registers). Lowered to a chain over every
            // VGPR the program uses.
            let (Opnd::Vgpr(dn), Opnd::Vgpr(sn)) = (d, s[0]) else {
                return Err("register-relative move with a non-VGPR operand".into());
            };
            if !self.vreg_array {
                return Err("register-relative move outside VGPR-array mode".into());
            }
            let t = self.tmp;
            self.tmp += 1;
            let src = if m == "v_movreld_b32" { format!("tb_v[{sn}]") } else { format!("tb_v[({sn}u + m0) & 255u]") };
            self.line(&format!("uint t{t} = {src};"));
            let dst = if m == "v_movrels_b32" { format!("tb_v[{dn}]") } else { format!("tb_v[({dn}u + m0) & 255u]") };
            self.line(&format!("if (exec_lo & 1u) {dst} = t{t};"));
            return Ok(());
        }
        if matches!(m.as_str(), "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16") {
            // VOP3P: D = S0 * S1 + S2 where each source is either a full float or a half
            // float taken from the low/high 16 bits of its register and widened:
            //   OPSEL_HI bit i (dword1[28:27], dword0[14] for i = 2) = 0 -> 32-bit float,
            //                                                         1 -> 16-bit float;
            //   OPSEL bit i (dword0[13:11]) picks the high half when the source is 16-bit;
            //   NEG bit i (dword1[31:29]) negates, NEG_HI bit i (dword0[10:8]) takes |x|.
            let (w0, w1) = (ins.raw[0], ins.raw.get(1).copied().unwrap_or(0));
            let opsel = (w0 >> 11) & 7;
            let opsel_hi = ((w1 >> 27) & 3) | (((w0 >> 14) & 1) << 2);
            let (neg, abs) = ((w1 >> 29) & 7, (w0 >> 8) & 7);
            let mut v = Vec::new();
            for i in 0..3 {
                let x = r(&s[i])?;
                let mut e = if (opsel_hi >> i) & 1 != 0 {
                    if (opsel >> i) & 1 != 0 { format!("f16tof32(({x}) >> 16)") } else { format!("f16tof32(({x}) & 0xffffu)") }
                } else {
                    format!("asfloat({x})")
                };
                if (abs >> i) & 1 != 0 {
                    e = format!("abs({e})");
                }
                if (neg >> i) & 1 != 0 {
                    e = format!("(-{e})");
                }
                v.push(e);
            }
            let e = format!("{} * {} + {}", v[0], v[1], v[2]);
            return match m.as_str() {
                "v_fma_mix_f32" => self.wv(&d, &format!("asuint({e})")),
                // The f16 result replaces one half of the destination; the other half is kept.
                "v_fma_mixlo_f16" => {
                    self.touch(&d);
                    self.wv(&d, &format!("(({}) & 0xffff0000u) | f32tof16({e})", r(&d)?))
                }
                _ => {
                    self.touch(&d);
                    self.wv(&d, &format!("(({}) & 0xffffu) | (f32tof16({e}) << 16)", r(&d)?))
                }
            };
        }
        // Compares (VOPC, and VOP3 with a scalar mask destination).
        // v_cmpx: same compare, destination already decoded as EXEC.
        if m.starts_with("v_cmp_") || m.starts_with("v_cmpx_") {
            let c = if m.ends_with("_class_f32") {
                // Floating-point class test against a 10-bit class mask (see tb_cmp_class).
                format!("tb_cmp_class({}, {})", r(&s[0])?, r(&s[1])?)
            } else {
                Self::cmp_expr(&m, &s[0], &s[1])?
            };
            self.ws(&d, &format!("(((exec_lo & 1u) != 0u) && {c}) ? 1u : 0u"))?;
            let dh = hi(&d);
            if !matches!(dh, Opnd::Null | Opnd::ConstInt(_)) {
                self.touch(&dh);
                self.ws(&dh, "0u")?;
            }
            return Ok(());
        }
        if matches!(
            m.as_str(),
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32" | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32"
        ) {
            // Add/subtract with carry. VOP3B: carry-out register in bits [14:8] of dword0 and
            // carry-in is src2; VOP2: both are VCC. Lane model: carry = this lane's bit.
            let vop3f = ins.enc == Enc::Vop3;
            let cout = if vop3f {
                match (ins.raw[0] >> 8) & 0x7f {
                    c @ 0..=105 => Opnd::Sgpr(c as u8),
                    106 => Opnd::VccLo,
                    c => return Err(format!("carry-out register code {c} unsupported")),
                }
            } else {
                Opnd::VccLo
            };
            let cin = if m.contains("_ci_") { if vop3f { r(&s[2])? } else { "vcc_lo".to_string() } } else { "0u".to_string() };
            let (mut a, mut b) = (r(&s[0])?, r(&s[1])?);
            if m.starts_with("v_subrev") {
                core::mem::swap(&mut a, &mut b);
            }
            let t = self.tmp;
            self.tmp += 1;
            if m.starts_with("v_add") {
                self.line(&format!(
                    "uint t{t}c = ({cin}) & 1u; uint t{t}a = {a} + {b}; uint t{t} = t{t}a + t{t}c; uint t{t}o = ((t{t}a < {a}) || (t{t} < t{t}a)) ? 1u : 0u;"
                ));
            } else {
                self.line(&format!(
                    "uint t{t}c = ({cin}) & 1u; uint t{t}a = {a} - {b}; uint t{t} = t{t}a - t{t}c; uint t{t}o = (({b} > {a}) || (t{t}c > t{t}a)) ? 1u : 0u;"
                ));
            }
            self.wv(&d, &format!("t{t}"))?;
            self.touch(&cout);
            self.ws(&cout, &format!("((exec_lo & 1u) != 0u) ? t{t}o : 0u"))?;
            let ch = hi(&cout);
            self.touch(&ch);
            self.ws(&ch, "0u")?;
            return Ok(());
        }
        if let Some(sd) = ins.aux.sdwa {
            // SDWA: operate on a selected byte/word of each source. Materialise the selected
            // values into temporaries, then translate the instruction as usual.
            let is_cmp = ins.enc == Enc::Vopc;
            let dst_sel = if is_cmp { 6 } else { (sd >> 8) & 7 };
            if dst_sel == 7 {
                return Err("SDWA destination select 7 is reserved".into());
            }
            if sd & 0x3030_0000 != 0 {
                return Err("SDWA float neg/abs modifiers are unsupported".into());
            }
            let mut c = ins.clone();
            c.aux.sdwa = None;
            for (i, shift) in [(0usize, 16u32), (1usize, 24u32)] {
                if i >= c.src.len() {
                    continue;
                }
                let sel = (sd >> shift) & 7;
                let sext = (sd >> (shift + 3)) & 1 != 0;
                if sel == 6 {
                    continue;
                }
                let x = r(&c.src[i])?;
                let (bits, lsb) = match sel {
                    0..=3 => (8u32, sel * 8),
                    4 | 5 => (16, (sel - 4) * 16),
                    _ => return Err(format!("SDWA select {sel} is reserved")),
                };
                let e = if sext {
                    format!("asuint(asint({x} << {}) >> {})", 32 - bits - lsb, 32 - bits)
                } else {
                    format!("(({x} >> {lsb}) & {:#x}u)", (1u32 << bits) - 1)
                };
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!("uint tb_s{t} = {e};"));
                c.src[i] = Opnd::Temp(t as u16);
            }
            if dst_sel == 6 {
                return self.emit_vec(&c);
            }
            // Partial destination: compute into a temporary, then place the selected
            // byte/word into the register. DST_UNUSED (bits 12:11): 0 = pad with zeros,
            // 1 = sign-extend upwards, 2 = keep the other bits of the destination.
            let (bits, lsb) = if dst_sel <= 3 { (8u32, dst_sel * 8) } else { (16, (dst_sel - 4) * 16) };
            let mask = (1u32 << bits) - 1;
            let t = self.tmp;
            self.tmp += 1;
            self.line(&format!("uint tb_s{t} = 0u;"));
            c.dst = vec![Opnd::Temp(t as u16)];
            self.emit_vec(&c)?;
            let val = format!("(tb_s{t} & {mask:#x}u)");
            let e = match (sd >> 11) & 3 {
                0 => format!("({val} << {lsb})"),
                1 => format!("asuint(asint({val} << {}) >> {})", 32 - bits, 32 - bits - lsb),
                2 => {
                    self.touch(&d);
                    format!("((({}) & {:#x}u) | ({val} << {lsb}))", r(&d)?, !(mask << lsb))
                }
                _ => return Err("SDWA DST_UNUSED 3 is reserved".into()),
            };
            return self.wv(&d, &e);
        }
        if matches!(m.as_str(), "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64") {
            // 64-bit vector shifts: value and result are register pairs (lo, hi).
            let pair_hi = |o: &Opnd| match o {
                Opnd::Vgpr(n) => Opnd::Vgpr(n + 1),
                other => hi(other),
            };
            let n = format!("({} & 63u)", r(&s[0])?);
            let (lo, hi_e) = (r(&s[1])?, r(&pair_hi(&s[1]))?);
            let t = self.tmp;
            self.tmp += 1;
            let (rl, rh) = match m.as_str() {
                "v_lshlrev_b64" => (
                    format!("(({n} >= 32u) ? 0u : ({lo} << ({n} & 31u)))"),
                    format!("(({n} == 0u) ? {hi_e} : (({n} >= 32u) ? ({lo} << ({n} - 32u)) : (({hi_e} << ({n} & 31u)) | ({lo} >> (32u - ({n} & 31u))))))"),
                ),
                "v_lshrrev_b64" => (
                    format!("(({n} == 0u) ? {lo} : (({n} >= 32u) ? ({hi_e} >> ({n} - 32u)) : (({lo} >> ({n} & 31u)) | ({hi_e} << (32u - ({n} & 31u))))))"),
                    format!("(({n} >= 32u) ? 0u : ({hi_e} >> ({n} & 31u)))"),
                ),
                _ => (
                    format!("(({n} == 0u) ? {lo} : (({n} >= 32u) ? asuint(asint({hi_e}) >> ({n} - 32u)) : (({lo} >> ({n} & 31u)) | ({hi_e} << (32u - ({n} & 31u))))))"),
                    format!("(({n} >= 32u) ? asuint(asint({hi_e}) >> 31) : asuint(asint({hi_e}) >> ({n} & 31u)))"),
                ),
            };
            self.line(&format!("uint t{t}l = {rl}; uint t{t}h = {rh};"));
            self.wv(&d, &format!("t{t}l"))?;
            let dh = pair_hi(&d);
            self.touch(&dh);
            return self.wv(&dh, &format!("t{t}h"));
        }
        if m == "v_swap_b32" {
            let (x, y) = (name(&d)?, name(&s[0])?);
            let t = self.tmp;
            self.tmp += 1;
            self.line(&format!("if (exec_lo & 1u) {{ uint t{t} = {x}; {x} = {y}; {y} = t{t}; }}"));
            return Ok(());
        }
        if m == "v_readfirstlane_b32" {
            // Lane model: the first active lane's value. Compilers only use this on values
            // that are uniform across the wave, so this lane's value is equivalent.
            return self.ws(&d, &r(&s[0])?);
        }
        let vop3 = ins.enc == Enc::Vop3;
        let k = format!("asfloat({:#x}u)", ins.raw.last().copied().unwrap_or(0));
        let dn = name(&d)?;
        let fa = || self.fm(ins, 0);
        let fb = || self.fm(ins, 1);
        let fc = || self.fm(ins, 2);
        let (a, b) = (r(&s[0])?, s.get(1).map(r).transpose()?.unwrap_or_default());
        let c = s.get(2).map(|o| r(o)).transpose().unwrap_or(None).unwrap_or_default();
        // `v_mul_legacy_f32` / `v_mad_legacy_f32`: DX9 multiply rules — the product is 0 when
        // either operand is 0, even against infinity or NaN (`tb_mull`). A plain multiply
        // only matched under fast-math; with exact arithmetic it produced NaN positions.
        let legacy = m.ends_with("_legacy_f32");
        if m.ends_with("_legacy_f32") {
            m = m.replace("_legacy", "");
        }
        let fexpr: Option<String> = match m.as_str() {
            "v_add_f32" => Some(format!("{} + {}", fa()?, fb()?)),
            "v_sub_f32" => Some(format!("{} - {}", fa()?, fb()?)),
            "v_subrev_f32" => Some(format!("{} - {}", fb()?, fa()?)),
            "v_mul_f32" if legacy => Some(format!("tb_mull({}, {})", fa()?, fb()?)),
            "v_mul_f32" => Some(format!("{} * {}", fa()?, fb()?)),
            "v_min_f32" => Some(format!("min({}, {})", fa()?, fb()?)),
            "v_max_f32" => Some(format!("max({}, {})", fa()?, fb()?)),
            "v_mac_f32" | "v_fmac_f32" => Some(format!("{} * {} + asfloat({dn})", fa()?, fb()?)),
            "v_madmk_f32" | "v_fmamk_f32" => Some(format!("{} * {k} + {}", fa()?, fb()?)),
            "v_madak_f32" | "v_fmaak_f32" => Some(format!("{} * {} + {k}", fa()?, fb()?)),
            "v_mad_f32" if legacy => Some(format!("tb_mull({}, {}) + {}", fa()?, fb()?, fc()?)),
            "v_mad_f32" | "v_fma_f32" => Some(format!("{} * {} + {}", fa()?, fb()?, fc()?)),
            "v_min3_f32" => Some(format!("min(min({}, {}), {})", fa()?, fb()?, fc()?)),
            "v_max3_f32" => Some(format!("max(max({}, {}), {})", fa()?, fb()?, fc()?)),
            "v_med3_f32" => Some(format!(
                "max(min({0}, {1}), min(max({0}, {1}), {2}))",
                fa()?,
                fb()?,
                fc()?
            )),
            "v_cvt_f32_i32" => Some(format!("(float)asint({a})")),
            "v_cvt_f32_u32" => Some(format!("(float)({a})")),
            "v_cvt_f32_ubyte0" => Some(format!("(float)({a} & 0xffu)")),
            "v_cvt_f32_ubyte1" => Some(format!("(float)(({a} >> 8) & 0xffu)")),
            "v_cvt_f32_ubyte2" => Some(format!("(float)(({a} >> 16) & 0xffu)")),
            "v_cvt_f32_ubyte3" => Some(format!("(float)(({a} >> 24) & 0xffu)")),
            "v_cvt_f32_f16" => Some(format!("f16tof32({a})")),
            "v_fract_f32" => Some(format!("frac({})", fa()?)),
            "v_trunc_f32" => Some(format!("trunc({})", fa()?)),
            "v_ceil_f32" => Some(format!("ceil({})", fa()?)),
            "v_rndne_f32" => Some(format!("round({})", fa()?)),
            "v_floor_f32" => Some(format!("floor({})", fa()?)),
            "v_exp_f32" => Some(format!("exp2({})", fa()?)),
            "v_log_f32" => Some(format!("log2({})", fa()?)),
            "v_rcp_f32" | "v_rcp_iflag_f32" => Some(format!("1.0 / {}", fa()?)),
            "v_rsq_f32" => Some(format!("rsqrt({})", fa()?)),
            "v_sqrt_f32" => Some(format!("sqrt({})", fa()?)),
            "v_sin_f32" => Some(format!("sin({} * 6.28318530718)", fa()?)),
            "v_cubema_f32" => Some(format!("tb_cubema({}, {}, {})", fa()?, fb()?, fc()?)),
            "v_cubesc_f32" => Some(format!("tb_cubesc({}, {}, {})", fa()?, fb()?, fc()?)),
            "v_cubetc_f32" => Some(format!("tb_cubetc({}, {}, {})", fa()?, fb()?, fc()?)),
            "v_cubeid_f32" => Some(format!("tb_cubeid({}, {}, {})", fa()?, fb()?, fc()?)),
            "v_cos_f32" => Some(format!("cos({} * 6.28318530718)", fa()?)),
            _ => None,
        };
        if let Some(e) = fexpr {
            let e = self.fres(ins, e);
            return self.wv(&d, &e);
        }
        let mask = if vop3 { c.clone() } else { "vcc_lo".to_string() };
        let e = match m.as_str() {
            "v_mov_b32" => a,
            "v_cndmask_b32" => format!("(({mask} & 1u) != 0u) ? {b} : {a}"),
            "v_and_b32" => format!("({a} & {b})"),
            "v_or_b32" => format!("({a} | {b})"),
            "v_xor_b32" => format!("({a} ^ {b})"),
            "v_xnor_b32" => format!("(~({a} ^ {b}))"),
            "v_not_b32" => format!("(~{a})"),
            "v_bfrev_b32" => format!("reversebits({a})"),
            // Find-first-bit: -1 when no bit is set (HLSL intrinsics agree for zero input).
            "v_ffbl_b32" => format!("firstbitlow({a})"),
            "v_ffbh_u32" => format!("(({a}) == 0u ? 0xffffffffu : (31u - firstbithigh({a})))"),
            "v_lshrrev_b32" => format!("({b} >> ({a} & 31u))"),
            "v_lshlrev_b32" => format!("({b} << ({a} & 31u))"),
            "v_ashrrev_i32" => format!("asuint(asint({b}) >> ({a} & 31u))"),
            "v_add_nc_u32" => format!("({a} + {b})"),
            "v_sub_nc_u32" => format!("({a} - {b})"),
            "v_subrev_nc_u32" => format!("({b} - {a})"),
            "v_min_u32" => format!("min({a}, {b})"),
            "v_max_u32" => format!("max({a}, {b})"),
            "v_min_i32" => format!("asuint(min(asint({a}), asint({b})))"),
            "v_max_i32" => format!("asuint(max(asint({a}), asint({b})))"),
            "v_mul_u32_u24" => format!("(({a} & 0xffffffu) * ({b} & 0xffffffu))"),
            "v_mul_i32_i24" => format!("asuint((asint({a} << 8) >> 8) * (asint({b} << 8) >> 8))"),
            "v_mul_lo_u32" | "v_mul_lo_i32" => format!("({a} * {b})"),
            // Count of mask bits below this lane, plus src1. One invocation = one lane with
            // nothing below it, so the count term is zero.
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => b,
            "v_mul_hi_u32" => format!("tb_mulhi_u32({a}, {b})"),
            "v_mul_hi_i32" => format!("tb_mulhi_i32({a}, {b})"),
            "v_mad_u32_u24" => format!("(({a} & 0xffffffu) * ({b} & 0xffffffu) + {c})"),
            "v_mad_i32_i24" => format!("asuint((asint({a} << 8) >> 8) * (asint({b} << 8) >> 8) + asint({c}))"),
            "v_bfe_u32" => format!("(({a} >> ({b} & 31u)) & ((1u << ({c} & 31u)) - 1u))"),
            "v_bfi_b32" => format!("(({a} & {b}) | (~{a} & {c}))"),
            "v_lshl_add_u32" => format!("(({a} << ({b} & 31u)) + {c})"),
            "v_add_lshl_u32" => format!("(({a} + {b}) << ({c} & 31u))"),
            "v_add3_u32" => format!("({a} + {b} + {c})"),
            "v_lshl_or_b32" => format!("(({a} << ({b} & 31u)) | {c})"),
            "v_and_or_b32" => format!("(({a} & {b}) | {c})"),
            "v_or3_b32" => format!("({a} | {b} | {c})"),
            "v_xor3_b32" => format!("({a} ^ {b} ^ {c})"),
            "v_xad_u32" => format!("(({a} ^ {b}) + {c})"),
            "v_min3_u32" => format!("min(min({a}, {b}), {c})"),
            "v_max3_u32" => format!("max(max({a}, {b}), {c})"),
            "v_med3_u32" => format!("max(min({a}, {b}), min(max({a}, {b}), {c}))"),
            "v_min3_i32" => format!("asuint(min(min(asint({a}), asint({b})), asint({c})))"),
            "v_max3_i32" => format!("asuint(max(max(asint({a}), asint({b})), asint({c})))"),
            "v_med3_i32" => format!(
                "asuint(max(min(asint({a}), asint({b})), min(max(asint({a}), asint({b})), asint({c}))))"
            ),
            // {S0:S1} >> S2[4:0], low 32 bits (S0 is the high word).
            "v_alignbit_b32" => format!(
                "((({c}) & 31u) == 0u ? {b} : (({b} >> (({c}) & 31u)) | ({a} << (32u - (({c}) & 31u)))))"
            ),
            "v_cvt_u32_f32" => format!("(uint)max({}, 0.0)", fa()?),
            "v_cvt_i32_f32" => format!("asuint((int){})", fa()?),
            "v_cvt_flr_i32_f32" => format!("asuint((int)floor({}))", fa()?),
            "v_cvt_rpi_i32_f32" => format!("asuint((int)floor({} + 0.5))", fa()?),
            "v_cvt_f16_f32" => format!("f32tof16({})", fa()?),
            "v_cvt_pkrtz_f16_f32" => format!("(f32tof16({}) | (f32tof16({}) << 16))", fa()?, fb()?),
            _ => return Err("unsupported vector op".into()),
        };
        self.wv(&d, &e)
    }

    /// Number of dwords moved by a MUBUF dword load/store mnemonic.
    fn mubuf_dwords(m: &str) -> Option<u32> {
        Some(match m.rsplit('_').next()? {
            "dword" => 1,
            "dwordx2" => 2,
            "dwordx3" => 3,
            "dwordx4" => 4,
            _ => return None,
        })
    }

    /// MUBUF on a structured buffer: `vaddr` is the element index (IDXEN), the 12-bit
    /// immediate the byte offset inside the element. Dynamic offsets (OFFEN / SOFFSET
    /// register) are not handled yet and fail explicitly.
    fn emit_mubuf(&mut self, idx: usize, ins: &Insn) -> Result<(), String> {
        if let Some(Res::CbDyn(slot)) = self.res.get(&idx).and_then(|v| v.first()).copied() {
            let (offen, idxen) = (ins.aux.mask & 1 != 0, ins.aux.mask & 2 != 0);
            if (idxen && offen) || !matches!(ins.src[2], Opnd::ConstInt(0)) || !ins.mnemonic.starts_with("buffer_load_dword") {
                return Err("constant-buffer MUBUF form unsupported".into());
            }
            let n = Self::mubuf_dwords(&ins.mnemonic).ok_or("unsupported MUBUF op")?;
            let vd = match ins.dst[0] {
                Opnd::Vgpr(v) => v,
                _ => return Err("bad vdata".into()),
            };
            let base = if offen {
                format!("({:#x}u + {})", ins.aux.imm as u32, r(&ins.src[0])?)
            } else if idxen {
                // Indexed access to a constant buffer = a dynamically indexed cbuffer array.
                // Elements of cbuffer arrays are 16-byte registers, which is the stride the
                // CBV descriptor carries (inferred: the title reads 3 dwords per element).
                format!("({:#x}u + {} * 16u)", ins.aux.imm as u32, r(&ins.src[0])?)
            } else {
                format!("{:#x}u", ins.aux.imm as u32)
            };
            let t = self.tmp;
            self.tmp += 1;
            self.line(&format!("uint t{t} = {base} >> 2;"));
            for k in 0..n {
                self.line(&format!("uint t{t}_{k} = cb{slot}[((t{t} + {k}u) >> 2) & 4095u][(t{t} + {k}u) & 3u];"));
            }
            for k in 0..n {
                self.wv(&Opnd::Vgpr(vd + k as u8), &format!("t{t}_{k}"))?;
            }
            return Ok(());
        }
        if ins.mnemonic.contains("_atomic_") {
            // Buffer atomics → HLSL Interlocked* on the UAV element. With GLC set the
            // pre-operation value is returned in vdata.
            let (offen, idxen, glc) = (ins.aux.mask & 1 != 0, ins.aux.mask & 2 != 0, ins.aux.mask & 4 != 0);
            if (offen && idxen) || !matches!(ins.src[2], Opnd::ConstInt(0)) {
                return Err("atomic MUBUF addressing unsupported".into());
            }
            // Without IDXEN the address is just the immediate offset: element 0 (the usual
            // wave-level counter allocation `InterlockedAdd(buf[0], n)`). With OFFEN alone the
            // address register is a byte offset.
            let index = if idxen { r(&ins.src[0])? } else { "0u".to_string() };
            let target = match (self.res.get(&idx).and_then(|v| v.first()).copied(), offen) {
                (Some(Res::TBuf(slot, true)), false) if ins.aux.imm == 0 => format!("utb{slot}[{index}]"),
                (Some(Res::TBuf(slot, true)), true) => format!("utb{slot}[(({}) + {:#x}u) >> 2]", r(&ins.src[0])?, ins.aux.imm as u32),
                (Some(Res::Buf(slot, true)), false) => {
                    let (nwords, w) = (self.sbuf_words((true, slot)), ins.aux.imm as u32 / 4);
                    format!("ubuf{slot}[{index} + {}u].w[{}]", w / nwords, w % nwords)
                }
                (Some(Res::Buf(slot, true)), true) => {
                    let nwords = self.sbuf_words((true, slot));
                    let w = format!("((({}) + {:#x}u) >> 2)", r(&ins.src[0])?, ins.aux.imm as u32);
                    format!("ubuf{slot}[{w} / {nwords}u].w[{w} % {nwords}u]")
                }
                _ => return Err("unresolved atomic target".into()),
            };
            let vdata = ins.dst[0];
            let val = r(&vdata)?;
            let (func, arg) = match ins.mnemonic.as_str() {
                "buffer_atomic_add" => ("InterlockedAdd", val),
                "buffer_atomic_sub" => ("InterlockedAdd", format!("(0u - {val})")),
                "buffer_atomic_inc" => ("InterlockedAdd", "1u".to_string()),
                "buffer_atomic_dec" => ("InterlockedAdd", "0xffffffffu".to_string()),
                "buffer_atomic_umin" => ("InterlockedMin", val),
                "buffer_atomic_umax" => ("InterlockedMax", val),
                "buffer_atomic_and" => ("InterlockedAnd", val),
                "buffer_atomic_or" => ("InterlockedOr", val),
                "buffer_atomic_xor" => ("InterlockedXor", val),
                "buffer_atomic_swap" => ("InterlockedExchange", val),
                other => return Err(format!("{other} unsupported")),
            };
            let t = self.tmp;
            self.tmp += 1;
            self.line(&format!("uint t{t} = 0u; if (exec_lo & 1u) {{ {func}({target}, {arg}, t{t}); }}"));
            if glc {
                self.wv(&vdata, &format!("t{t}"))?;
            }
            return Ok(());
        }
        if let Some(Res::TBuf(slot, true)) = self.res.get(&idx).and_then(|v| v.first()).copied() {
            if self.tbuf_uint.contains(&slot) {
                // uint-typed UAV (see atomics): single-component raw loads/stores.
                let (offen, idxen) = (ins.aux.mask & 1 != 0, ins.aux.mask & 2 != 0);
                // A buffer used with atomics has a single 32-bit channel (D3D requires
                // R32_UINT/SINT for Interlocked*), so multi-component `_format_xy/xyz/xyzw`
                // accesses move only X; a load fills the missing channels with (0, 0, 1).
                let comps = match ins.mnemonic.rsplit('_').next() {
                    Some("x") => 1u8,
                    Some("xy") => 2,
                    Some("xyz") => 3,
                    Some("xyzw") => 4,
                    _ => 0,
                };
                if offen || ins.aux.imm != 0 || comps == 0 {
                    return Err("uint typed-buffer MUBUF form unsupported".into());
                }
                // Without IDXEN the access addresses element 0.
                let index = if idxen { r(&ins.src[0])? } else { "0u".to_string() };
                let vd = ins.dst[0];
                if ins.mnemonic.starts_with("buffer_store") {
                    self.touch(&vd);
                    self.line(&format!("if (exec_lo & 1u) utb{slot}[{index}] = {};", r(&vd)?));
                } else {
                    let t = self.tmp;
                    self.tmp += 1;
                    self.line(&format!("uint t{t} = utb{slot}[{index}];"));
                    self.wv(&vd, &format!("t{t}"))?;
                    if let Opnd::Vgpr(v) = vd {
                        for k in 1..comps {
                            self.wv(&Opnd::Vgpr(v + k), if k == 3 { "1u" } else { "0u" })?;
                        }
                    }
                }
                return Ok(());
            }
        }
        if let Some(Res::TBuf(slot, is_uav)) = self.res.get(&idx).and_then(|v| v.first()).copied() {
            // Typed buffer. The element type is declared float4: the view format is not
            // visible in the shader, so integer-format buffers are a known gap (TECHNICAL_DEBT).
            let (offen, idxen) = (ins.aux.mask & 1 != 0, ins.aux.mask & 2 != 0);
            if offen || ins.aux.imm != 0 || !matches!(ins.src[2], Opnd::ConstInt(0)) {
                return Err("typed-buffer MUBUF with offset addressing unsupported".into());
            }
            let n = match ins.mnemonic.rsplit('_').next() {
                Some("x") => 1u8,
                Some("xy") => 2,
                Some("xyz") => 3,
                Some("xyzw") => 4,
                _ => return Err("unsupported typed-buffer op".into()),
            };
            // Without IDXEN (and with no offset) the access addresses element 0.
            let index = if idxen { r(&ins.src[0])? } else { "0u".to_string() };
            let vd = match ins.dst[0] {
                Opnd::Vgpr(v) => v,
                _ => return Err("bad vdata".into()),
            };
            let nm = if is_uav { format!("utb{slot}") } else { format!("stb{slot}") };
            if ins.mnemonic.starts_with("buffer_store") {
                if !is_uav {
                    return Err("buffer_store_format to a non-UAV table".into());
                }
                let mut c = vec![];
                for k in 0..4u8 {
                    if k < n {
                        let o = Opnd::Vgpr(vd + k);
                        self.touch(&o);
                        c.push(f(&o)?);
                    } else {
                        c.push("0.0".into());
                    }
                }
                match self.tbuf_ty.get(&(is_uav, slot)) {
                    Some(t) => {
                        let mut c = vec![];
                        for k in 0..4u8 {
                            c.push(if k < n { r(&Opnd::Vgpr(vd + k))? } else { "0u".into() });
                        }
                        let ctor = if *t == 2 { "int4" } else { "uint4" };
                        self.line(&format!("if (exec_lo & 1u) {nm}[{index}] = {ctor}({});", c.join(", ")));
                    }
                    None => self.line(&format!("if (exec_lo & 1u) {nm}[{index}] = float4({});", c.join(", "))),
                }
            } else {
                let t = self.tmp;
                self.tmp += 1;
                let ety = match self.tbuf_ty.get(&(is_uav, slot)) {
                    Some(2) => "int4",
                    Some(_) => "uint4",
                    None => "float4",
                };
                self.line(&format!("{ety} t{t} = {nm}[{index}];"));
                for k in 0..n {
                    let c = XYZW[k as usize];
                    let e = match ety {
                        "float4" => format!("asuint(t{t}.{c})"),
                        "int4" => format!("(uint)t{t}.{c}"),
                        _ => format!("t{t}.{c}"),
                    };
                    self.wv(&Opnd::Vgpr(vd + k), &e)?;
                }
            }
            return Ok(());
        }
        let (slot, is_uav) = match self.res.get(&idx).and_then(|v| v.first()) {
            Some(Res::Buf(s, u)) => (*s, *u),
            _ => return Err("unresolved buffer".into()),
        };
        // Address = element index (IDXEN: first address register) * stride + byte offset
        // (OFFEN: the next address register, or the first when there is no index) + imm.
        let (offen, idxen) = (ins.aux.mask & 1 != 0, ins.aux.mask & 2 != 0);
        if !matches!(ins.src[2], Opnd::ConstInt(0)) {
            return Err("MUBUF with a dynamic SOFFSET is unsupported".into());
        }
        let n = Self::mubuf_dwords(&ins.mnemonic).ok_or("unsupported MUBUF op")?;
        let a0 = ins.src[0];
        let a1 = match a0 {
            Opnd::Vgpr(v) => Opnd::Vgpr(v + 1),
            other => other,
        };
        let index = if idxen { r(&a0)? } else { "0u".to_string() };
        let voff = match (idxen, offen) {
            (_, false) => None,
            (false, true) => Some(r(&a0)?),
            (true, true) => {
                self.touch(&a1);
                Some(r(&a1)?)
            }
        };
        let w0 = ins.aux.imm as u32 / 4;
        let vd = match ins.dst[0] {
            Opnd::Vgpr(v) => v,
            _ => return Err("bad vdata".into()),
        };
        let nm = if is_uav { format!("ubuf{slot}") } else { format!("sbuf{slot}") };
        // Element `k` of this access. With a run-time byte offset the buffer is addressed
        // flat over the declared element size (which assumes, like every structured-buffer
        // access here, that the declared element matches the view's stride).
        let nwords = self.sbuf_words((is_uav, slot));
        let t = self.tmp;
        self.tmp += 1;
        if let Some(vo) = &voff {
            self.line(&format!("uint t{t}w = (({vo}) + {:#x}u) >> 2;", ins.aux.imm as u32));
        }
        let elem = |k: u32| match &voff {
            // A constant offset past the element addresses the following elements.
            None if w0 + k >= nwords => format!("{nm}[{index} + {}u].w[{}]", (w0 + k) / nwords, (w0 + k) % nwords),
            None => format!("{nm}[{index}].w[{}]", w0 + k),
            Some(_) => format!("{nm}[{index} + (t{t}w + {k}u) / {nwords}u].w[(t{t}w + {k}u) % {nwords}u]"),
        };
        if ins.mnemonic.starts_with("buffer_store") {
            if !is_uav {
                return Err("buffer_store to a non-UAV table".into());
            }
            for k in 0..n {
                let o = Opnd::Vgpr(vd + k as u8);
                self.touch(&o);
                self.line(&format!("if (exec_lo & 1u) {} = {};", elem(k), r(&o)?));
            }
        } else {
            for k in 0..n {
                self.line(&format!("uint t{t}_{k} = {};", elem(k)));
            }
            for k in 0..n {
                self.wv(&Opnd::Vgpr(vd + k as u8), &format!("t{t}_{k}"))?;
            }
        }
        Ok(())
    }

    /// Geometry-shader LDS layout (read off WriteToSliceMainGS): input vertex data lives at
    /// `vertexBase + register*16 + component*4`, emitted vertices at
    /// `primBase + vertex*stride + register*16 + component*4`, stride = 16 * output registers.
    /// The translation seeds the vertex-offset registers with synthetic offsets (vertex k at
    /// dword offset k*0x400, i.e. byte k*0x1000 after the shader's own <<2), so an input
    /// address decodes exactly as (vertex = A >> 12, slot = (A & 0xfff) >> 2).
    fn gs_dims(&self) -> Result<(usize, usize, usize, &'static str, &'static str), String> {
        let gi = self.ex.gs_info.ok_or("geometry shader without PSV0 GS info")?;
        let (prim, nv) = match gi.input_primitive {
            1 => ("point", 1usize),
            2 => ("line", 2),
            3 => ("triangle", 3),
            other => return Err(format!("GS input primitive {other} unsupported")),
        };
        let stream = match gi.output_topology {
            1 => "PointStream",
            3 => "LineStream",
            5 => "TriangleStream",
            other => return Err(format!("GS output topology {other} unsupported")),
        };
        let in_dw = 4 * (self.ex.input_sig.iter().map(|e| e.register).max().unwrap_or(0) as usize + 1);
        let out_dw = 4 * (self.ex.output_sig.iter().map(|e| e.register).max().unwrap_or(0) as usize + 1);
        Ok((nv, in_dw, out_dw, prim, stream))
    }

    fn gs_max_vertices(&self) -> usize {
        let emits = self
            .insns
            .iter()
            .filter(|i| i.enc == Enc::Sopp && i.op == 0x10 && (i.aux.imm as u32 & 0xf) == 2 && (i.aux.imm as u32 >> 4) & 2 != 0)
            .count();
        (self.ex.gs_info.map(|g| g.max_vertex_count as usize).unwrap_or(0)).max(emits).max(1)
    }

    fn emit_gs_lds(&mut self, ins: &Insn) -> Result<(), String> {
        let (nv, in_dw, out_dw, _, _) = self.gs_dims()?;
        let gs_dw = out_dw * self.gs_max_vertices();
        let off = ins.aux.imm as u32;
        let (off0, off1) = (off & 0xff, (off >> 8) & 0xff);
        let addr = r(&ins.src[0])?;
        let t = self.tmp;
        self.tmp += 1;
        let in_at = |a: String| format!("tb_in[({a} >> 12) % {nv}u][(({a} & 0xfffu) >> 2) % {in_dw}u]");
        let vd = match ins.dst[0] {
            Opnd::Vgpr(v) => v,
            _ => return Err("bad vdst".into()),
        };
        match ins.mnemonic.as_str() {
            "ds_read_b32" => {
                self.line(&format!("uint t{t} = {};", in_at(format!("({addr} + {:#x}u)", off & 0xffff))));
                self.wv(&Opnd::Vgpr(vd), &format!("t{t}"))
            }
            "ds_read2_b32" => {
                self.line(&format!(
                    "uint t{t}a = {}; uint t{t}b = {};",
                    in_at(format!("({addr} + {:#x}u)", off0 * 4)),
                    in_at(format!("({addr} + {:#x}u)", off1 * 4))
                ));
                self.wv(&Opnd::Vgpr(vd), &format!("t{t}a"))?;
                self.wv(&Opnd::Vgpr(vd + 1), &format!("t{t}b"))
            }
            "ds_write_b32" => {
                self.line(&format!(
                    "if (exec_lo & 1u) tb_gs[(({addr} + {:#x}u) >> 2) % {gs_dw}u] = {};",
                    off & 0xffff,
                    r(&ins.src[1])?
                ));
                Ok(())
            }
            "ds_write2_b32" => {
                self.line(&format!(
                    "if (exec_lo & 1u) {{ tb_gs[(({addr} + {:#x}u) >> 2) % {gs_dw}u] = {}; tb_gs[(({addr} + {:#x}u) >> 2) % {gs_dw}u] = {}; }}",
                    off0 * 4,
                    r(&ins.src[1])?,
                    off1 * 4,
                    r(&ins.src[2])?
                ));
                Ok(())
            }
            _ => Err("unsupported LDS op".into()),
        }
    }

    /// `s_sendmsg`: in a geometry shader, message 2 = GS with op bits (1 = cut, 2 = emit),
    /// message 3 = GS_DONE.
    fn emit_sendmsg(&mut self, ins: &Insn) -> Result<(), String> {
        if !self.is_gs {
            return Err("s_sendmsg outside a geometry shader".into());
        }
        let imm = ins.aux.imm as u32 & 0xffff;
        let (msg, op, stream) = (imm & 0xf, (imm >> 4) & 3, (imm >> 8) & 3);
        match msg {
            3 => Ok(()),
            2 => {
                if stream != 0 {
                    return Err("GS output stream != 0 unsupported".into());
                }
                let (_, _, out_dw, _, _) = self.gs_dims()?;
                let maxv = self.gs_max_vertices();
                if op & 2 != 0 {
                    let mut body = format!("if (exec_lo & 1u) {{ SOut o = (SOut)0; uint tb_b = min(tb_emit, {}u) * {out_dw}u;", maxv - 1);
                    for (ei, e) in self.ex.output_sig.iter().enumerate() {
                        let (ty, _) = sig_type(e);
                        let comps: Vec<String> = (0..4)
                            .filter(|c| (e.mask >> c) & 1 != 0)
                            .map(|c| {
                                let w = format!("tb_gs[tb_b + {}u]", e.register as usize * 4 + c);
                                if e.comp_type == 3 { format!("asfloat({w})") } else { w }
                            })
                            .collect();
                        let _ = write!(body, " o.o{ei} = ({ty}){};", if comps.len() == 1 { comps[0].clone() } else { format!("{ty}({})", comps.join(", ")) });
                    }
                    body.push_str(" stream.Append(o); tb_emit++; }");
                    self.line(&body);
                }
                if op & 1 != 0 {
                    self.line("if (exec_lo & 1u) stream.RestartStrip();");
                }
                Ok(())
            }
            other => Err(format!("s_sendmsg message {other} unsupported")),
        }
    }

    fn emit_mimg(&mut self, idx: usize, ins: &Insn) -> Result<(), String> {
        let rs = self.res.get(&idx).cloned().ok_or("unresolved image")?;
        // Texel-addressed ops (no sampler): image_load / image_store, on SRVs or UAVs.
        if matches!(ins.mnemonic.as_str(), "image_load" | "image_load_mip" | "image_store" | "image_atomic_add" | "image_atomic_umax") {
            let with_mip = ins.mnemonic == "image_load_mip";
            let (slot, dim, is_uav) = match rs.first() {
                Some(Res::Tex(t, d)) => (*t, *d, false),
                Some(Res::Uav(t, d)) => (*t, *d, true),
                _ => return Err("unresolved image".into()),
            };
            let ncoord = match dim {
                0 => 1,     // 1D
                1 => 2,     // 2D
                2 | 5 => 3, // 3D / 2D array
                4 => 2,     // 1D array
                6 => 2,     // 2D multisampled (the next component is the sample index)
                _ => return Err(format!("image dim {dim} unsupported for load/store")),
            };
            let nsa = ins.aux.imm2 as usize;
            let first = match ins.src[0] {
                Opnd::Vgpr(n) => n,
                _ => return Err("bad vaddr".into()),
            };
            let mut c = vec![];
            for i in 0..ncoord {
                let o = if nsa > 0 { ins.src[i] } else { Opnd::Vgpr(first + i as u8) };
                self.touch(&o);
                c.push(r(&o)?);
            }
            let vd = match ins.dst[0] {
                Opnd::Vgpr(n) => n,
                _ => return Err("bad vdata".into()),
            };
            let as_uint = is_uav && self.uav_uint.contains(&slot);
            if ins.mnemonic.starts_with("image_atomic_") {
                // Image atomics → Interlocked* on a RWTexture<uint> texel; vdata carries the
                // operand in and the previous value out.
                if !as_uint {
                    return Err("image atomic on a non-UAV table".into());
                }
                let func = if ins.mnemonic == "image_atomic_add" { "InterlockedAdd" } else { "InterlockedMax" };
                let o = Opnd::Vgpr(vd);
                self.touch(&o);
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!("uint t{t} = 0u;"));
                self.line(&format!("if (exec_lo & 1u) {func}(uav{slot}[uint{ncoord}({})], {}, t{t});", c.join(", "), r(&o)?));
                return self.wv(&o, &format!("t{t}"));
            }
            if ins.mnemonic == "image_store" && as_uint {
                let o = Opnd::Vgpr(vd);
                self.touch(&o);
                self.line(&format!("if (exec_lo & 1u) uav{slot}[uint{ncoord}({})] = {};", c.join(", "), r(&o)?));
                return Ok(());
            }
            if ins.mnemonic == "image_load" && as_uint {
                let t = self.tmp;
                self.tmp += 1;
                self.line(&format!("uint4 t{t} = uint4(uav{slot}[uint{ncoord}({})], 0u, 0u, 0u);", c.join(", ")));
                let mut n = 0u8;
                for ch in 0..4 {
                    if (ins.aux.mask >> ch) & 1 != 0 {
                        self.wv(&Opnd::Vgpr(vd + n), &format!("t{t}.{}", XYZW[ch]))?;
                        n += 1;
                    }
                }
                return Ok(());
            }
            let ity = self.img_ty.get(&(is_uav, slot)).copied();
            if ins.mnemonic == "image_store" {
                if !is_uav {
                    return Err("image_store to a non-UAV table".into());
                }
                if let Some(t) = ity {
                    // Integer texels: the registers are stored as they are.
                    let mut comp = vec![];
                    let mut n = 0u8;
                    for ch in 0..4 {
                        if (ins.aux.mask >> ch) & 1 != 0 {
                            let o = Opnd::Vgpr(vd + n);
                            self.touch(&o);
                            comp.push(r(&o)?);
                            n += 1;
                        } else {
                            comp.push("0u".into());
                        }
                    }
                    let ctor = if t == 2 { "int4" } else { "uint4" };
                    self.line(&format!("if (exec_lo & 1u) uav{slot}[uint{ncoord}({})] = {ctor}({});", c.join(", "), comp.join(", ")));
                    return Ok(());
                }
                let mut comp = vec![];
                let mut n = 0u8;
                for ch in 0..4 {
                    if (ins.aux.mask >> ch) & 1 != 0 {
                        let o = Opnd::Vgpr(vd + n);
                        self.touch(&o);
                        comp.push(f(&o)?);
                        n += 1;
                    } else {
                        comp.push("0.0".into());
                    }
                }
                self.line(&format!(
                    "if (exec_lo & 1u) uav{slot}[uint{ncoord}({})] = float4({});",
                    c.join(", "),
                    comp.join(", ")
                ));
            } else {
                let t = self.tmp;
                self.tmp += 1;
                // image_load_mip carries the mip level as one more address component.
                let mip = if with_mip {
                    let o = if nsa > 0 { ins.src[ncoord] } else { Opnd::Vgpr(first + ncoord as u8) };
                    self.touch(&o);
                    format!("(int){}", r(&o)?)
                } else {
                    "0".to_string()
                };
                let e = if dim == 6 {
                    if is_uav {
                        return Err("multisampled UAV load unsupported".into());
                    }
                    let o = if nsa > 0 { ins.src[ncoord] } else { Opnd::Vgpr(first + ncoord as u8) };
                    self.touch(&o);
                    format!("tex{slot}.Load(int2({}), (int){})", c.iter().map(|x| format!("(int){x}")).collect::<Vec<_>>().join(", "), r(&o)?)
                } else if is_uav {
                    if with_mip {
                        return Err("image_load_mip on a UAV unsupported".into());
                    }
                    format!("uav{slot}[uint{ncoord}({})]", c.join(", "))
                } else {
                    format!("tex{slot}.Load(int{}({}, {mip}))", ncoord + 1, c.iter().map(|x| format!("(int){x}")).collect::<Vec<_>>().join(", "))
                };
                let ety = match ity {
                    Some(2) => "int4",
                    Some(_) => "uint4",
                    None => "float4",
                };
                self.line(&format!("{ety} t{t} = {e};"));
                let mut n = 0u8;
                for ch in 0..4 {
                    if (ins.aux.mask >> ch) & 1 != 0 {
                        let e = match ety {
                            "float4" => format!("asuint(t{t}.{})", XYZW[ch]),
                            "int4" => format!("(uint)t{t}.{}", XYZW[ch]),
                            _ => format!("t{t}.{}", XYZW[ch]),
                        };
                        self.wv(&Opnd::Vgpr(vd + n), &e)?;
                        n += 1;
                    }
                }
            }
            return Ok(());
        }
        let (tslot, dim) = match rs.first() {
            Some(Res::Tex(t, d)) => (*t, *d),
            _ => return Err("unresolved image".into()),
        };
        let sslot = match rs.get(1) {
            Some(Res::Samp(s)) => *s,
            _ => return Err("image op without sampler unsupported".into()),
        };
        // RDNA2 sample/gather family. Opcode = family base (0x20 sample, 0x40 gather4) +
        // modifier (0 none, 1 cl, 2 d, 3 d_cl, 4 l, 5 b, 6 b_cl, 7 lz) + 8 (depth compare)
        // + 0x10 (texel offset). Address operands arrive in this order:
        //   [offset] [bias] [zref] [ddx.. ddy..] coords.. [lod] [lod clamp]
        let op = ins.op;
        let gather = (0x40..0x60).contains(&op);
        if !gather && !(0x20..0x40).contains(&op) {
            return Err("unsupported image op".into());
        }
        let (modi, cmp, has_off) = (op & 7, op & 8 != 0, op & 0x10 != 0);
        // (coordinate count, derivative components, offset components)
        let (ncoord, nder, noff) = match dim {
            0 => (1usize, 1usize, 1usize),
            1 => (2, 2, 2),
            2 => (3, 3, 3),
            3 => (3, 3, 0),
            4 => (2, 1, 1),
            5 => (3, 2, 2),
            _ => return Err(format!("image dim {dim} unsupported")),
        };
        // Address operands: explicit list with NSA, else consecutive VGPRs from vaddr.
        let nsa = ins.aux.imm2 as usize;
        let first = match ins.src[0] {
            Opnd::Vgpr(n) => n,
            _ => return Err("bad vaddr".into()),
        };
        let addr = |i: usize| -> Opnd {
            if nsa > 0 { ins.src[i] } else { Opnd::Vgpr(first + i as u8) }
        };
        let mut next = 0usize;
        let mut take = |em: &mut Self| -> Opnd {
            let o = addr(next);
            next += 1;
            em.touch(&o);
            o
        };
        // Texel offset: packed signed 6-bit fields (x [5:0], y [13:8], z [21:16]).
        let offset = if has_off {
            let o = take(self);
            if noff == 0 {
                return Err("texel offset on a cube image".into());
            }
            let sx = |v: u32, sh: u32| (((v >> sh) & 0x3f) as i32) << 26 >> 26;
            let reg = match o {
                Opnd::Vgpr(n) => n,
                _ => return Err("bad offset operand".into()),
            };
            match self.const_vgpr(idx, reg) {
                Some(v) => {
                    let c: Vec<String> = (0..noff).map(|k| sx(v, 8 * k as u32).to_string()).collect();
                    Some(if noff == 1 { c[0].clone() } else { format!("int{noff}({})", c.join(", ")) })
                }
                None if gather => {
                    // SM5 Gather accepts a run-time offset.
                    let x = r(&o)?;
                    let c: Vec<String> =
                        (0..noff).map(|k| format!("(asint({x} << {}) >> 26)", 26 - 8 * k)).collect();
                    Some(if noff == 1 { c[0].clone() } else { format!("int{noff}({})", c.join(", ")) })
                }
                None => return Err("non-constant texel offset on a sample op".into()),
            }
        } else {
            None
        };
        let bias = if matches!(modi, 5 | 6) { Some(f(&take(self))?) } else { None };
        let zref = if cmp { Some(f(&take(self))?) } else { None };
        let grads = if matches!(modi, 2 | 3) {
            let mut dx = vec![];
            for _ in 0..nder {
                dx.push(f(&take(self))?);
            }
            let mut dy = vec![];
            for _ in 0..nder {
                dy.push(f(&take(self))?);
            }
            let v = |c: &Vec<String>| if c.len() == 1 { c[0].clone() } else { format!("float{}({})", c.len(), c.join(", ")) };
            Some((v(&dx), v(&dy)))
        } else {
            None
        };
        let mut coords = vec![];
        for _ in 0..ncoord {
            coords.push(f(&take(self))?);
        }
        let lod = if modi == 4 { Some(f(&take(self))?) } else { None };
        let clamp = if matches!(modi, 1 | 3 | 6) { Some(f(&take(self))?) } else { None };
        // Cube images are addressed by hardware (s, t, face) coordinates (see tb_cube_dir).
        let cvec = if dim == 3 {
            format!("tb_cube_dir({})", coords.join(", "))
        } else if ncoord == 1 {
            coords[0].clone()
        } else {
            format!("float{ncoord}({})", coords.join(", "))
        };
        let tex = format!("tex{tslot}");
        let smp = if cmp { format!("scmp{sslot}") } else { format!("samp{sslot}") };
        let zero_off = if noff == 1 { "0".to_string() } else { format!("int{}({})", noff.max(1), vec!["0"; noff.max(1)].join(", ")) };
        // Optional trailing ", offset[, clamp]" arguments.
        let tail = |offset: &Option<String>, clamp: &Option<String>| -> Result<String, String> {
            Ok(match (offset, clamp) {
                (None, None) => String::new(),
                (Some(o), None) => format!(", {o}"),
                (o, Some(c)) => {
                    if noff == 0 {
                        return Err("LOD clamp on a cube image unsupported".into());
                    }
                    format!(", {}, {c}", o.clone().unwrap_or_else(|| zero_off.clone()))
                }
            })
        };
        let t = self.tmp;
        self.tmp += 1;
        let vd = match ins.dst[0] {
            Opnd::Vgpr(n) => n,
            _ => return Err("bad vdata".into()),
        };
        if gather {
            // gather4: dmask picks the channel; the four texels fill vdata..vdata+3.
            if !matches!(modi, 0 | 7) {
                return Err("gather4 with explicit LOD/bias unsupported".into());
            }
            let chan = match ins.aux.mask {
                1 => "Red",
                2 => "Green",
                4 => "Blue",
                8 => "Alpha",
                m => return Err(format!("gather4 with dmask {m:#x} unsupported")),
            };
            let off = offset.map(|o| format!(", {o}")).unwrap_or_default();
            let call = match &zref {
                Some(z) => format!("{tex}.GatherCmp{chan}({smp}, {cvec}, {z}{off})"),
                None => format!("{tex}.Gather{chan}({smp}, {cvec}{off})"),
            };
            self.line(&format!("float4 t{t} = {call};"));
            for k in 0..4u8 {
                self.wv(&Opnd::Vgpr(vd + k), &format!("asuint(t{t}.{})", XYZW[k as usize]))?;
            }
            return Ok(());
        }
        // FXC (SM5) refuses implicit-derivative samples inside a loop it cannot unroll
        // (X3570/X3511). The console GPU computes those derivatives per quad regardless of
        // control flow; there is no SM5 equivalent, so inside emitted loops such samples
        // are APPROXIMATED with LOD 0 (exact for single-mip textures, too sharp otherwise).
        let in_loop = self.loop_depth > 0;
        if in_loop && matches!(modi, 0 | 1 | 5 | 6) {
            self.line("// APPROX-LOOP-SAMPLE: implicit-derivative sample inside a loop emitted as LOD 0");
        }
        let call = if let Some(z) = &zref {
            // Depth-compare sampling returns a single value.
            match modi {
                0 if in_loop => format!("{tex}.SampleCmpLevelZero({smp}, {cvec}, {z}{})", tail(&offset, &None)?),
                0 => format!("{tex}.SampleCmp({smp}, {cvec}, {z}{})", tail(&offset, &None)?),
                7 => format!("{tex}.SampleCmpLevelZero({smp}, {cvec}, {z}{})", tail(&offset, &None)?),
                _ => return Err("depth-compare sample with explicit LOD/bias/gradient unsupported in SM5".into()),
            }
        } else {
            match modi {
                0 | 1 if in_loop => format!("{tex}.SampleLevel({smp}, {cvec}, 0.0{})", tail(&offset, &None)?),
                5 | 6 if in_loop => format!("{tex}.SampleLevel({smp}, {cvec}, {}{})", bias.clone().ok_or("missing bias")?, tail(&offset, &None)?),
                0 | 1 => format!("{tex}.Sample({smp}, {cvec}{})", tail(&offset, &clamp)?),
                2 | 3 => {
                    let (dx, dy) = grads.clone().ok_or("missing gradients")?;
                    format!("{tex}.SampleGrad({smp}, {cvec}, {dx}, {dy}{})", tail(&offset, &clamp)?)
                }
                4 => format!("{tex}.SampleLevel({smp}, {cvec}, {}{})", lod.clone().ok_or("missing lod")?, tail(&offset, &None)?),
                5 | 6 => format!("{tex}.SampleBias({smp}, {cvec}, {}{})", bias.clone().ok_or("missing bias")?, tail(&offset, &clamp)?),
                _ => format!("{tex}.SampleLevel({smp}, {cvec}, 0.0{})", tail(&offset, &None)?),
            }
        };
        if zref.is_some() {
            self.line(&format!("float t{t} = {call};"));
            // The compare result is replicated to every requested channel.
            let mut n = 0u8;
            for ch in 0..4 {
                if (ins.aux.mask >> ch) & 1 != 0 {
                    self.wv(&Opnd::Vgpr(vd + n), &format!("asuint(t{t})"))?;
                    n += 1;
                }
            }
            return Ok(());
        }
        self.line(&format!("float4 t{t} = {call};"));
        let mut n = 0u8;
        for ch in 0..4 {
            if (ins.aux.mask >> ch) & 1 != 0 {
                let dst = Opnd::Vgpr(vd + n);
                self.wv(&dst, &format!("asuint(t{t}.{})", XYZW[ch]))?;
                n += 1;
            }
        }
        Ok(())
    }

    /// Statically known value of a VGPR at instruction `idx`: the most recent write in
    /// program order must be a `v_mov_b32` of a constant. Used where HLSL demands a
    /// literal (texel offsets).
    fn const_vgpr(&self, idx: usize, reg: u8) -> Option<u32> {
        for ins in self.insns[..idx].iter().rev() {
            if ins.dst.iter().any(|d| matches!(d, Opnd::Vgpr(n) if *n == reg)) {
                if ins.mnemonic != "v_mov_b32" {
                    return None;
                }
                return match ins.src.first() {
                    Some(Opnd::ConstInt(v)) => Some(*v as i32 as u32),
                    Some(Opnd::Literal(v)) => Some(*v),
                    Some(Opnd::ConstF32(f)) => Some(f.to_bits()),
                    _ => None,
                };
            }
        }
        None
    }

    fn emit_exp(&mut self, ins: &Insn, is_ps: bool) -> Result<(), String> {
        let target = ins.aux.imm as u32;
        let en = ins.aux.mask;
        let compr = ins.aux.imm2 & 1 != 0;
        let mut comp = vec![];
        if compr {
            for i in 0..4usize {
                let src = &ins.src[i / 2];
                let on = (en >> (i / 2 * 2)) & 1 != 0 || (en >> i) & 1 != 0;
                comp.push(if on {
                    format!("f16tof32(({} >> {}) & 0xffffu)", r(src)?, (i % 2) * 16)
                } else {
                    "0.0".into()
                });
            }
        } else {
            for i in 0..4usize {
                comp.push(if (en >> i) & 1 != 0 { f(&ins.src[i])? } else { "0.0".into() });
            }
        }
        let v4 = format!("float4({})", comp.join(", "));
        // The exported registers as raw 32-bit values, when they are available unpacked.
        let raw: Option<Vec<String>> = if compr {
            self.epilog_exp.get(&self.cur_idx).copied().map(|(a, b)| {
                (0..4usize)
                    .map(|i| {
                        let id = if i < 2 { a } else { b };
                        let on = (en >> (i / 2 * 2)) & 1 != 0 || (en >> i) & 1 != 0;
                        if on && id != usize::MAX { format!("tb_pk{id}{}", if i % 2 == 0 { "a" } else { "b" }) } else { "0u".to_string() }
                    })
                    .collect()
            })
        } else {
            let mut v = vec![];
            for i in 0..4usize {
                v.push(if (en >> i) & 1 != 0 { r(&ins.src[i])? } else { "0u".into() });
            }
            Some(v)
        };
        if is_ps && ins.aux.imm2 & 4 != 0 {
            // Valid-mask export: lanes that are inactive here are killed (pixel discard).
            self.line("if ((exec_lo & 1u) == 0u) discard;");
        }
        if is_ps && target == 9 {
            return Ok(()); // null export (no colour written)
        }
        if is_ps && target == 8 {
            // MRTZ: x = depth.
            if en & 1 == 0 {
                return Err("MRTZ export without depth (stencil/sample-mask only) unsupported".into());
            }
            if !self.ex.output_sig.iter().any(|e| matches!(e.sysval, 65 | 67 | 68)) {
                return Err("depth export without an SV_Depth output".into());
            }
            self.line(&format!("if (exec_lo & 1u) o.od = {};", f(&ins.src[0])?));
            return Ok(());
        }
        if !is_ps && (32..64).contains(&target) && (target - 32) as usize >= self.out_params.len() {
            // Hardware needs at least one parameter export; with no matching output
            // element nothing downstream can read it.
            self.line("// (parameter export with no output-signature element: ignored)");
            return Ok(());
        }
        // (element index, swizzle of the exported register that feeds it)
        let osig = &self.ex.output_sig;
        let mut assigns: Vec<(usize, String)> = Vec::new();
        if is_ps {
            if target > 7 {
                return Err(format!("PS export target {target} unsupported"));
            }
            // MRT export k feeds the k-th declared SV_Target (outputs are compacted:
            // a shader writing SV_Target0 and SV_Target4 exports MRT 0 and MRT 1).
            let ei = osig
                .iter()
                .enumerate()
                .filter(|(_, e)| e.sysval == 64)
                .nth(target as usize)
                .map(|(i, _)| i)
                .ok_or_else(|| format!("MRT export {target} has no SV_Target output element"))?;
            assigns.push((ei, "xyzw"[..sig_type(&osig[ei]).1].to_string()));
        } else if (12..=15).contains(&target) {
            if target != 12 {
                return Err(format!("position export {target} unsupported"));
            }
            match osig.iter().position(|e| e.sysval == 1) {
                Some(ei) => assigns.push((ei, "xyzw".to_string())),
                None => {
                    // A vertex shader feeding a geometry shader (entry "MainForGS") still
                    // carries a position export, but its D3D output signature has no
                    // SV_Position: the next stage consumes only the parameter exports.
                    self.line("// (position export with no SV_Position output element: ignored)");
                    return Ok(());
                }
            }
        } else if (32..64).contains(&target) {
            let reg = self.out_params[(target - 32) as usize];
            for (ei, e) in osig.iter().enumerate() {
                if e.sysval == 0 && e.register == reg {
                    let sw: String = (0..4).filter(|c| (e.mask >> c) & 1 != 0).map(|c| XYZW[c]).collect();
                    assigns.push((ei, sw));
                }
            }
        } else {
            return Err(format!("export target {target} unsupported"));
        }
        let mut lines = vec![];
        for (ei, sw) in assigns {
            let (ty, _) = sig_type(&osig[ei]);
            // Integer elements carry the register bits themselves; converting them through
            // float (or half) would destroy them.
            let val = match (osig[ei].comp_type, &raw) {
                (1, Some(rc)) => format!("uint4({})", rc.join(", ")),
                (2, Some(rc)) => format!("int4(uint4({}))", rc.join(", ")),
                (1 | 2, None) => return Err("integer output element fed by a packed export without a recognised pack epilogue".into()),
                (_, Some(rc)) if compr => format!("float4({})", rc.iter().map(|x| format!("asfloat({x})")).collect::<Vec<_>>().join(", ")),
                _ => v4.clone(),
            };
            lines.push(format!("if (exec_lo & 1u) o.o{ei} = ({ty})({val}.{sw});"));
        }
        for l in lines {
            self.line(&l);
        }
        Ok(())
    }

    /// Declarations of every register variable the body uses. Shaders with
    /// register-relative moves keep their VGPRs in one array (`tb_v`), with each `vN` name
    /// defined as an element of it, so `VGPR[n + M0]` is a plain dynamic index.
    fn reg_decls(&self) -> String {
        let is_v = |r: &str| r.len() > 1 && r.starts_with('v') && r[1..].bytes().all(|b| b.is_ascii_digit());
        let mut out = String::new();
        let scalars: Vec<String> =
            self.regs.iter().filter(|r| !(self.vreg_array && is_v(r))).map(|r| format!("{r} = 0u")).collect();
        if !scalars.is_empty() {
            out.push_str(&format!("    uint {};", scalars.join(", ")));
        }
        if self.vreg_array {
            out.push_str("\n    uint tb_v[256] = (uint[256])0;");
            for n in 0..256 {
                out.push_str(&format!("\n#define v{n} tb_v[{n}]"));
            }
        }
        out
    }

    fn finish(&mut self, is_ps: bool) -> Result<String, String> {
        let ex = self.ex;
        let mut h = String::new();
        let _ = writeln!(h, "// Generated by tombox-xshader from RDNA2 microcode ({:?}, {} instrs).", ex.stage, self.insns.len());
        h.push_str(CUBE_HELPERS);
        for (slot, dwords) in &self.cb_size {
            let _ = writeln!(h, "cbuffer CB{slot} : register(b{slot}) {{ uint4 cb{slot}[{}]; }};", dwords.div_ceil(4).max(1));
        }
        for (slot, dim) in &self.tex {
            let ty = match dim {
                0 => "Texture1D",
                4 => "Texture1DArray",
                1 => "Texture2D",
                2 => "Texture3D",
                3 => "TextureCube",
                5 => "Texture2DArray",
                6 => "Texture2DMS",
                _ => return Err(format!("texture dim {dim} unsupported")),
            };
            let e = match self.img_ty.get(&(false, *slot)) {
                Some(2) => "int4",
                Some(_) => "uint4",
                None => "float4",
            };
            let _ = writeln!(h, "{ty}<{e}> tex{slot} : register(t{slot});");
        }
        for slot in &self.samp {
            let _ = writeln!(h, "SamplerState samp{slot} : register(s{slot});");
        }
        for slot in &self.samp_cmp {
            if self.samp.contains(slot) {
                return Err(format!("sampler s{slot} used both as a regular and a comparison sampler"));
            }
            let _ = writeln!(h, "SamplerComparisonState scmp{slot} : register(s{slot});");
        }
        for (is_uav, slot) in self.sbuf.keys() {
            let (ty, nm, reg) = if *is_uav { ("RWStructuredBuffer", "ubuf", "u") } else { ("StructuredBuffer", "sbuf", "t") };
            let _ = writeln!(h, "struct TbElem_{nm}{slot} {{ uint w[{}]; }};
{ty}<TbElem_{nm}{slot}> {nm}{slot} : register({reg}{slot});", self.sbuf_words((*is_uav, *slot)));
            if self.sbuf_flat.contains(&(*is_uav, *slot)) {
                // Word at a byte offset: the element stride is a property of the view, not of
                // the shader, so it is queried.
                let _ = writeln!(h, "uint tb_sbw_{nm}{slot}(uint off) {{ uint n, st; {nm}{slot}.GetDimensions(n, st); st = max(st, 4u); uint e = off / st; return {nm}{slot}[e].w[(off - e * st) >> 2]; }}");
            }
        }
        for (is_uav, slot) in &self.tbuf {
            let _ = if *is_uav && self.tbuf_uint.contains(slot) {
                writeln!(h, "RWBuffer<uint> utb{slot} : register(u{slot});")
            } else {
                let e = match self.tbuf_ty.get(&(*is_uav, *slot)) {
                    Some(2) => "int4",
                    Some(_) => "uint4",
                    None => "float4",
                };
                if *is_uav {
                    writeln!(h, "RWBuffer<{e}> utb{slot} : register(u{slot});")
                } else {
                    writeln!(h, "Buffer<{e}> stb{slot} : register(t{slot});")
                }
            };
        }
        for (slot, dim) in &self.uav {
            // Textures used with image atomics hold one uint per texel.
            let e = match self.img_ty.get(&(true, *slot)) {
                Some(2) => "int4",
                Some(_) => "uint4",
                None => "float4",
            };
            let ty = match (dim, self.uav_uint.contains(slot)) {
                (1, false) => format!("RWTexture2D<{e}>"),
                (2, false) => format!("RWTexture3D<{e}>"),
                (5, false) => format!("RWTexture2DArray<{e}>"),
                (1, true) => "RWTexture2D<uint>".to_string(),
                (2, true) => "RWTexture3D<uint>".to_string(),
                (5, true) => "RWTexture2DArray<uint>".to_string(),
                _ => return Err(format!("UAV dim {dim} unsupported")),
            };
            let _ = writeln!(h, "{ty} uav{slot} : register(u{slot});");
        }
        if self.uses_code {
            // Constant tables embedded after the program in the code section. Code pointers
            // are byte offsets from the program start, so readers subtract the program
            // length (in dwords) before indexing. Only the tail is emitted: FXC allows at
            // most 4096 indexable literals.
            let words: Vec<String> = ex
                .isa_data
                .get(ex.isa.len()..)
                .unwrap_or(&[])
                .chunks_exact(4)
                .map(|c| format!("{:#x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect();
            let _ = writeln!(h, "static const uint tb_code[{}] = {{ {} }};", words.len().max(1), if words.is_empty() { "0".to_string() } else { words.join(", ") });
        }
        if self.is_cs && self.uses_lds {
            let _ = writeln!(h, "groupshared uint tb_lds[8192];");
        }
        if self.is_cs {
            // Compute ABI (from ClearCS + FSSM metadata): local thread id in v0..v2, the
            // thread-group id in the three SGPRs following the user-data (root-parameter)
            // registers, thread-group size in the ISA preamble dwords 1..3.
            let p = ex.isa_preamble;
            let (tx, ty_, tz) = (p[1].max(1), p[2].max(1), p[3].max(1));
            let nparams = ex.root_sig.as_ref().map(|r| r.params.len()).unwrap_or(4);
            let g = 4 + nparams;
            let _ = writeln!(h, "[numthreads({tx}, {ty_}, {tz})]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {{");
            let _ = writeln!(h, "    uint exec_lo = 1u, exec_hi = 0u, vcc_lo = 0u, vcc_hi = 0u, scc = 0u, m0 = 0u, nul = 0u;");
            for r in ["v0", "v1", "v2"] {
                self.regs.insert(r.to_string());
            }
            for k in 0..3 {
                self.regs.insert(format!("s{}", g + k));
            }
            let _ = writeln!(h, "{}", self.reg_decls());
            let _ = writeln!(h, "    v0 = gtid.x; v1 = gtid.y; v2 = gtid.z;");
            let _ = writeln!(h, "    s{} = gid.x; s{} = gid.y; s{} = gid.z;", g, g + 1, g + 2);
            h.push_str(&self.body);
            h.push_str("}
");
            return Ok(h);
        }
        let _ = writeln!(h, "struct SIn {{");
        // HLSL requires system-generated inputs (SV_IsFrontFace, SV_SampleIndex,
        // SV_PrimitiveID, ...) to follow all ordinary parameters, so emit them last.
        // Field names stay tied to the signature element index.
        let sysgen = |e: &SigElement| is_ps && !matches!(e.sysval, 0 | 1);
        for pass in 0..2 {
            for (ei, e) in ex.input_sig.iter().enumerate() {
                if sysgen(e) != (pass == 1) {
                    continue;
                }
                let (ty, _) = sig_type(e);
                // DXIL validation: with a conservative depth output (SV_DepthLessEqual /
                // SV_DepthGreaterEqual) the position input must be centroid-interpolated.
                let cons_depth = ex.output_sig.iter().any(|o| {
                    let s = semantic(o);
                    s.starts_with("SV_DepthLessEqual") || s.starts_with("SV_DepthGreaterEqual")
                });
                let interp = if !is_ps {
                    ""
                } else if e.sysval == 1 && cons_depth {
                    "linear noperspective centroid "
                } else if e.comp_type != 3 {
                    "nointerpolation "
                } else if e.name.contains("_centroid") {
                    "centroid "
                } else {
                    ""
                };
                let _ = writeln!(h, "    {interp}{ty} i{ei} : {};", semantic(e));
            }
        }
        let pos_field = match ex.input_sig.iter().position(|e| e.sysval == 1) {
            Some(ei) => format!("i.i{ei}"),
            None => "i.tb_pos".to_string(),
        };
        if is_ps && self.uses_quad && pos_field == "i.tb_pos" {
            let _ = writeln!(h, "    float4 tb_pos : SV_Position;");
        }
        let _ = writeln!(h, "}};\nstruct SOut {{");
        for (ei, e) in ex.output_sig.iter().enumerate() {
            let (ty, _) = sig_type(e);
            if matches!(e.sysval, 65 | 67 | 68) {
                let _ = writeln!(h, "    float od : {};", semantic(e).trim_end_matches('0'));
            } else if e.sysval == 1 && !is_ps {
                // `precise`: no fast-math on the position computation. The title relies on a
                // depth-only pass and the base pass producing identical depth (the base pass
                // tests without writing); the two are separate shaders here, and without this
                // the compiler may contract or reorder them differently (observed as holes
                // in the character where the base pass failed the depth test).
                let _ = writeln!(h, "    precise {ty} o{ei} : {};", semantic(e));
            } else {
                let _ = writeln!(h, "    {ty} o{ei} : {};", semantic(e));
            }
        }
        let no_out = ex.output_sig.is_empty();
        if no_out {
            // No outputs (e.g. depth-only / kill-only pixel shader): HLSL forbids empty structs.
            let cut = h.rfind("struct SOut {").unwrap_or(h.len());
            h.truncate(cut);
            let _ = writeln!(h, "void main(SIn i) {{");
        } else if self.is_gs {
            let (nv, _, _, prim, stream) = self.gs_dims()?;
            let _ = writeln!(
                h,
                "}};\n[maxvertexcount({})]\nvoid main({prim} SIn i[{nv}], inout {stream}<SOut> stream) {{",
                self.gs_max_vertices()
            );
        } else {
            let _ = writeln!(h, "}};\nSOut main(SIn i) {{\n    SOut o = (SOut)0;");
        }
        let _ = writeln!(h, "    uint exec_lo = 1u, exec_hi = 0u, vcc_lo = 0u, vcc_hi = 0u, scc = 0u, m0 = 0u, nul = 0u;");
        // Vertex attributes arrive after the fetch call in consecutive groups of four VGPRs
        // from v4, one group per input register the shader reads. A register none of whose
        // elements is read (read mask 0 in the input signature) gets no group: in the skinned
        // depth-only shader d15c18830a4616f9 ATTRIBUTE1/2 (tangents) are unread and the bone
        // indices of ATTRIBUTE3 arrive in v8..v11, not v16..v19.
        let mut init = String::new();
        if self.is_gs {
            // GS entry ABI (from WriteToSliceMainGS): s3 carries the wave's primitive count in
            // bits [15:8]; v0/v1 hold the vertex offsets (v0 = offset0 | offset1 << 16, v1 =
            // offset2), in dwords. One invocation = one primitive.
            let (nv, in_dw, out_dw, _, _) = self.gs_dims()?;
            let _ = writeln!(init, "    uint tb_in[{nv}][{in_dw}]; uint tb_gs[{}]; uint tb_emit = 0u;", out_dw * self.gs_max_vertices());
            let _ = writeln!(init, "    {{ for (uint a = 0u; a < {}u; a++) tb_gs[a] = 0u; }}", out_dw * self.gs_max_vertices());
            let _ = writeln!(init, "    {{ for (uint k = 0u; k < {nv}u; k++) for (uint a = 0u; a < {in_dw}u; a++) tb_in[k][a] = 0u; }}");
            for k in 0..nv {
                for (ei, e) in ex.input_sig.iter().enumerate() {
                    let mut n = 0;
                    for c in 0..4usize {
                        if (e.mask >> c) & 1 != 0 {
                            let acc = format!("i[{k}].i{ei}.{}", XYZW[n]);
                            let val = if e.comp_type == 3 { format!("asuint({acc})") } else { format!("(uint)({acc})") };
                            let _ = writeln!(init, "    tb_in[{k}][{}] = {val};", e.register as usize * 4 + c);
                            n += 1;
                        }
                    }
                }
            }
            for rname in ["s3", "v0", "v1"] {
                self.regs.insert(rname.to_string());
            }
            let _ = writeln!(init, "    s3 = 0x100u; v0 = 0x04000000u; v1 = 0x800u;");
        }
        if !is_ps && !self.is_gs {
            let mut group = 0usize;
            for reg in distinct_regs(ex.input_sig.iter().filter(|e| e.sysval == 0)) {
                let read = ex.input_sig.iter().any(|e| e.sysval == 0 && e.register == reg && e.rw_mask != 0);
                if ex.stage == Stage::Vertex && !read {
                    continue;
                }
                let first = if ex.stage == Stage::Vertex { 4 + 4 * group } else { 4 + 4 * reg as usize };
                group += 1;
                for c in 0..4usize {
                    let v = format!("v{}", first + c);
                    self.regs.insert(v.clone());
                    let val = match sig_lookup(&ex.input_sig, reg, c) {
                        Some((ei, k)) => {
                            let acc = format!("i.i{ei}.{}", XYZW[k]);
                            if ex.input_sig[ei].comp_type == 3 { format!("asuint({acc})") } else { format!("(uint)({acc})") }
                        }
                        // Unsupplied components default to (0,0,0,1) like the hardware fetch.
                        None => (if c == 3 { "0x3f800000u" } else { "0u" }).to_string(),
                    };
                    let _ = writeln!(init, "    {v} = {val};");
                }
            }
        }
        if !self.regs.is_empty() {
            let _ = writeln!(h, "{}", self.reg_decls());
        }
        if is_ps && self.uses_quad {
            let _ = writeln!(init, "    uint tb_quad = ((uint){pos_field}.x & 1u) | (((uint){pos_field}.y & 1u) << 1);");
        }
        if is_ps {
            // PS entry ABI (observed): v0/v1 = perspective-centre barycentrics, then the
            // fragment position x, y, z, 1/w in v2..v5 when the shader reads it.
            if let Some(ei) = ex.input_sig.iter().position(|e| e.sysval == 1) {
                let p = format!("i.i{ei}");
                let exprs = [format!("{p}.x"), format!("{p}.y"), format!("{p}.z"), format!("1.0 / {p}.w")];
                for (k, expr) in exprs.iter().enumerate() {
                    let v = format!("v{}", 2 + k);
                    if self.live.contains(&v) {
                        let _ = writeln!(init, "    {v} = asuint({expr});");
                    }
                }
            }
        }
        h.push_str(&init);
        h.push_str(&self.body);
        h.push_str(if no_out || self.is_gs { "}\n" } else { "    return o;\n}\n" });
        Ok(h)
    }
}
