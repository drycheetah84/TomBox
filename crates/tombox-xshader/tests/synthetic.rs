//! Translator regression tests on hand-assembled RDNA2 programs.
//!
//! These cover behaviour that was found wrong while running the title, without needing a
//! game shader as a fixture: each program is a few instructions written out as raw dwords,
//! wrapped in a `ShaderExtract` with a made-up signature.

use tombox_xshader::hlsl::translate;
use tombox_xshader::meta::{RootParam, RootParamKind, RootRange, RootSig, SigElement};
use tombox_xshader::{ShaderExtract, Stage};

#[cfg(windows)]
#[path = "support/d3d12.rs"]
mod d3d12;

const S_ENDPGM: u32 = 0xbf81_0000;
const S_NOP: u32 = 0xbf80_0000;

#[test]
fn mesh_allocation_and_triangle_exports_compile_to_dxil() {
    let mut ex = shader(Stage::Mesh, &[
        0xbefc03ff, 0x80080, 0xbf900009,
        0x7e000280, 0x7e020280, 0x7e040280, 0x7e060280,
        0xf8000141, 0,
        0xf80008cf, 0x03020100,
        S_ENDPGM,
    ], vec![], vec![sig("SV_Position", 1, 3, 0, 15)], None);
    ex.mesh_info = Some(tombox_xshader::meta::MeshPipelineInfo::Mesh {
        threads: Some([128, 1, 1]), payload_bytes: 0, max_vertices: 128,
        max_primitives: 128, topology: "triangle",
    });
    let translated = translate(&ex).unwrap();
    assert_eq!(translated.profile, "ms_6_5");
    // Counts come from M0 (GS_ALLOC_REQ). The strict path names them before the call; the
    // earlier assertion looked for `SetMeshOutputCounts(m0 & 0xfffu` and failed on its own
    // branch as well.
    assert!(translated.hlsl.contains("tb_vertex_count = m0 & 0xfffu"), "{}", translated.hlsl);
    assert!(translated.hlsl.contains("SetMeshOutputCounts(tb_vertex_count, tb_primitive_count)"), "{}", translated.hlsl);
    #[cfg(windows)]
    {
        let dxil = tombox_xshader::dxc::compile(&translated.hlsl, translated.profile).unwrap();
        assert!(dxil.starts_with(b"DXBC"));
    }
}

#[test]
fn mesh_rejects_unresolved_entry_registers() {
    for (instruction, expected) in [(0x7e000305, "VGPR ABI unresolved: v5"), (0xbe800304, "SGPR ABI unresolved: s4")] {
        let mut ex = shader(Stage::Mesh, &[instruction, 0xbefc03ff, 0x80080, 0xbf900009, S_ENDPGM], vec![], vec![], None);
        ex.mesh_info = Some(tombox_xshader::meta::MeshPipelineInfo::Mesh {
            threads: Some([128, 1, 1]), payload_bytes: 0, max_vertices: 128,
            max_primitives: 128, topology: "triangle",
        });
        let error = translate(&ex).err().unwrap();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn mesh_rejects_unverified_control_flow_counts_and_wave_dimensions() {
    let cases: &[(&[u32], Option<[u32; 3]>, &str)] = &[
        (&[0xbf840002, 0xbefc03ff, 0x80080, 0xbf900009, S_ENDPGM], Some([128, 1, 1]), "uniform allocation dominance"),
        (&[0xbe800380, 0xbefc0300, 0xbf900009, S_ENDPGM], Some([128, 1, 1]), "uniform immediate allocation"),
        (&[0xbefc03ff, 0x81081, 0xbf900009, S_ENDPGM], Some([128, 1, 1]), "exceeds PSV bounds"),
        (&[0xbefc03ff, 0x80080, 0xbf900009, S_ENDPGM], Some([64, 1, 1]), "verified 128-thread ABI"),
        (&[0xbefc03ff, 0x80080, 0xbf900009, S_ENDPGM], None, "requires PSV thread dimensions"),
    ];
    for (words, threads, expected) in cases {
        let mut ex = shader(Stage::Mesh, words, vec![], vec![], None);
        ex.mesh_info = Some(tombox_xshader::meta::MeshPipelineInfo::Mesh {
            threads: *threads, payload_bytes: 0, max_vertices: 128,
            max_primitives: 128, topology: "triangle",
        });
        let error = translate(&ex).err().unwrap();
        assert!(error.contains(expected), "{error}");
    }
}
/// `v_cvt_pkrtz_f16_f32 v0, v2, v2` / `v1, v2, v2`: the default colour-export packing.
const PACK_V0_V2_V2: u32 = 0x5e00_0502;
const PACK_V1_V2_V2: u32 = 0x5e02_0502;
/// `exp mrt0 v0, v1` compressed, done, valid-mask.
const EXP_MRT0_COMPR: [u32; 2] = [0xf800_1c0f, 0x0000_0100];

fn sig(name: &str, sysval: u32, comp_type: u32, register: u32, mask: u8) -> SigElement {
    SigElement { name: name.into(), index: 0, sysval, comp_type, register, mask, rw_mask: 0 }
}

fn table(range_type: u32, num: u32, visibility: u32) -> RootParam {
    RootParam {
        visibility,
        kind: RootParamKind::Table(vec![RootRange { range_type, num, base_register: 0, space: 0, offset_in_table: u32::MAX }]),
    }
}

fn shader(stage: Stage, words: &[u32], input_sig: Vec<SigElement>, output_sig: Vec<SigElement>, root: Option<RootSig>) -> ShaderExtract {
    let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ShaderExtract {
        stage,
        shader_model: (6, 0),
        bitcode_stripped: true,
        fssm: Vec::new(),
        entry: Some("main".into()),
        isa_data: isa.clone(),
        isa,
        isa_offset_in_fssm: 0,
        isa_preamble: [0; 4],
        has_input_sig: true,
        has_output_sig: true,
        has_psv0: false,
        has_root_sig: root.is_some(),
        input_sig,
        output_sig,
        resources: Vec::new(),
        gs_info: None,
        ms_info: None,
        psv_numthreads: None,
        mesh_info: None,
        root_sig: root,
    }
}

#[test]
fn floating_point_lg_and_nlg_compare_nan_with_opposite_results() {
    let words = [
        0x7c0a_0702,
        0x7c14_0702,
        S_ENDPGM,
    ];
    let ex = shader(Stage::Pixel, &words, vec![], vec![], None);
    let t = translate(&ex).expect("translate");
    assert!(t.hlsl.contains("!isnan(asfloat(v2)) && !isnan(asfloat(v3)) && (asfloat(v2) != asfloat(v3))"));
    assert!(t.hlsl.contains("isnan(asfloat(v2)) || isnan(asfloat(v3)) || (asfloat(v2) == asfloat(v3))"));
}

#[test]
fn pso_bindings_preserve_mixed_table_types_and_register_spaces() {
    use tombox_xshader::hlsl::translate_with_pso;
    use tombox_xshader::meta::PsvResource;
    use tombox_xshader::psocache::{BindingLayout, UserDataBinding, UserDataKind};
    let words = [
        0xf408_0206, 0xfa00_0060,
        0xf420_0084, 0xfa00_0000,
        0xf408_0106, 0xfa00_01e0,
        0xf420_0082, 0xfa00_0000,
        S_ENDPGM,
    ];
    let mut ex = shader(Stage::Compute, &words, vec![], vec![], None);
    ex.isa_preamble = [0, 1, 1, 1];
    ex.resources = vec![
        PsvResource { res_type: 4, space: 1, lower: 320, upper: 320 },
        PsvResource { res_type: 2, space: 1, lower: 38, upper: 38 },
    ];
    let layout = BindingLayout {
        metadata_index: 0,
        entries: vec![
            UserDataBinding { kind: UserDataKind::ShaderResource, byte_offset: 0x60, register: 320, raw_fields: [0, 0x10c04, u32::MAX, 0] },
            UserDataBinding { kind: UserDataKind::ConstantBuffer, byte_offset: 0x1e0, register: 38, raw_fields: [0, 0x10c04, u32::MAX, 0] },
        ],
    };
    let translated = translate_with_pso(&ex, &[layout]).unwrap();
    assert!(translated.hlsl.contains("register(t320, space1)"));
    assert!(translated.hlsl.contains("register(b38, space1)"));
    assert!(translated.hlsl.contains(".Load(t"));
    #[cfg(windows)]
    if tombox_xshader::dxc::available() {
        tombox_xshader::dxc::compile(&translated.hlsl, "cs_6_0").unwrap();
    }
}

#[test]
fn pso_dynamic_descriptor_offset_selects_raw_buffers_at_runtime() {
    use tombox_xshader::hlsl::translate_with_pso;
    use tombox_xshader::meta::PsvResource;
    use tombox_xshader::psocache::{BindingLayout, UserDataBinding, UserDataKind};
    let words = [
        0xbe84_0380,
        0xf408_0206, 0x0800_0000,
        0xf420_0084, 0xfa00_0000,
        S_ENDPGM,
    ];
    let mut ex = shader(Stage::Compute, &words, vec![], vec![], None);
    ex.isa_preamble = [0, 1, 1, 1];
    ex.resources = vec![
        PsvResource { res_type: 4, space: 1, lower: 5, upper: 5 },
        PsvResource { res_type: 4, space: 2, lower: 6, upper: 6 },
    ];
    let layout = BindingLayout {
        metadata_index: 0,
        entries: vec![
            UserDataBinding { kind: UserDataKind::ShaderResource, byte_offset: 0, register: 5, raw_fields: [0, 0x10c04, u32::MAX, 0] },
            UserDataBinding { kind: UserDataKind::ShaderResource, byte_offset: 32, register: 6, raw_fields: [0, 0x20c04, u32::MAX, 0] },
        ],
    };
    let translated = translate_with_pso(&ex, &[layout]).unwrap();
    assert!(translated.hlsl.contains("register(t5, space1)"));
    assert!(translated.hlsl.contains("register(t6, space2)"));
    assert!(translated.hlsl.contains("tb_dsel8 = 0xc000000u | (s4 + 0u);"));
    assert!(translated.hlsl.contains("if (tb_dsel8 == 0xc000020u)"));
    #[cfg(windows)]
    if tombox_xshader::dxc::available() {
        tombox_xshader::dxc::compile(&translated.hlsl, "cs_6_0").unwrap();
    }
}

/// A pixel shader that writes the integer 5 to a `uint4` render target through the default
/// pack-and-export tail. Converting the integer through float-to-half (what the tail does
/// literally) would give 0; the integer itself must be exported.
#[test]
fn integer_colour_export_keeps_the_integer() {
    let words = [
        0x7e04_0285, // v_mov_b32 v2, 5
        S_NOP,
        S_NOP,
        PACK_V0_V2_V2,
        PACK_V1_V2_V2,
        EXP_MRT0_COMPR[0],
        EXP_MRT0_COMPR[1],
        S_ENDPGM,
    ];
    let ex = shader(Stage::Pixel, &words, vec![], vec![sig("SV_Target", 64, 1, 0, 0xf)], None);
    let t = translate(&ex).expect("translate");
    assert!(t.hlsl.contains("uint4 o0 : SV_Target0"), "{}", t.hlsl);
    assert!(t.hlsl.contains("o.o0 = (uint4)(uint4(tb_pk0a, tb_pk0b, tb_pk1a, tb_pk1b).xyzw);"), "{}", t.hlsl);
    assert!(t.hlsl.contains("tb_pk0a = v2;"), "{}", t.hlsl);
    assert!(!t.hlsl.contains("f16tof32"), "integer export must not go through half floats:\n{}", t.hlsl);
}

/// The same tail feeding a float render target exports the unpacked floats (the target's own
/// format conversion applies, not a fixed half-float one).
#[test]
fn float_colour_export_is_not_squeezed_through_half() {
    let words = [
        0x7e04_02f2, // v_mov_b32 v2, 1.0
        S_NOP,
        S_NOP,
        PACK_V0_V2_V2,
        PACK_V1_V2_V2,
        EXP_MRT0_COMPR[0],
        EXP_MRT0_COMPR[1],
        S_ENDPGM,
    ];
    let ex = shader(Stage::Pixel, &words, vec![], vec![sig("SV_Target", 64, 3, 0, 0xf)], None);
    let t = translate(&ex).expect("translate");
    assert!(
        t.hlsl.contains("o.o0 = (float4)(float4(asfloat(tb_pk0a), asfloat(tb_pk0b), asfloat(tb_pk1a), asfloat(tb_pk1b)).xyzw);"),
        "{}",
        t.hlsl
    );
}

/// A packed export whose operands do not come from the pack instructions directly before it
/// keeps the literal behaviour (unpack the halves).
#[test]
fn packed_export_without_the_tail_is_left_alone() {
    let words = [
        0x7e00_0280, // v_mov_b32 v0, 0
        0x7e02_0280, // v_mov_b32 v1, 0
        EXP_MRT0_COMPR[0],
        EXP_MRT0_COMPR[1],
        S_ENDPGM,
    ];
    let ex = shader(Stage::Pixel, &words, vec![], vec![sig("SV_Target", 64, 3, 0, 0xf)], None);
    let t = translate(&ex).expect("translate");
    assert!(t.hlsl.contains("f16tof32"), "{}", t.hlsl);
    assert!(!t.hlsl.contains("tb_pk"), "{}", t.hlsl);
    // ...and an integer element cannot be produced from it: that has to fail loudly.
    let ex = shader(Stage::Pixel, &words, vec![], vec![sig("SV_Target", 64, 1, 0, 0xf)], None);
    assert!(translate(&ex).is_err());
}

/// A vertex shader exporting an integer parameter (the primitive id handed to the pixel
/// shader) must pass the register bits through, not convert them as if they were a float.
#[test]
fn integer_parameter_export_keeps_the_integer() {
    let words = [
        0x7e10_0287, // v_mov_b32 v8, 7
        0x7e02_02f2, // v_mov_b32 v1, 1.0
        0xf800_08cf, // exp pos0 v1, v1, v1, v1 (done)
        0x0101_0101,
        0xf800_020f, // exp param0 v8, v8, v8, v8
        0x0808_0808,
        S_ENDPGM,
    ];
    let out = vec![sig("PRIMITIVE_ID", 0, 1, 0, 0x1), sig("SV_Position", 1, 3, 1, 0xf)];
    let ex = shader(Stage::Vertex, &words, vec![], out, None);
    let t = translate(&ex).expect("translate");
    assert!(t.hlsl.contains("o.o0 = (uint)(uint4(v8, v8, v8, v8).x);"), "{}", t.hlsl);
}

/// Texel type of a texture read with texel fetches: integer when the loaded value is used as
/// an integer, float when it is used as a float. (The lighting-channel mask is an R16_UINT
/// texture; declared as float its bits never matched the light's channel mask.)
#[test]
fn texel_type_follows_the_use_of_the_loaded_value() {
    let root = || Some(RootSig { version: 2, flags: 1, params: vec![table(0, 64, 5), table(2, 16, 5)], num_static_samplers: 0 });
    let program = |use_op: u32| {
        vec![
            0xf40c_0004, // s_load_dwordx8 s0, s8 [off=0]: texture descriptor t0
            0xfa00_0000,
            0x7e00_0280, // v_mov_b32 v0, 0
            0x7e02_0280, // v_mov_b32 v1, 0
            0xf000_1108, // image_load v0 <- 2D texture, coords v0..
            0x0000_0000,
            use_op,
            S_NOP,
            S_NOP,
            PACK_V0_V2_V2,
            PACK_V1_V2_V2,
            EXP_MRT0_COMPR[0],
            EXP_MRT0_COMPR[1],
            S_ENDPGM,
        ]
    };
    // v_and_b32 v2, 1, v0
    let ex = shader(Stage::Pixel, &program(0x3604_0081), vec![], vec![sig("SV_Target", 64, 3, 0, 0xf)], root());
    let t = translate(&ex).expect("translate (integer use)");
    assert!(t.hlsl.contains("Texture2D<uint4> tex0 : register(t0);"), "{}", t.hlsl);
    assert!(t.hlsl.contains("uint4 t0 = tex0.Load("), "{}", t.hlsl);
    // v_add_f32 v2, v0, v0
    let ex = shader(Stage::Pixel, &program(0x0604_0100), vec![], vec![sig("SV_Target", 64, 3, 0, 0xf)], root());
    let t = translate(&ex).expect("translate (float use)");
    assert!(t.hlsl.contains("Texture2D<float4> tex0 : register(t0);"), "{}", t.hlsl);
    assert!(t.hlsl.contains("float4 t0 = tex0.Load("), "{}", t.hlsl);
}

fn buffer_add_shader() -> ShaderExtract {
    let words = [
        0xf408_0002, 0xfa00_0000, // s_load_dwordx4 s[0:3], s[4:5], 0
        0xbf8c_0000,              // s_waitcnt all
        0xe030_2000, 0x8000_0300, // buffer_load_dword v3, v0, s[0:3], 0 idxen
        0xbf8c_0000,              // s_waitcnt all
        0x4a06_0687,              // v_add_nc_u32 v3, 7, v3
        0xe070_2000, 0x8000_0300, // buffer_store_dword v3, v0, s[0:3], 0 idxen
        S_ENDPGM,
    ];
    let root = RootSig {
        version: 2,
        flags: 0,
        params: vec![table(1, 1, 0)],
        num_static_samplers: 0,
    };
    let mut ex = shader(Stage::Compute, &words, vec![], vec![], Some(root));
    ex.isa_preamble = [0, 4, 1, 1];
    ex
}

#[test]
fn buffer_add_translates() {
    let translated = translate(&buffer_add_shader()).expect("translate buffer add");
    assert!(translated.hlsl.contains("RWStructuredBuffer<TbElem_ubuf0> ubuf0 : register(u0);"));
    assert!(translated.hlsl.contains("[numthreads(4, 1, 1)]"));
}

/// Run explicitly with a Windows SDK and a shader model 6-capable D3D12 WARP runtime.
#[cfg(windows)]
#[test]
#[ignore = "requires DXC and the Windows D3D12 WARP runtime"]
fn translated_compute_buffer_adds_seven() {
    let translated = translate(&buffer_add_shader()).expect("translate buffer add");
    let profile = tombox_xshader::dxc::profile_for(&translated.profile);
    let dxil = tombox_xshader::dxc::compile(&translated.hlsl, &profile).expect("compile with DXC");

    let input = [3, 29, u32::MAX - 2, 950];
    let output = d3d12::dispatch_buffer(&dxil, input).expect("dispatch on D3D12 WARP");
    assert_eq!(output, [10, 36, 4, 957]);
}

/// A scalar constant-buffer load whose byte offset comes from an entry SGPR (SOFFSET field,
/// not a decoded source operand) must count that register as read at entry; it was left
/// undeclared, and DXC rejected the shader (second title, PS_CascadeShadows_texture).
#[test]
fn smem_soffset_register_is_entry_live() {
    let root = Some(RootSig { version: 2, flags: 1, params: vec![table(2, 16, 5)], num_static_samplers: 0 });
    let words = [
        0xf408_0004, // s_load_dwordx4 s0, s8 [off=0]: constant buffer descriptor
        0xfa00_0000,
        0xf420_0100, // s_buffer_load_dword s4, s0 [off=0], soffset = s13
        0x1a00_0000,
        0x7e04_0204, // v_mov_b32 v2, s4
        PACK_V0_V2_V2,
        PACK_V1_V2_V2,
        EXP_MRT0_COMPR[0],
        EXP_MRT0_COMPR[1],
        S_ENDPGM,
    ];
    let ex = shader(Stage::Pixel, &words, vec![], vec![sig("SV_Target", 64, 3, 0, 0xf)], root);
    let t = translate(&ex).expect("translate");
    let live = t.hlsl.lines().find(|l| l.starts_with("// entry-live registers")).expect("live line");
    assert!(live.split_whitespace().any(|r| r == "s13"), "{live}\n{}", t.hlsl);
    assert!(t.hlsl.contains("+ s13) >> 2"), "{}", t.hlsl);
}

/// GFX10 VOPC opcode layout (LLVM AMDGPU VOPCInstructions.td): bit 4 is the v_cmpx form,
/// which writes EXEC only, in every encoding; 0xd0-0xd7 is v_cmpx_*_u32, and the f16
/// compares continue at 0xe8, not 0xd0.
#[test]
fn vopc_cmpx_writes_exec_and_f16_compares_are_split() {
    use tombox_xshader::rdna2::{disassemble, Opnd};
    let words = [
        0x7c28_0300u32,          // v_cmpx_gt_f32 v0, v1 (e32)
        0x7da2_0300,             // v_cmpx_lt_u32 v0, v1 (e32)
        0x7dd0_0300,             // v_cmp_u_f16 v0, v1 (e32)
        0xd496_006a, 0x0002_0300, // v_cmpx_ge_i32 v0, v1 (e64, SDST field ignored)
        S_ENDPGM,
    ];
    let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let got: Vec<(String, Opnd)> = disassemble(&isa).into_iter().map(|i| (i.mnemonic, i.dst[0])).take(4).collect();
    let want = [
        ("v_cmpx_gt_f32", Opnd::ExecLo),
        ("v_cmpx_lt_u32", Opnd::ExecLo),
        ("v_cmp_u_f16", Opnd::VccLo),
        ("v_cmpx_ge_i32", Opnd::ExecLo),
    ];
    assert_eq!(got, want.map(|(m, d)| (m.to_string(), d)));
}

/// One-wave compute shader (8x8 threads) that uses LDS without `s_barrier`, as in the
/// title's bloom-chain setup (`251ed6ba5631cd6d`): every lane writes slot 0, then lanes add
/// into it atomically, then read it back. The console runs the 64 lanes in lockstep; on a
/// PC the group may be several waves, so a group barrier must separate each LDS phase.
/// Without them the count was read early and groups took the wrong path (flickering blocks
/// of infinite values in the bloom chain).
#[test]
fn one_wave_compute_lds_phases_get_group_barriers() {
    let words = [
        0xd834_0000, 0x0000_0909, // ds_write_b32 v0, v9 (addr v9)
        0xbf8c_c07f,              // s_waitcnt lgkmcnt(0)
        0xd800_0000, 0x0000_0609, // ds_add_u32 v9, v6
        0xbf8c_c07f,              // s_waitcnt
        0xd8d8_0000, 0x0700_0007, // ds_read_b32 v7, v7
        0xbf8c_c07f,
        S_ENDPGM,
    ];
    let mut ex = shader(Stage::Compute, &words, vec![], vec![], None);
    ex.psv_numthreads = Some([8, 8, 1]);
    let t = translate(&ex).expect("translate");
    let lines: Vec<&str> = t.hlsl.lines().collect();
    let at = |pat: &str| lines.iter().position(|l| l.contains(pat)).unwrap_or_else(|| panic!("{pat} missing\n{}", t.hlsl));
    let write = at("] = v9;");
    let add = at("InterlockedAdd(tb_lds");
    let read = lines.iter().position(|l| l.contains("= tb_lds[")).expect("read");
    let barriers: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| l.contains("GroupMemoryBarrierWithGroupSync")).map(|(i, _)| i).collect();
    assert!(barriers.iter().any(|&b| write < b && b < add), "no barrier between write and add\n{}", t.hlsl);
    assert!(barriers.iter().any(|&b| add < b && b < read), "no barrier between add and read\n{}", t.hlsl);

    // Groups larger than one wave are left alone (they would carry their own s_barrier).
    ex.psv_numthreads = Some([16, 16, 1]);
    let t = translate(&ex).expect("translate");
    assert!(!t.hlsl.contains("wave_lds_sync"), "{}", t.hlsl);
}

/// GFX10 FLAT/GLOBAL opcode 0x0e is `load_dwordx4` and 0x0f is `load_dwordx3` (the same
/// order as MUBUF; LLVM `FLATInstructions.td`, `FLAT_Real_Base_gfx10` and
/// `FLAT_Real_AllAddr_gfx10`). With the two swapped, a dwordx3 load from a code-section
/// table also overwrote the register after its destination range.
#[test]
fn global_load_dwordx3_and_dwordx4_write_their_own_register_count() {
    for (op, mnemonic, count) in [(0x0e_u32, "flat_load_dwordx4", 4u8), (0x0f, "flat_load_dwordx3", 3)] {
        let words = [
            0xbe80_1f00,                         // s_getpc_b64 s[0:1]
            0x7e00_0200,                         // v_mov_b32 v0, s0
            0x7e02_0201,                         // v_mov_b32 v1, s1
            0xdc00_8000 | op << 18, 0x047d_0000, // global_load_dwordx{4,3} v[4:..], v[0:1], off
            0xbf8c_0000,                         // s_waitcnt 0
            S_ENDPGM,
        ];
        let ex = shader(Stage::Compute, &words, vec![], vec![], None);
        let t = translate(&ex).expect("translate");
        for v in 4..8u8 {
            let written = t.hlsl.contains(&format!("v{v} = t"));
            assert_eq!(written, v < 4 + count, "{mnemonic} (op {op:#x}) and v{v}
{}", t.hlsl);
        }
        assert_eq!(tombox_xshader::rdna2::disassemble(&ex.isa)[3].mnemonic, mnemonic);
    }
}
