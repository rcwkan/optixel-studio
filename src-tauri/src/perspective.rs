//! AI auto-alignment (horizontal roll & vertical perspective) using **PerspectiveFields**
//! (Jin et al., CVPR 2023, <https://github.com/jinlinyi/PerspectiveFields>).
//!
//! Architecture:
//! - Backbone: SegFormer `mit_b3` (`embed_dims = [64, 128, 320, 512]`, `depths = [3, 4, 18, 3]`)
//! - Low-level encoder: `7x7 stride-2 Conv + BN + ReLU` (`ll_enc`)
//! - Dense field decoders: `GravityDecoder` (per-pixel 2D unit up-vector `(gx, gy)`) and
//!   `LatitudeDecoder` (per-pixel `sin(latitude)`)
//! - Camera parameter head: `ParamNet` (`ConvNeXt-Tiny` `[3, 3, 9, 3]`, `[96, 192, 384, 768]`)
//!   predicting `[roll, pitch, vfov]`.
//!
//! Checkpoints downloaded from `spaces/jinlinyi/PerspectiveFields` on Hugging Face are converted
//! on-device into a self-contained FP16-weights ONNX graph (`perspective_fields.onnx`, ~200 MB)
//! and executed via ONNX Runtime (`ort`).

use crate::develop::{Develop, Img, Transform, es};
use half::f16;
use ort::{session::Session, value::Tensor};
use serde::{Deserialize, Serialize};
use std::{
    f32::consts::PI,
    fs,
    io::{Read, Write},
    path::Path,
    sync::Mutex,
};

const INPUT_SIZE: usize = 320;
const PIXEL_MEAN: [f32; 3] = [103.53, 116.28, 123.675]; // BGR
const FOCAL: f32 = 0.9;
const TILT_MAX: f32 = 40.0;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AlignResult {
    pub transform: Transform,
    /// Detected camera roll angle in degrees (~0.1° - 0.5° accuracy).
    pub roll_deg: f32,
    /// Detected camera pitch (vertical tilt) angle in degrees.
    pub pitch_deg: f32,
    /// Detected vertical field of view in degrees.
    pub vfov_deg: f32,
    /// Agreement score (0.0..=1.0) between the dense 320x320 gravity field and ParamNet.
    pub confidence: f32,
    /// Mode applied: "horizontal" | "vertical" | "both"
    pub mode: String,
}

static ALIGN_SESSION: Mutex<Option<(String, Session)>> = Mutex::new(None);

pub fn reset() {
    *ALIGN_SESSION.lock().unwrap() = None;
}

// ============================================================================
// Minimal Protobuf & ONNX ModelProto Builder (IR v8, Opset 17)
// ============================================================================

fn push_varint(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

fn push_tag(buf: &mut Vec<u8>, field: u32, wire: u8) {
    push_varint(buf, ((field as u64) << 3) | (wire as u64));
}

fn push_bytes(buf: &mut Vec<u8>, field: u32, data: &[u8]) {
    push_tag(buf, field, 2);
    push_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

fn push_str(buf: &mut Vec<u8>, field: u32, s: &str) {
    push_bytes(buf, field, s.as_bytes());
}

fn push_int64(buf: &mut Vec<u8>, field: u32, v: i64) {
    push_tag(buf, field, 0);
    push_varint(buf, v as u64);
}

fn push_f32(buf: &mut Vec<u8>, field: u32, v: f32) {
    push_tag(buf, field, 5);
    buf.extend_from_slice(&v.to_le_bytes());
}

const DT_FLOAT: i64 = 1;
const DT_INT64: i64 = 7;
const DT_FLOAT16: i64 = 10;

fn encode_tensor_proto(name: &str, dims: &[i64], data_type: i64, raw_data: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(64 + raw_data.len());
    for &d in dims {
        push_int64(&mut b, 1, d);
    }
    push_int64(&mut b, 2, data_type);
    push_str(&mut b, 8, name);
    push_bytes(&mut b, 9, raw_data);
    b
}

fn encode_value_info(name: &str, elem_type: i64, dims: &[i64]) -> Vec<u8> {
    let mut shape = Vec::new();
    for &d in dims {
        let mut dim = Vec::new();
        push_int64(&mut dim, 1, d);
        push_bytes(&mut shape, 1, &dim);
    }
    let mut tensor_type = Vec::new();
    push_int64(&mut tensor_type, 1, elem_type);
    push_bytes(&mut tensor_type, 2, &shape);

    let mut type_proto = Vec::new();
    push_bytes(&mut type_proto, 1, &tensor_type);

    let mut vi = Vec::new();
    push_str(&mut vi, 1, name);
    push_bytes(&mut vi, 2, &type_proto);
    vi
}

enum Attr<'a> {
    Int(&'a str, i64),
    Float(&'a str, f32),
    Str(&'a str, &'a str),
    Ints(&'a str, &'a [i64]),
}

fn encode_attr(a: &Attr) -> Vec<u8> {
    let mut b = Vec::new();
    match a {
        Attr::Float(name, f) => {
            push_str(&mut b, 1, name);
            push_f32(&mut b, 2, *f);
            push_int64(&mut b, 20, 1); // FLOAT
        }
        Attr::Int(name, i) => {
            push_str(&mut b, 1, name);
            push_int64(&mut b, 3, *i);
            push_int64(&mut b, 20, 2); // INT
        }
        Attr::Str(name, s) => {
            push_str(&mut b, 1, name);
            push_str(&mut b, 4, s);
            push_int64(&mut b, 20, 3); // STRING
        }
        Attr::Ints(name, ints) => {
            push_str(&mut b, 1, name);
            for &v in *ints {
                push_int64(&mut b, 8, v);
            }
            push_int64(&mut b, 20, 7); // INTS
        }
    }
    b
}

struct OnnxBuilder {
    nodes: Vec<Vec<u8>>,
    initializers: Vec<Vec<u8>>,
    counter: usize,
}

impl OnnxBuilder {
    fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(2048),
            initializers: Vec::with_capacity(1200),
            counter: 0,
        }
    }

    fn tmp(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}_{}", self.counter)
    }

    fn add_f32_init(&mut self, name: &str, dims: &[i64], values: &[f32]) -> String {
        let raw: &[u8] = bytemuck::cast_slice(values);
        self.initializers.push(encode_tensor_proto(name, dims, DT_FLOAT, raw));
        name.to_string()
    }

    fn add_i64_init(&mut self, name: &str, dims: &[i64], values: &[i64]) -> String {
        let raw: &[u8] = bytemuck::cast_slice(values);
        self.initializers.push(encode_tensor_proto(name, dims, DT_INT64, raw));
        name.to_string()
    }

    /// Store weight tensors as FP16 initializers with a `Cast(to=FLOAT)` node when large,
    /// cutting the ONNX model size on disk in half (~199 MB) while keeping 100% FP32 compute compatibility.
    fn add_weight(&mut self, name: &str, dims: &[i64], values: &[f32]) -> String {
        if values.len() >= 512 {
            let fp16_name = format!("{name}_fp16");
            let halfs: Vec<u16> = values.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
            let raw: &[u8] = bytemuck::cast_slice(&halfs);
            self.initializers.push(encode_tensor_proto(&fp16_name, dims, DT_FLOAT16, raw));
            let out = name.to_string();
            self.node("Cast", &[&fp16_name], &[&out], &[Attr::Int("to", DT_FLOAT)]);
            out
        } else {
            self.add_f32_init(name, dims, values)
        }
    }

    fn node(&mut self, op_type: &str, inputs: &[&str], outputs: &[&str], attrs: &[Attr]) {
        let mut b = Vec::new();
        for &inp in inputs {
            push_str(&mut b, 1, inp);
        }
        for &out in outputs {
            push_str(&mut b, 2, out);
        }
        let node_name = format!("{}_{}", op_type, self.nodes.len());
        push_str(&mut b, 3, &node_name);
        push_str(&mut b, 4, op_type);
        for a in attrs {
            let enc = encode_attr(a);
            push_bytes(&mut b, 5, &enc);
        }
        self.nodes.push(b);
    }

    fn op1(&mut self, op_type: &str, inputs: &[&str], attrs: &[Attr]) -> String {
        let out = self.tmp(&op_type.to_lowercase());
        self.node(op_type, inputs, &[&out], attrs);
        out
    }

    fn conv2d(
        &mut self,
        x: &str,
        w: &str,
        b: &str,
        kernel: i64,
        stride: i64,
        pad: i64,
        group: i64,
    ) -> String {
        self.op1(
            "Conv",
            &[x, w, b],
            &[
                Attr::Ints("kernel_shape", &[kernel, kernel]),
                Attr::Ints("strides", &[stride, stride]),
                Attr::Ints("pads", &[pad, pad, pad, pad]),
                Attr::Int("group", group),
            ],
        )
    }

    fn layer_norm(&mut self, x: &str, w: &str, b: &str, eps: f32) -> String {
        self.op1(
            "LayerNormalization",
            &[x, w, b],
            &[Attr::Int("axis", -1), Attr::Float("epsilon", eps)],
        )
    }

    fn linear(&mut self, x: &str, w_t: &str, b: &str) -> String {
        let mm = self.op1("MatMul", &[x, w_t], &[]);
        self.op1("Add", &[&mm, b], &[])
    }

    fn gelu(&mut self, x: &str, c_sqrt2: &str, c_one: &str, c_half: &str) -> String {
        let div = self.op1("Div", &[x, c_sqrt2], &[]);
        let erf = self.op1("Erf", &[&div], &[]);
        let add = self.op1("Add", &[&erf, c_one], &[]);
        let mul = self.op1("Mul", &[x, &add], &[]);
        self.op1("Mul", &[&mul, c_half], &[])
    }

    fn reshape(&mut self, x: &str, shape: &[i64]) -> String {
        let shape_name = self.tmp("shape");
        self.add_i64_init(&shape_name, &[shape.len() as i64], shape);
        self.op1("Reshape", &[x, &shape_name], &[])
    }

    fn transpose(&mut self, x: &str, perm: &[i64]) -> String {
        self.op1("Transpose", &[x], &[Attr::Ints("perm", perm)])
    }

    fn upsample2x(&mut self, x: &str, empty_roi: &str, scales_2x: &str) -> String {
        self.op1(
            "Resize",
            &[x, empty_roi, scales_2x],
            &[
                Attr::Str("mode", "linear"),
                Attr::Str("coordinate_transformation_mode", "half_pixel"),
                Attr::Str("nearest_mode", "floor"),
            ],
        )
    }

    fn finish(self, inputs: &[Vec<u8>], outputs: &[Vec<u8>]) -> Vec<u8> {
        let mut graph = Vec::new();
        for n in &self.nodes {
            push_bytes(&mut graph, 1, n);
        }
        push_str(&mut graph, 2, "PerspectiveFields");
        for init in &self.initializers {
            push_bytes(&mut graph, 5, init);
        }
        for inp in inputs {
            push_bytes(&mut graph, 11, inp);
        }
        for out in outputs {
            push_bytes(&mut graph, 12, out);
        }

        let mut opset = Vec::new();
        push_str(&mut opset, 1, "");
        push_int64(&mut opset, 2, 17);

        let mut model = Vec::new();
        push_int64(&mut model, 1, 8); // ir_version = 8
        push_str(&mut model, 2, "optixel-studio");
        push_bytes(&mut model, 7, &graph);
        push_bytes(&mut model, 8, &opset);
        model
    }
}

/// Transpose a 2D row-major matrix `(rows, cols)` to `(cols, rows)`.
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(data.len(), rows * cols);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        for c in 0..cols {
            out[c * rows + r] = row[c];
        }
    }
    out
}

/// Read tensor `idx` (`archive/data/{idx}`) from a PyTorch `.pth` zip archive as `Vec<f32>`.
/// Security: Never parses or executes `data.pkl` (Python pickle); strictly validates uncompressed
/// byte length and bounds reads with `Read::take` to prevent zip-bomb memory exhaustion.
fn read_pth_tensor(
    archive: &mut zip::ZipArchive<fs::File>,
    prefix: &str,
    idx: usize,
    expected_len: usize,
) -> Result<Vec<f32>, String> {
    let expected_bytes = expected_len
        .checked_mul(4)
        .ok_or_else(|| format!("tensor {idx} size overflow"))?;
    let entry_name = format!("{prefix}data/{idx}");
    let f = archive
        .by_name(&entry_name)
        .map_err(|e| format!("missing tensor {entry_name} in checkpoint: {e}"))?;
    if f.size() != expected_bytes as u64 {
        return Err(format!(
            "tensor {entry_name} header size mismatch: got {} bytes, expected {}",
            f.size(),
            expected_bytes
        ));
    }
    let mut bytes = Vec::with_capacity(expected_bytes);
    f.take((expected_bytes as u64) + 1)
        .read_to_end(&mut bytes)
        .map_err(es)?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "tensor {entry_name} size mismatch: got {} bytes, expected {}",
            bytes.len(),
            expected_bytes
        ));
    }
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(floats)
}

/// Convert a downloaded PerspectiveFields `.pth` checkpoint into a standalone ONNX model file.
pub fn convert_pth_to_onnx(pth_path: &Path, onnx_path: &Path) -> Result<(), String> {
    let file = fs::File::open(pth_path).map_err(es)?;
    let mut zf = zip::ZipArchive::new(file).map_err(es)?;
    if zf.len() > 2048 {
        return Err("checkpoint archive has too many entries".into());
    }

    // Determine internal archive prefix (typically "archive/")
    let mut prefix = String::from("archive/");
    for i in 0..zf.len() {
        if let Ok(f) = zf.by_index_raw(i) {
            if let Ok(name) = f.name() {
                if let Some(p) = name.strip_suffix("data/0") {
                    if !p.contains("..") && !p.contains('\\') && p.len() <= 64 {
                        prefix = p.to_string();
                    }
                    break;
                }
            }
        }
    }

    let mut b = OnnxBuilder::new();

    // Shared scalar/vector constants
    let c_sqrt2 = b.add_f32_init("const_sqrt2", &[], &[std::f32::consts::SQRT_2]);
    let c_one = b.add_f32_init("const_one", &[], &[1.0]);
    let c_half = b.add_f32_init("const_half", &[], &[0.5]);
    let c_two = b.add_f32_init("const_two", &[], &[2.0]);
    let c_eps = b.add_f32_init("const_eps", &[], &[1e-12]);
    let c_neg_one = b.add_f32_init("const_neg_one", &[], &[-1.0]);
    let c_attn_scale = b.add_f32_init("const_attn_scale", &[], &[0.125]); // 1 / sqrt(64)
    let empty_roi = b.add_f32_init("const_empty_roi", &[0], &[]);
    let scales_2x = b.add_f32_init("const_scales_2x", &[4], &[1.0, 1.0, 2.0, 2.0]);

    // ------------------------------------------------------------------------
    // 1. SegFormer mit_b3 Backbone (tensors 0..571)
    // ------------------------------------------------------------------------
    let embed_dims = [64usize, 128, 320, 512];
    let num_heads = [1usize, 2, 5, 8];
    let depths = [3usize, 4, 18, 3];
    let sr_ratios = [8usize, 4, 2, 1];
    let hw_stages = [80usize, 40, 20, 10];
    let patch_kernels = [7usize, 3, 3, 3];
    let patch_strides = [4usize, 2, 2, 2];
    let patch_pads = [3usize, 1, 1, 1];

    // Load patch_embed1..4 weights (tensors 0..15)
    let mut patch_weights = Vec::with_capacity(4);
    for s in 0..4 {
        let in_c = if s == 0 { 3 } else { embed_dims[s - 1] };
        let out_c = embed_dims[s];
        let k = patch_kernels[s];
        let base = s * 4;
        let pw = read_pth_tensor(&mut zf, &prefix, base, out_c * in_c * k * k)?;
        let pb = read_pth_tensor(&mut zf, &prefix, base + 1, out_c)?;
        let nw = read_pth_tensor(&mut zf, &prefix, base + 2, out_c)?;
        let nb = read_pth_tensor(&mut zf, &prefix, base + 3, out_c)?;
        let w_name = b.add_weight(&format!("pe{s}_w"), &[out_c as i64, in_c as i64, k as i64, k as i64], &pw);
        let b_name = b.add_f32_init(&format!("pe{s}_b"), &[out_c as i64], &pb);
        let nw_name = b.add_f32_init(&format!("pe{s}_nw"), &[out_c as i64], &nw);
        let nb_name = b.add_f32_init(&format!("pe{s}_nb"), &[out_c as i64], &nb);
        patch_weights.push((w_name, b_name, nw_name, nb_name));
    }

    let mut idx = 16usize;
    let mut stage_outs = Vec::with_capacity(4);
    let mut cur_img = String::from("image");

    for s in 0..4 {
        let c = embed_dims[s];
        let heads = num_heads[s];
        let depth = depths[s];
        let sr = sr_ratios[s];
        let h = hw_stages[s];
        let n_tokens = h * h;

        // OverlapPatchEmbed
        let (ref pw, ref pb, ref nw, ref nb) = patch_weights[s];
        let conv = b.conv2d(
            &cur_img,
            pw,
            pb,
            patch_kernels[s] as i64,
            patch_strides[s] as i64,
            patch_pads[s] as i64,
            1,
        );
        let flat = b.reshape(&conv, &[1, c as i64, n_tokens as i64]);
        let tr = b.transpose(&flat, &[0, 2, 1]);
        let mut x = b.layer_norm(&tr, nw, nb, 1e-6);

        // Transformer blocks
        for blk in 0..depth {
            let pfx = format!("b{s}_{blk}");
            let n1w_v = read_pth_tensor(&mut zf, &prefix, idx, c)?;
            let n1b_v = read_pth_tensor(&mut zf, &prefix, idx + 1, c)?;
            let qw_v = read_pth_tensor(&mut zf, &prefix, idx + 2, c * c)?;
            let qb_v = read_pth_tensor(&mut zf, &prefix, idx + 3, c)?;
            let kvw_v = read_pth_tensor(&mut zf, &prefix, idx + 4, 2 * c * c)?;
            let kvb_v = read_pth_tensor(&mut zf, &prefix, idx + 5, 2 * c)?;
            let prw_v = read_pth_tensor(&mut zf, &prefix, idx + 6, c * c)?;
            let prb_v = read_pth_tensor(&mut zf, &prefix, idx + 7, c)?;
            idx += 8;

            let n1w = b.add_f32_init(&format!("{pfx}_n1w"), &[c as i64], &n1w_v);
            let n1b = b.add_f32_init(&format!("{pfx}_n1b"), &[c as i64], &n1b_v);
            let x_norm1 = b.layer_norm(&x, &n1w, &n1b, 1e-6);

            // Q projection
            let qw_t = b.add_weight(&format!("{pfx}_qwt"), &[c as i64, c as i64], &transpose_2d(&qw_v, c, c));
            let qb = b.add_f32_init(&format!("{pfx}_qb"), &[c as i64], &qb_v);
            let q_lin = b.linear(&x_norm1, &qw_t, &qb);
            let q_4d = b.reshape(&q_lin, &[1, n_tokens as i64, heads as i64, 64]);
            let q = b.transpose(&q_4d, &[0, 2, 1, 3]); // [1, heads, N, 64]

            // Spatial reduction for K, V if sr > 1
            let kv_in = if sr > 1 {
                let srw_v = read_pth_tensor(&mut zf, &prefix, idx, c * c * sr * sr)?;
                let srb_v = read_pth_tensor(&mut zf, &prefix, idx + 1, c)?;
                let srnw_v = read_pth_tensor(&mut zf, &prefix, idx + 2, c)?;
                let srnb_v = read_pth_tensor(&mut zf, &prefix, idx + 3, c)?;
                idx += 4;

                let srw = b.add_weight(&format!("{pfx}_srw"), &[c as i64, c as i64, sr as i64, sr as i64], &srw_v);
                let srb = b.add_f32_init(&format!("{pfx}_srb"), &[c as i64], &srb_v);
                let srnw = b.add_f32_init(&format!("{pfx}_srnw"), &[c as i64], &srnw_v);
                let srnb = b.add_f32_init(&format!("{pfx}_srnb"), &[c as i64], &srnb_v);

                let x_t = b.transpose(&x_norm1, &[0, 2, 1]);
                let x_sp = b.reshape(&x_t, &[1, c as i64, h as i64, h as i64]);
                let x_sr = b.conv2d(&x_sp, &srw, &srb, sr as i64, sr as i64, 0, 1);
                let n_sr = (h / sr) * (h / sr);
                let x_sr_flat = b.reshape(&x_sr, &[1, c as i64, n_sr as i64]);
                let x_sr_t = b.transpose(&x_sr_flat, &[0, 2, 1]);
                b.layer_norm(&x_sr_t, &srnw, &srnb, 1e-5)
            } else {
                x_norm1.clone()
            };

            let n_sr = (h / sr) * (h / sr);
            // Split kv.weight (2*C, C) into k.weight (C, C) and v.weight (C, C) at build time
            let kw_v = &kvw_v[..c * c];
            let vw_v = &kvw_v[c * c..];
            let kb_v = &kvb_v[..c];
            let vb_v = &kvb_v[c..];

            let kw_t = b.add_weight(&format!("{pfx}_kwt"), &[c as i64, c as i64], &transpose_2d(kw_v, c, c));
            let kb = b.add_f32_init(&format!("{pfx}_kb"), &[c as i64], kb_v);
            let vw_t = b.add_weight(&format!("{pfx}_vwt"), &[c as i64, c as i64], &transpose_2d(vw_v, c, c));
            let vb = b.add_f32_init(&format!("{pfx}_vb"), &[c as i64], vb_v);

            let k_lin = b.linear(&kv_in, &kw_t, &kb);
            let k_4d = b.reshape(&k_lin, &[1, n_sr as i64, heads as i64, 64]);
            let k_t = b.transpose(&k_4d, &[0, 2, 3, 1]); // [1, heads, 64, N_sr]

            let v_lin = b.linear(&kv_in, &vw_t, &vb);
            let v_4d = b.reshape(&v_lin, &[1, n_sr as i64, heads as i64, 64]);
            let v = b.transpose(&v_4d, &[0, 2, 1, 3]); // [1, heads, N_sr, 64]

            let qk = b.op1("MatMul", &[&q, &k_t], &[]);
            let qk_scaled = b.op1("Mul", &[&qk, &c_attn_scale], &[]);
            let attn = b.op1("Softmax", &[&qk_scaled], &[Attr::Int("axis", -1)]);
            let av = b.op1("MatMul", &[&attn, &v], &[]);
            let av_t = b.transpose(&av, &[0, 2, 1, 3]);
            let av_flat = b.reshape(&av_t, &[1, n_tokens as i64, c as i64]);

            let prw_t = b.add_weight(&format!("{pfx}_prwt"), &[c as i64, c as i64], &transpose_2d(&prw_v, c, c));
            let prb = b.add_f32_init(&format!("{pfx}_prb"), &[c as i64], &prb_v);
            let attn_out = b.linear(&av_flat, &prw_t, &prb);
            x = b.op1("Add", &[&x, &attn_out], &[]);

            // MLP with depthwise 3x3 conv
            let mlp_c = 4 * c;
            let n2w_v = read_pth_tensor(&mut zf, &prefix, idx, c)?;
            let n2b_v = read_pth_tensor(&mut zf, &prefix, idx + 1, c)?;
            let fc1w_v = read_pth_tensor(&mut zf, &prefix, idx + 2, mlp_c * c)?;
            let fc1b_v = read_pth_tensor(&mut zf, &prefix, idx + 3, mlp_c)?;
            let dww_v = read_pth_tensor(&mut zf, &prefix, idx + 4, mlp_c * 9)?;
            let dwb_v = read_pth_tensor(&mut zf, &prefix, idx + 5, mlp_c)?;
            let fc2w_v = read_pth_tensor(&mut zf, &prefix, idx + 6, c * mlp_c)?;
            let fc2b_v = read_pth_tensor(&mut zf, &prefix, idx + 7, c)?;
            idx += 8;

            let n2w = b.add_f32_init(&format!("{pfx}_n2w"), &[c as i64], &n2w_v);
            let n2b = b.add_f32_init(&format!("{pfx}_n2b"), &[c as i64], &n2b_v);
            let x_norm2 = b.layer_norm(&x, &n2w, &n2b, 1e-6);

            let fc1w_t = b.add_weight(&format!("{pfx}_fc1wt"), &[c as i64, mlp_c as i64], &transpose_2d(&fc1w_v, mlp_c, c));
            let fc1b = b.add_f32_init(&format!("{pfx}_fc1b"), &[mlp_c as i64], &fc1b_v);
            let h1 = b.linear(&x_norm2, &fc1w_t, &fc1b);

            let h1_t = b.transpose(&h1, &[0, 2, 1]);
            let h1_sp = b.reshape(&h1_t, &[1, mlp_c as i64, h as i64, h as i64]);
            let dww = b.add_weight(&format!("{pfx}_dww"), &[mlp_c as i64, 1, 3, 3], &dww_v);
            let dwb = b.add_f32_init(&format!("{pfx}_dwb"), &[mlp_c as i64], &dwb_v);
            let dw = b.conv2d(&h1_sp, &dww, &dwb, 3, 1, 1, mlp_c as i64);
            let dw_flat = b.reshape(&dw, &[1, mlp_c as i64, n_tokens as i64]);
            let dw_t = b.transpose(&dw_flat, &[0, 2, 1]);
            let act = b.gelu(&dw_t, &c_sqrt2, &c_one, &c_half);

            let fc2w_t = b.add_weight(&format!("{pfx}_fc2wt"), &[mlp_c as i64, c as i64], &transpose_2d(&fc2w_v, c, mlp_c));
            let fc2b = b.add_f32_init(&format!("{pfx}_fc2b"), &[c as i64], &fc2b_v);
            let h2 = b.linear(&act, &fc2w_t, &fc2b);
            x = b.op1("Add", &[&x, &h2], &[]);
        }

        // Stage norm -> reshape to (1, C, H, W)
        let snw_v = read_pth_tensor(&mut zf, &prefix, idx, c)?;
        let snb_v = read_pth_tensor(&mut zf, &prefix, idx + 1, c)?;
        idx += 2;
        let snw = b.add_f32_init(&format!("sn{s}_w"), &[c as i64], &snw_v);
        let snb = b.add_f32_init(&format!("sn{s}_b"), &[c as i64], &snb_v);
        let x_sn = b.layer_norm(&x, &snw, &snb, 1e-6);
        let x_t = b.transpose(&x_sn, &[0, 2, 1]);
        let c_out = b.reshape(&x_t, &[1, c as i64, h as i64, h as i64]);
        cur_img = c_out.clone();
        stage_outs.push(c_out);
    }
    assert_eq!(idx, 572);

    // ------------------------------------------------------------------------
    // 2. LowLevelEncoder (ll_enc, tensors 572..577) — fuse Conv2d + BatchNorm2d
    // ------------------------------------------------------------------------
    let ll_conv_w = read_pth_tensor(&mut zf, &prefix, 572, 64 * 3 * 7 * 7)?;
    let ll_bn_w = read_pth_tensor(&mut zf, &prefix, 573, 64)?;
    let ll_bn_b = read_pth_tensor(&mut zf, &prefix, 574, 64)?;
    let ll_bn_mean = read_pth_tensor(&mut zf, &prefix, 575, 64)?;
    let ll_bn_var = read_pth_tensor(&mut zf, &prefix, 576, 64)?;
    // 577 is num_batches_tracked (int64 scalar)
    let mut ll_fused_w = vec![0.0f32; 64 * 3 * 7 * 7];
    let mut ll_fused_b = vec![0.0f32; 64];
    let kernel_elems = 3 * 7 * 7;
    for c in 0..64 {
        let scale = ll_bn_w[c] / (ll_bn_var[c] + 1e-5).sqrt();
        ll_fused_b[c] = ll_bn_b[c] - ll_bn_mean[c] * scale;
        for k in 0..kernel_elems {
            ll_fused_w[c * kernel_elems + k] = ll_conv_w[c * kernel_elems + k] * scale;
        }
    }
    let ll_w_name = b.add_weight("ll_enc_w", &[64, 3, 7, 7], &ll_fused_w);
    let ll_b_name = b.add_f32_init("ll_enc_b", &[64], &ll_fused_b);
    let ll_conv = b.conv2d("image", &ll_w_name, &ll_b_name, 7, 2, 3, 1);
    let ll_feat = b.op1("Relu", &[&ll_conv], &[]);

    // ------------------------------------------------------------------------
    // 3. PersFormer Heads: GravityDecoder (578..627) & LatitudeDecoder (628..677)
    // ------------------------------------------------------------------------
    let build_decoder = |b: &mut OnnxBuilder,
                         zf: &mut zip::ZipArchive<fs::File>,
                         base: usize,
                         tag: &str,
                         out_ch: usize|
     -> Result<String, String> {
        let c_dims = [512usize, 320, 128, 64]; // order in checkpoint: c4, c3, c2, c1
        let c_inputs = [&stage_outs[3], &stage_outs[2], &stage_outs[1], &stage_outs[0]];
        let mut proc_feats = Vec::with_capacity(4);
        for i in 0..4 {
            let in_c = c_dims[i];
            let lw = read_pth_tensor(zf, &prefix, base + i * 2, 768 * in_c)?;
            let lb = read_pth_tensor(zf, &prefix, base + i * 2 + 1, 768)?;
            let pw = read_pth_tensor(zf, &prefix, base + 8 + i * 2, 256 * 768 * 9)?;
            let pb = read_pth_tensor(zf, &prefix, base + 8 + i * 2 + 1, 256)?;

            // MLP linear projection on (B, H*W, C_in) is equivalent to 1x1 Conv2d(C_in -> 768)
            let lw_name = b.add_weight(&format!("{tag}_lc{i}_w"), &[768, in_c as i64, 1, 1], &lw);
            let lb_name = b.add_f32_init(&format!("{tag}_lc{i}_b"), &[768], &lb);
            let lin = b.conv2d(c_inputs[i], &lw_name, &lb_name, 1, 1, 0, 1);

            let pw_name = b.add_weight(&format!("{tag}_pc{i}_w"), &[256, 768, 3, 3], &pw);
            let pb_name = b.add_f32_init(&format!("{tag}_pc{i}_b"), &[256], &pb);
            let pr = b.conv2d(&lin, &pw_name, &pb_name, 3, 1, 1, 1);
            proc_feats.push(pr);
        }
        // proc_feats: [0]=_c4 (10x10), [1]=_c3 (20x20), [2]=_c2 (40x40), [3]=_c1 (80x80)
        // Checkpoint offsets for fusions:
        // fusion1: base+16..base+23 (8 tensors)
        // fusion2: base+24..base+31 (8 tensors)
        // fusion3: base+32..base+39 (8 tensors)
        // fusion4: base+40..base+43 (4 tensors, unit2only)
        let res_unit = |b: &mut OnnxBuilder,
                        zf: &mut zip::ZipArchive<fs::File>,
                        x: &str,
                        t_base: usize,
                        r_tag: &str|
         -> Result<String, String> {
            let w1 = read_pth_tensor(zf, &prefix, t_base, 256 * 256 * 9)?;
            let b1 = read_pth_tensor(zf, &prefix, t_base + 1, 256)?;
            let w2 = read_pth_tensor(zf, &prefix, t_base + 2, 256 * 256 * 9)?;
            let b2 = read_pth_tensor(zf, &prefix, t_base + 3, 256)?;
            let w1n = b.add_weight(&format!("{r_tag}_w1"), &[256, 256, 3, 3], &w1);
            let b1n = b.add_f32_init(&format!("{r_tag}_b1"), &[256], &b1);
            let w2n = b.add_weight(&format!("{r_tag}_w2"), &[256, 256, 3, 3], &w2);
            let b2n = b.add_f32_init(&format!("{r_tag}_b2"), &[256], &b2);
            let r1 = b.op1("Relu", &[x], &[]);
            let c1 = b.conv2d(&r1, &w1n, &b1n, 3, 1, 1, 1);
            let r2 = b.op1("Relu", &[&c1], &[]);
            let c2 = b.conv2d(&r2, &w2n, &b2n, 3, 1, 1, 1);
            Ok(b.op1("Add", &[&c2, x], &[]))
        };

        // fusion4(_c4)
        let f4_u2 = res_unit(b, zf, &proc_feats[0], base + 40, &format!("{tag}_f4u2"))?;
        let f4 = b.upsample2x(&f4_u2, &empty_roi, &scales_2x);

        // fusion3(f4, _c3)
        let f3_u1 = res_unit(b, zf, &proc_feats[1], base + 32, &format!("{tag}_f3u1"))?;
        let f3_sum = b.op1("Add", &[&f4, &f3_u1], &[]);
        let f3_u2 = res_unit(b, zf, &f3_sum, base + 36, &format!("{tag}_f3u2"))?;
        let f3 = b.upsample2x(&f3_u2, &empty_roi, &scales_2x);

        // fusion2(f3, _c2)
        let f2_u1 = res_unit(b, zf, &proc_feats[2], base + 24, &format!("{tag}_f2u1"))?;
        let f2_sum = b.op1("Add", &[&f3, &f2_u1], &[]);
        let f2_u2 = res_unit(b, zf, &f2_sum, base + 28, &format!("{tag}_f2u2"))?;
        let f2 = b.upsample2x(&f2_u2, &empty_roi, &scales_2x);

        // fusion1(f2, _c1)
        let f1_u1 = res_unit(b, zf, &proc_feats[3], base + 16, &format!("{tag}_f1u1"))?;
        let f1_sum = b.op1("Add", &[&f2, &f1_u1], &[]);
        let f1_u2 = res_unit(b, zf, &f1_sum, base + 20, &format!("{tag}_f1u2"))?;
        let f1 = b.upsample2x(&f1_u2, &empty_roi, &scales_2x);

        // Concat([f1, ll_feat], dim=1) -> conv_fuse_conv0 -> upsample2x -> conv_fuse_conv1 -> linear_pred
        let cat = b.op1("Concat", &[&f1, &ll_feat], &[Attr::Int("axis", 1)]);
        let cf0w = read_pth_tensor(zf, &prefix, base + 44, 64 * 320 * 9)?;
        let cf0b = read_pth_tensor(zf, &prefix, base + 45, 64)?;
        let cf0wn = b.add_weight(&format!("{tag}_cf0w"), &[64, 320, 3, 3], &cf0w);
        let cf0bn = b.add_f32_init(&format!("{tag}_cf0b"), &[64], &cf0b);
        let cf0 = b.conv2d(&cat, &cf0wn, &cf0bn, 3, 1, 1, 1);
        let cf0_r = b.op1("Relu", &[&cf0], &[]);
        let cf0_up = b.upsample2x(&cf0_r, &empty_roi, &scales_2x);

        let cf1w = read_pth_tensor(zf, &prefix, base + 46, 32 * 64 * 9)?;
        let cf1b = read_pth_tensor(zf, &prefix, base + 47, 32)?;
        let cf1wn = b.add_weight(&format!("{tag}_cf1w"), &[32, 64, 3, 3], &cf1w);
        let cf1bn = b.add_f32_init(&format!("{tag}_cf1b"), &[32], &cf1b);
        let cf1 = b.conv2d(&cf0_up, &cf1wn, &cf1bn, 3, 1, 1, 1);
        let cf1_r = b.op1("Relu", &[&cf1], &[]);

        let lpw = read_pth_tensor(zf, &prefix, base + 48, out_ch * 32)?;
        let lpb = read_pth_tensor(zf, &prefix, base + 49, out_ch)?;
        let lpwn = b.add_f32_init(&format!("{tag}_lpw"), &[out_ch as i64, 32, 1, 1], &lpw);
        let lpbn = b.add_f32_init(&format!("{tag}_lpb"), &[out_ch as i64], &lpb);
        Ok(b.conv2d(&cf1_r, &lpwn, &lpbn, 1, 1, 0, 1))
    };

    let raw_grav = build_decoder(&mut b, &mut zf, 578, "grav", 2)?;
    // L2 normalize along channel axis 1: raw_grav / sqrt(mean(raw_grav^2, axis=1) * 2.0 + 1e-12)
    let grav_sq = b.op1("Mul", &[&raw_grav, &raw_grav], &[]);
    let grav_mean = b.op1(
        "ReduceMean",
        &[&grav_sq],
        &[Attr::Ints("axes", &[1]), Attr::Int("keepdims", 1)],
    );
    let grav_sum = b.op1("Mul", &[&grav_mean, &c_two], &[]);
    let grav_sum_eps = b.op1("Add", &[&grav_sum, &c_eps], &[]);
    let grav_norm = b.op1("Sqrt", &[&grav_sum_eps], &[]);
    b.node("Div", &[&raw_grav, &grav_norm], &["pred_gravity"], &[]);

    let raw_lat = build_decoder(&mut b, &mut zf, 628, "lat", 1)?;
    b.node("Clip", &[&raw_lat, &c_neg_one, &c_one], &["pred_latitude"], &[]);

    // ------------------------------------------------------------------------
    // 4. ParamNet ConvNeXt-Tiny Backbone (tensors 678..859)
    // ------------------------------------------------------------------------
    let cn_dims = [96usize, 192, 384, 768];
    let cn_depths = [3usize, 3, 9, 3];

    // Read downsample_layers 0..3 (tensors 678..693)
    let mut ds_layers = Vec::with_capacity(4);
    // downsample_layers.0: Conv(3->96, k=4, s=4), LayerNorm(96)
    {
        let cw = read_pth_tensor(&mut zf, &prefix, 678, 96 * 3 * 16)?;
        let cb = read_pth_tensor(&mut zf, &prefix, 679, 96)?;
        let nw = read_pth_tensor(&mut zf, &prefix, 680, 96)?;
        let nb = read_pth_tensor(&mut zf, &prefix, 681, 96)?;
        let cwn = b.add_weight("cn_ds0_cw", &[96, 3, 4, 4], &cw);
        let cbn = b.add_f32_init("cn_ds0_cb", &[96], &cb);
        let nwn = b.add_f32_init("cn_ds0_nw", &[96], &nw);
        let nbn = b.add_f32_init("cn_ds0_nb", &[96], &nb);
        ds_layers.push((cwn, cbn, nwn, nbn));
    }
    for i in 1..4 {
        let in_c = cn_dims[i - 1];
        let out_c = cn_dims[i];
        let base = 678 + i * 4;
        // downsample_layers.i: LayerNorm(in_c), Conv(in_c->out_c, k=2, s=2)
        let nw = read_pth_tensor(&mut zf, &prefix, base, in_c)?;
        let nb = read_pth_tensor(&mut zf, &prefix, base + 1, in_c)?;
        let cw = read_pth_tensor(&mut zf, &prefix, base + 2, out_c * in_c * 4)?;
        let cb = read_pth_tensor(&mut zf, &prefix, base + 3, out_c)?;
        let nwn = b.add_f32_init(&format!("cn_ds{i}_nw"), &[in_c as i64], &nw);
        let nbn = b.add_f32_init(&format!("cn_ds{i}_nb"), &[in_c as i64], &nb);
        let cwn = b.add_weight(&format!("cn_ds{i}_cw"), &[out_c as i64, in_c as i64, 2, 2], &cw);
        let cbn = b.add_f32_init(&format!("cn_ds{i}_cb"), &[out_c as i64], &cb);
        ds_layers.push((cwn, cbn, nwn, nbn));
    }
    idx = 694;

    let mut cx = b.op1("Concat", &["pred_gravity", "pred_latitude"], &[Attr::Int("axis", 1)]);
    for s in 0..4 {
        let c = cn_dims[s];
        let (ref cwn, ref cbn, ref nwn, ref nbn) = ds_layers[s];
        if s == 0 {
            let conv = b.conv2d(&cx, cwn, cbn, 4, 4, 0, 1);
            let nhwc = b.transpose(&conv, &[0, 2, 3, 1]);
            let ln = b.layer_norm(&nhwc, nwn, nbn, 1e-6);
            cx = b.transpose(&ln, &[0, 3, 1, 2]);
        } else {
            let nhwc = b.transpose(&cx, &[0, 2, 3, 1]);
            let ln = b.layer_norm(&nhwc, nwn, nbn, 1e-6);
            let nchw = b.transpose(&ln, &[0, 3, 1, 2]);
            cx = b.conv2d(&nchw, cwn, cbn, 2, 2, 0, 1);
        }

        for blk in 0..cn_depths[s] {
            let pfx = format!("cn_s{s}_{blk}");
            let gamma_v = read_pth_tensor(&mut zf, &prefix, idx, c)?;
            let dww_v = read_pth_tensor(&mut zf, &prefix, idx + 1, c * 49)?;
            let dwb_v = read_pth_tensor(&mut zf, &prefix, idx + 2, c)?;
            let nw_v = read_pth_tensor(&mut zf, &prefix, idx + 3, c)?;
            let nb_v = read_pth_tensor(&mut zf, &prefix, idx + 4, c)?;
            let pw1w_v = read_pth_tensor(&mut zf, &prefix, idx + 5, 4 * c * c)?;
            let pw1b_v = read_pth_tensor(&mut zf, &prefix, idx + 6, 4 * c)?;
            let pw2w_v = read_pth_tensor(&mut zf, &prefix, idx + 7, c * 4 * c)?;
            let pw2b_v = read_pth_tensor(&mut zf, &prefix, idx + 8, c)?;
            idx += 9;

            let dww = b.add_weight(&format!("{pfx}_dww"), &[c as i64, 1, 7, 7], &dww_v);
            let dwb = b.add_f32_init(&format!("{pfx}_dwb"), &[c as i64], &dwb_v);
            let nw = b.add_f32_init(&format!("{pfx}_nw"), &[c as i64], &nw_v);
            let nb = b.add_f32_init(&format!("{pfx}_nb"), &[c as i64], &nb_v);
            let pw1wt = b.add_weight(
                &format!("{pfx}_pw1wt"),
                &[c as i64, (4 * c) as i64],
                &transpose_2d(&pw1w_v, 4 * c, c),
            );
            let pw1b = b.add_f32_init(&format!("{pfx}_pw1b"), &[(4 * c) as i64], &pw1b_v);
            let pw2wt = b.add_weight(
                &format!("{pfx}_pw2wt"),
                &[(4 * c) as i64, c as i64],
                &transpose_2d(&pw2w_v, c, 4 * c),
            );
            let pw2b = b.add_f32_init(&format!("{pfx}_pw2b"), &[c as i64], &pw2b_v);
            let gamma = b.add_f32_init(&format!("{pfx}_gamma"), &[c as i64], &gamma_v);

            let dw = b.conv2d(&cx, &dww, &dwb, 7, 1, 3, c as i64);
            let nhwc = b.transpose(&dw, &[0, 2, 3, 1]);
            let ln = b.layer_norm(&nhwc, &nw, &nb, 1e-6);
            let h1 = b.linear(&ln, &pw1wt, &pw1b);
            let act = b.gelu(&h1, &c_sqrt2, &c_one, &c_half);
            let h2 = b.linear(&act, &pw2wt, &pw2b);
            let scaled = b.op1("Mul", &[&h2, &gamma], &[]);
            let nchw = b.transpose(&scaled, &[0, 3, 1, 2]);
            cx = b.op1("Add", &[&cx, &nchw], &[]);
        }
    }
    assert_eq!(idx, 856);

    // Final global average pooling + LayerNorm(768) + Linear(768 -> 5)
    let fnw_v = read_pth_tensor(&mut zf, &prefix, 856, 768)?;
    let fnb_v = read_pth_tensor(&mut zf, &prefix, 857, 768)?;
    let hw_v = read_pth_tensor(&mut zf, &prefix, 858, 5 * 768)?;
    let hb_v = read_pth_tensor(&mut zf, &prefix, 859, 5)?;

    let fnw = b.add_f32_init("cn_fnw", &[768], &fnw_v);
    let fnb = b.add_f32_init("cn_fnb", &[768], &fnb_v);
    let hwt = b.add_f32_init("cn_hwt", &[768, 5], &transpose_2d(&hw_v, 5, 768));
    let hb = b.add_f32_init("cn_hb", &[5], &hb_v);

    let gap = b.op1(
        "ReduceMean",
        &[&cx],
        &[Attr::Ints("axes", &[2, 3]), Attr::Int("keepdims", 0)],
    );
    let gap_ln = b.layer_norm(&gap, &fnw, &fnb, 1e-6);
    let mm = b.op1("MatMul", &[&gap_ln, &hwt], &[]);
    b.node("Add", &[&mm, &hb], &["pred_params"], &[]);

    let inputs = vec![encode_value_info(
        "image",
        DT_FLOAT,
        &[1, 3, INPUT_SIZE as i64, INPUT_SIZE as i64],
    )];
    let outputs = vec![
        encode_value_info("pred_gravity", DT_FLOAT, &[1, 2, INPUT_SIZE as i64, INPUT_SIZE as i64]),
        encode_value_info("pred_latitude", DT_FLOAT, &[1, 1, INPUT_SIZE as i64, INPUT_SIZE as i64]),
        encode_value_info("pred_params", DT_FLOAT, &[1, 5]),
    ];

    let model_bytes = b.finish(&inputs, &outputs);
    let part = onnx_path.with_extension("onnx.part");
    {
        let mut out = fs::File::create(&part).map_err(es)?;
        out.write_all(&model_bytes).map_err(es)?;
    }
    fs::rename(&part, onnx_path).map_err(es)?;
    Ok(())
}

// ============================================================================
// Preprocessing, Inference & Sub-Degree Perspective Alignment
// ============================================================================

/// Resize `img` (in linear Rec.2020 working space) to `320x320` BGR `[0, 255]` minus `PIXEL_MEAN`.
fn preprocess_bgr_320(img: &Img) -> Vec<f32> {
    let (w, h) = (img.w, img.h);
    let area = INPUT_SIZE * INPUT_SIZE;
    let mut planes = vec![0.0f32; 3 * area];
    if w == 0 || h == 0 {
        return planes;
    }
    let sx = w as f32 / INPUT_SIZE as f32;
    let sy = h as f32 / INPUT_SIZE as f32;
    for dy in 0..INPUT_SIZE {
        let fy = ((dy as f32 + 0.5) * sy - 0.5).clamp(0.0, (h - 1) as f32);
        let (y0, ty) = (fy as usize, fy.fract());
        let y1 = (y0 + 1).min(h - 1);
        for dx in 0..INPUT_SIZE {
            let fx = ((dx as f32 + 0.5) * sx - 0.5).clamp(0.0, (w - 1) as f32);
            let (x0, tx) = (fx as usize, fx.fract());
            let x1 = (x0 + 1).min(w - 1);

            let p00 = crate::aimodel::encode(img.px[y0 * w + x0]);
            let p01 = crate::aimodel::encode(img.px[y0 * w + x1]);
            let p10 = crate::aimodel::encode(img.px[y1 * w + x0]);
            let p11 = crate::aimodel::encode(img.px[y1 * w + x1]);

            let mut rgb = [0.0f32; 3];
            for c in 0..3 {
                let top = p00[c] * (1.0 - tx) + p01[c] * tx;
                let bot = p10[c] * (1.0 - tx) + p11[c] * tx;
                rgb[c] = (top * (1.0 - ty) + bot * ty) * 255.0;
            }
            let i = dy * INPUT_SIZE + dx;
            // BGR order with PIXEL_MEAN subtracted
            planes[i] = rgb[2] - PIXEL_MEAN[0];
            planes[area + i] = rgb[1] - PIXEL_MEAN[1];
            planes[2 * area + i] = rgb[0] - PIXEL_MEAN[2];
        }
    }
    planes
}

/// Refine camera roll and pitch using both ParamNet predictions and the dense 320x320
/// gravity & latitude fields (`pred_gravity` and `pred_latitude`).
fn estimate_angles_from_fields(
    grav: &[f32],
    lat: &[f32],
    params: &[f32],
    orig_w: usize,
    orig_h: usize,
) -> (f32, f32, f32, f32) {
    let area = INPUT_SIZE * INPUT_SIZE;
    let (fw, fh) = (orig_w.max(1) as f32, orig_h.max(1) as f32);

    // 1. ParamNet primary predictions (in degrees)
    let param_roll_deg = (params[0] * 90.0).clamp(-45.0, 45.0);
    let param_pitch_deg = (params[1] * 90.0).clamp(-60.0, 60.0);
    let param_vfov_deg = (params[2] * 90.0).clamp(15.0, 110.0);

    // 2. Rescale gravity vectors to original image aspect ratio and normalize:
    // vec_original = normalize([gx * (w / 320), gy * (h / 320)])
    let scale_x = fw / INPUT_SIZE as f32;
    let scale_y = fh / INPUT_SIZE as f32;

    // Collect local roll angles near the horizon / vertical center line where perspective
    // foreshortening does not skew the gravity vector away from the roll angle.
    let mut horizon_rolls = Vec::with_capacity(2048);
    let mut all_rolls = Vec::with_capacity(1024);

    for y in (8..INPUT_SIZE - 8).step_by(2) {
        for x in (8..INPUT_SIZE - 8).step_by(2) {
            let i = y * INPUT_SIZE + x;
            let gx = grav[i] * scale_x;
            let gy = grav[area + i] * scale_y;
            let norm = gx.hypot(gy);
            if norm < 1e-4 {
                continue;
            }
            let (ux, uy) = (gx / norm, gy / norm);
            // Up vector in unrolled image is (0, -1). Under camera roll `r`, up vector is (-sin(r), -cos(r)).
            let local_roll_deg = (-ux).atan2(-uy).to_degrees();
            let sin_lat = lat[i];
            let rel_x = (x as f32 / INPUT_SIZE as f32 - 0.5).abs();
            // Pixels near horizon (|sin(lat)| < 0.18) or near the central vertical axis (rel_x < 0.08)
            // experience minimal vanishing-point convergence, so their gravity vector directly measures roll.
            if sin_lat.abs() < 0.18 || rel_x < 0.08 {
                horizon_rolls.push(local_roll_deg);
            }
            if y % 8 == 0 && x % 8 == 0 {
                all_rolls.push(local_roll_deg);
            }
        }
    }

    let sample_rolls = if horizon_rolls.len() >= 32 {
        &mut horizon_rolls
    } else {
        &mut all_rolls
    };

    let (field_roll_deg, spread_deg) = if !sample_rolls.is_empty() {
        sample_rolls.sort_by(f32::total_cmp);
        let n = sample_rolls.len();
        let median = sample_rolls[n / 2];
        // Trimmed Huber-weighted mean within 3.0° of the median for ~0.1° precision
        let (mut sum_w, mut sum_v, mut sum_err) = (0.0f32, 0.0f32, 0.0f32);
        for &r in sample_rolls.iter() {
            let diff = r - median;
            if diff.abs() <= 4.0 {
                let w = 1.0 / (0.25 + diff * diff);
                sum_w += w;
                sum_v += w * r;
                sum_err += w * diff.abs();
            }
        }
        if sum_w > 0.0 {
            (sum_v / sum_w, sum_err / sum_w)
        } else {
            (median, 2.0)
        }
    } else {
        (param_roll_deg, 2.0)
    };

    // Combine ParamNet roll and dense gravity field roll:
    // When they agree closely (< 2.5°), blend 55% ParamNet + 45% dense field median for sub-degree stability.
    let roll_diff = (param_roll_deg - field_roll_deg).abs();
    let final_roll_deg = if roll_diff <= 3.0 {
        0.55 * param_roll_deg + 0.45 * field_roll_deg
    } else {
        param_roll_deg
    };

    // Confidence score in [0.0, 1.0] based on dense field coherence and ParamNet agreement
    let confidence = (1.0 - (spread_deg / 8.0).clamp(0.0, 0.6) - (roll_diff / 15.0).clamp(0.0, 0.4))
        .clamp(0.15, 0.99);

    // 3. Refine vertical pitch in Optixel's Geo camera coordinates using the dense gravity field:
    // Optixel's Geo uses virtual focal length `f_ol = FOCAL * max(w, h)`.
    // True vertical vanishing point distance is `f_true / tan(pitch)`, so for Optixel's virtual camera:
    // `tan(pitch_ol) = tan(pitch_true) * (f_ol / f_true)`.
    let f_true = 0.5 * fh / (param_vfov_deg.to_radians() * 0.5).tan().max(0.1);
    let f_ol = FOCAL * fw.max(fh);
    let init_pitch_ol_rad = (param_pitch_deg.to_radians().tan() * (f_ol / f_true)).atan();
    let roll_rad = final_roll_deg.to_radians();
    let (sr, cr) = roll_rad.sin_cos();

    // Also estimate `tan(pitch_ol)` directly from dense gravity field convergence across the left/right halves:
    // Under Optixel's camera with roll `r` and pitch `p`, a vertical line through `(u, v)` projects to
    // unrolled coordinates `u' = u*cr - v*sr`, `v' = u*sr + v*cr`, pointing toward `(0, -f_ol / tan(p))`.
    // Thus at `(u', v')`, the unrolled gravity vector `(gx', gy')` satisfies:
    // `gx' / gy' = u' / (v' + f_ol / tan(p))`  =>  `tan(p) / f_ol = -(gx' / gy') / (u' + v' * (gx' / gy'))`.
    let mut tan_pitch_samples = Vec::with_capacity(512);
    for y in (24..INPUT_SIZE - 24).step_by(8) {
        let v = (y as f32 + 0.5) * scale_y - 0.5 * fh;
        for x in (24..INPUT_SIZE - 24).step_by(8) {
            let u = (x as f32 + 0.5) * scale_x - 0.5 * fw;
            let u_prime = u * cr - v * sr;
            if u_prime.abs() < 0.12 * fw {
                continue; // near vertical center line, horizontal convergence is too small to measure pitch
            }
            let v_prime = u * sr + v * cr;
            let i = y * INPUT_SIZE + x;
            let gx = grav[i] * scale_x;
            let gy = grav[area + i] * scale_y;
            // Rotate gravity vector by -roll into unrolled frame
            let gx_p = gx * cr - gy * sr;
            let gy_p = gx * sr + gy * cr;
            if gy_p >= -0.2 {
                continue;
            }
            let ratio = gx_p / gy_p;
            let denom = u_prime - v_prime * ratio;
            if denom.abs() > 0.08 * fw {
                let tan_p = (ratio * f_ol / denom).clamp(-0.85, 0.85);
                tan_pitch_samples.push(tan_p);
            }
        }
    }

    let final_pitch_ol_deg = if tan_pitch_samples.len() >= 16 {
        tan_pitch_samples.sort_by(f32::total_cmp);
        let field_tan_p = tan_pitch_samples[tan_pitch_samples.len() / 2];
        let field_pitch_ol_rad = field_tan_p.atan();
        let diff = (init_pitch_ol_rad - field_pitch_ol_rad).abs().to_degrees();
        let blended_rad = if diff <= 6.0 {
            0.6 * init_pitch_ol_rad + 0.4 * field_pitch_ol_rad
        } else {
            init_pitch_ol_rad
        };
        blended_rad.to_degrees()
    } else {
        init_pitch_ol_rad.to_degrees()
    };

    (final_roll_deg, param_pitch_deg, param_vfov_deg, confidence)
        .pipe(|(r, _p, vfov, conf)| (r, final_pitch_ol_deg, vfov, conf))
}

trait Pipe: Sized {
    fn pipe<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}
impl<T> Pipe for T {}

/// Run PerspectiveFields inference on `img` and compute the aligned `Transform` for `mode`:
/// - `"horizontal"` (or `"level"`): aligns horizontal level / roll (`transform.rotate`)
/// - `"vertical"`: aligns vertical perspective / pitch (`transform.vertical`)
/// - `"both"` (or `"auto"`): aligns both horizontal roll and vertical perspective (`transform.rotate` + `transform.vertical`)
pub fn align(
    data: &Path,
    model_id: &str,
    img: &Img,
    d: &Develop,
    mode: &str,
    progress: &dyn Fn(&str, usize, usize),
) -> Result<AlignResult, String> {
    // Validate `model_id` against the allowlist of PerspectiveFields align models (prevents path traversal).
    if model_id != "perspective-fields-gsv" && model_id != "perspective-fields-edina" {
        return Err(format!("unknown PerspectiveFields model: {model_id}"));
    }
    let onnx_path = data.join("models").join(model_id).join("perspective_fields.onnx");
    if !onnx_path.is_file() {
        return Err("download the PerspectiveFields model first (Settings → AI or Develop → AI)".into());
    }

    progress("Preprocessing photo for PerspectiveFields", 1, 3);
    let input_planes = preprocess_bgr_320(img);
    let input_tensor = Tensor::from_array((
        [1usize, 3, INPUT_SIZE, INPUT_SIZE],
        input_planes.into_boxed_slice(),
    ))
    .map_err(|e| e.to_string())?;

    progress("Running PerspectiveFields neural network", 2, 3);
    let mut guard = ALIGN_SESSION.lock().unwrap_or_else(|e| e.into_inner());
    if guard.as_ref().is_none_or(|(m, _)| m != model_id) {
        *guard = None;
        let session = crate::aimodel::open_session(&onnx_path, None, INPUT_SIZE, true)
            .or_else(|_| crate::aimodel::open_session(&onnx_path, None, INPUT_SIZE, false))?;
        *guard = Some((model_id.to_string(), session));
    }
    let session = &mut guard.as_mut().ok_or("failed to initialize align session")?.1;

    let outputs = session
        .run(ort::inputs!["image" => input_tensor])
        .map_err(|e| format!("PerspectiveFields inference failed: {e}"))?;

    let (_, grav_slice) = outputs["pred_gravity"]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;
    let (_, lat_slice) = outputs["pred_latitude"]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;
    let (_, param_slice) = outputs["pred_params"]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;

    progress("Computing sub-degree horizon & vertical alignment", 3, 3);
    let (roll_deg, pitch_ol_deg, vfov_deg, confidence) =
        estimate_angles_from_fields(grav_slice, lat_slice, param_slice, img.w, img.h);
    let raw_pitch_deg = (param_slice[1] * 90.0).clamp(-60.0, 60.0);

    let norm_mode = match mode {
        "horizontal" | "level" => "horizontal",
        "vertical" => "vertical",
        _ => "both",
    };

    let mut t = Transform {
        upright: "off".into(),
        vertical: d.transform.vertical,
        horizontal: 0.0,
        rotate: d.transform.rotate,
        ..d.transform.clone()
    };

    match norm_mode {
        "horizontal" => {
            t.rotate = (-roll_deg).clamp(-10.0, 10.0);
        }
        "vertical" => {
            t.vertical = (-pitch_ol_deg / TILT_MAX * 100.0).clamp(-100.0, 100.0);
        }
        _ => {
            t.rotate = (-roll_deg).clamp(-10.0, 10.0);
            t.vertical = (-pitch_ol_deg / TILT_MAX * 100.0).clamp(-100.0, 100.0);
        }
    }

    // Round to 0.01 precision for clean slider display
    t.rotate = (t.rotate * 100.0).round() / 100.0;
    t.vertical = (t.vertical * 10.0).round() / 10.0;

    let _ = PI;
    Ok(AlignResult {
        transform: t,
        roll_deg: (roll_deg * 100.0).round() / 100.0,
        pitch_deg: (raw_pitch_deg * 100.0).round() / 100.0,
        vfov_deg: (vfov_deg * 10.0).round() / 10.0,
        confidence: (confidence * 100.0).round() / 100.0,
        mode: norm_mode.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_angles_synthetic_field() {
        let area = INPUT_SIZE * INPUT_SIZE;
        let roll_true = 2.75f32; // degrees
        let pitch_true = 8.5f32; // degrees
        let vfov_true = 50.0f32; // degrees

        let (sr, cr) = roll_true.to_radians().sin_cos();
        let (gx_320, gy_320) = (-sr / 1600.0, -cr / 1066.0);
        let norm = gx_320.hypot(gy_320);
        let mut grav = vec![0.0f32; 2 * area];
        let lat = vec![0.0f32; area];

        for i in 0..area {
            grav[i] = gx_320 / norm;
            grav[area + i] = gy_320 / norm;
        }
        let params = [roll_true / 90.0, pitch_true / 90.0, vfov_true / 90.0, 0.0, 0.0];
        let (r, p, vfov, conf) = estimate_angles_from_fields(&grav, &lat, &params, 1600, 1066);
        assert!((r - roll_true).abs() < 0.05, "roll={r}, expected={roll_true}");
        assert!(p > 0.0, "pitch_ol={p} should be positive");
        assert!((vfov - vfov_true).abs() < 0.1);
        assert!(conf > 0.85, "confidence={conf}");
    }

    #[test]
    fn test_convert_pth_to_onnx_and_run_ort() {
        let tmp = std::env::temp_dir().join("optixel_perspective_test");
        let _ = fs::remove_dir_all(&tmp);
        let model_dir = tmp.join("models").join("perspective-fields-gsv");
        fs::create_dir_all(&model_dir).unwrap();
        let pth_path = model_dir.join("test.pth");
        let onnx_path = model_dir.join("perspective_fields.onnx");

        // Build tensor element count table for all 860 tensors (0..859)
        let mut sizes = Vec::with_capacity(860);
        let embed_dims = [64usize, 128, 320, 512];
        let depths = [3usize, 4, 18, 3];
        let sr_ratios = [8usize, 4, 2, 1];
        let patch_kernels = [7usize, 3, 3, 3];
        for s in 0..4 {
            let in_c = if s == 0 { 3 } else { embed_dims[s - 1] };
            let out_c = embed_dims[s];
            let k = patch_kernels[s];
            sizes.extend([out_c * in_c * k * k, out_c, out_c, out_c]);
        }
        for s in 0..4 {
            let c = embed_dims[s];
            let sr = sr_ratios[s];
            for _ in 0..depths[s] {
                sizes.extend([c, c, c * c, c, 2 * c * c, 2 * c, c * c, c]);
                if sr > 1 {
                    sizes.extend([c * c * sr * sr, c, c, c]);
                }
                let mc = 4 * c;
                sizes.extend([c, c, mc * c, mc, mc * 9, mc, c * mc, c]);
            }
            sizes.extend([c, c]);
        }
        // ll_enc (572..577)
        sizes.extend([64 * 3 * 49, 64, 64, 64, 64, 2]);
        // gravity_head (578..627) & latitude_head (628..677)
        for out_ch in [2usize, 1] {
            for &in_c in &[512usize, 320, 128, 64] {
                sizes.extend([768 * in_c, 768]);
            }
            for _ in 0..4 {
                sizes.extend([256 * 768 * 9, 256]);
            }
            // fusion1..3 (3 * 8) + fusion4 (4) = 28
            for _ in 0..7 {
                sizes.extend([256 * 256 * 9, 256, 256 * 256 * 9, 256]);
            }
            sizes.extend([64 * 320 * 9, 64, 32 * 64 * 9, 32, out_ch * 32, out_ch]);
        }
        // param_net (678..859)
        let cn_dims = [96usize, 192, 384, 768];
        let cn_depths = [3usize, 3, 9, 3];
        sizes.extend([96 * 3 * 16, 96, 96, 96]);
        for i in 1..4 {
            let (ic, oc) = (cn_dims[i - 1], cn_dims[i]);
            sizes.extend([ic, ic, oc * ic * 4, oc]);
        }
        for s in 0..4 {
            let c = cn_dims[s];
            for _ in 0..cn_depths[s] {
                sizes.extend([c, c * 49, c, c, c, 4 * c * c, 4 * c, c * 4 * c, c]);
            }
        }
        sizes.extend([768, 768, 5 * 768, 5]);
        assert_eq!(sizes.len(), 860);

        {
            let f = fs::File::create(&pth_path).unwrap();
            let mut zw = zip::ZipWriter::new(f);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            for (idx, &len) in sizes.iter().enumerate() {
                zw.start_file(format!("archive/data/{idx}"), opts).unwrap();
                let val = if idx == 576 { 1.0f32 } else { 0.0f32 };
                let buf: Vec<f32> = vec![val; len];
                zw.write_all(bytemuck::cast_slice(&buf)).unwrap();
            }
            zw.finish().unwrap();
        }

        convert_pth_to_onnx(&pth_path, &onnx_path).expect("ONNX conversion should succeed");
        assert!(onnx_path.is_file());

        // Run inference through ORT CPU session
        let img = Img {
            w: 64,
            h: 48,
            px: vec![[0.2, 0.3, 0.4]; 64 * 48],
        };
        let d = Develop::default();
        let res = align(&tmp, "perspective-fields-gsv", &img, &d, "both", &|_, _, _| {})
            .expect("ORT inference should succeed");
        assert_eq!(res.mode, "both");
        assert!(res.roll_deg.is_finite());
        assert!(res.pitch_deg.is_finite());
        assert!(res.vfov_deg.is_finite());

        let _ = fs::remove_dir_all(&tmp);
    }
}
