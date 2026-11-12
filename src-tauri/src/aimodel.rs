//! AI denoise: downloadable ONNX models, a hardware check that recommends the best one the GPU can
//! run, and tiled inference through ONNX Runtime (DirectML on Windows GPUs, CPU as the fallback).
//!
//! Models are third-party ONNX exports hosted on Hugging Face; every file is size- and SHA-256-checked
//! after download, and nothing is downloaded until the user asks.
//! ponytail: three models, ranked by hand; add a row to `MODELS` for a new one. Inference runs a tile at
//! a time on one session; batching tiles would be the next speed-up if a big GPU sits idle.

use crate::develop::{Img, M2020_709, es, mul, srgb_oetf, srgb_to_working};
use ort::{ep, session::Session, value::Tensor};
use serde::Serialize;
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::Mutex,
    time::Duration,
};

type Px = [f32; 3];

/// (file name, bytes, SHA-256)
type File = (&'static str, u64, &'static str);

struct Spec {
    id: &'static str,
    /// denoise | inpaint
    kind: &'static str,
    name: &'static str,
    blurb: &'static str,
    license: &'static str,
    /// Hugging Face repository the files come from
    repo: &'static str,
    files: &'static [File],
    /// the smallest GPU memory (MB) it runs well in, at the smallest tile
    min_vram_mb: u32,
    /// higher is better quality; the recommendation is the best one that fits
    rank: u32,
    /// spatial size must be a multiple of this
    multiple: usize,
    /// names of the model's batch, height and width axes: fixing them to the tile size is what keeps DirectML from crashing
    dims: Option<[&'static str; 3]>,
    /// run on the GPU (DirectML); false when its layers are not supported there
    gpu: bool,
    /// the single file is a zip: unpack it next to itself once it is verified
    unzip: bool,
}

const MODELS: &[Spec] = &[
    Spec {
        id: "scunet-real-psnr",
        kind: "denoise",
        name: "SCUNet · Real-world (clean)",
        blurb: "Blind denoiser for real camera noise of any level. The best all-round quality here: smooth, faithful, keeps detail.",
        license: "Apache-2.0 (SCUNet, Zhang et al.)",
        repo: "Heliosoph/scunet-onnx",
        files: &[
            ("scunet_color_real_psnr.onnx", 3_798_678, "231be201ab413dbc999d7951caa9844846b93a12a40a41e037d6b5888ed4e88c"),
            ("scunet_color_real_psnr.onnx.data", 73_138_176, "98825ea1210b641c71e5f052f582c70c49fd44b35387ebe2c034268c17df3feb"),
        ],
        min_vram_mb: 3000,
        rank: if cfg!(target_os = "macos") { 5 } else { 30 }, // CoreML cannot run its attention layers: 9 GB of RAM per 256 px tile on the CPU
        multiple: 64,
        dims: Some(["batch", "height", "width"]),
        gpu: true,
        unzip: false,
    },
    Spec {
        id: "scunet-real-gan",
        kind: "denoise",
        name: "SCUNet · Real-world (sharp)",
        blurb: "Same network trained for crisper, more textured results; can look a little processed at high strength.",
        license: "Apache-2.0 (SCUNet, Zhang et al.)",
        repo: "Heliosoph/scunet-onnx",
        files: &[
            ("scunet_color_real_gan.onnx", 3_798_249, "e2bf014ad711f99322f2a94304431ffd6766b79ff00bcebb87d1a4f71c127b2d"),
            ("scunet_color_real_gan.onnx.data", 73_138_176, "1e8d582046abf4a773215445c248d0f54751a57d69d2a2312ec59f1150256b79"),
        ],
        min_vram_mb: 3000,
        rank: if cfg!(target_os = "macos") { 4 } else { 20 },
        multiple: 64,
        dims: Some(["batch", "height", "width"]),
        gpu: true,
        unzip: false,
    },
    Spec {
        id: "nafnet-sidd-w32-fp16",
        kind: "denoise",
        name: "NAFNet · Fast (half precision)",
        blurb: "Small and fast, for smaller GPUs or a quick pass. Trained on smartphone noise, so it is gentler on camera files.",
        license: "MIT (NAFNet, Megvii)",
        repo: "tomfl/obscura-models",
        files: &[("nafnet-sidd-w32-fp16.onnx", 58_957_563, "74bb136d6e78ebab8490c7db29b5bda37c270bd382a03990c66d9cc17271f581")],
        min_vram_mb: 1500,
        rank: 10,
        multiple: 32,
        dims: Some(["", "h", "w"]),
        gpu: true,
        unzip: false,
    },
    Spec {
        id: "lama-big",
        kind: "inpaint",
        name: "LaMa · Object removal",
        blurb: "Fills in what is behind a removed object, from the rest of the picture: textures, lines and edges continue across the gap.",
        license: "Apache-2.0 (LaMa, Suvorov et al.)",
        repo: "Carve/LaMa-ONNX",
        files: &[("lama_fp32.onnx", 208_044_816, "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6")],
        min_vram_mb: 2000,
        rank: 100,
        multiple: 8,
        dims: None,
        gpu: false,
        unzip: false,
    },
    Spec {
        id: "moebius",
        kind: "inpaint",
        name: "Moebius (RemovalModel) · Object removal",
        blurb: "0.2B lightweight diffusion inpainting model with learned embedding conditioning for realistic object removal without text prompts.",
        license: "Apache-2.0 (Moebius, Duan & Xu et al.; ONNX conversion by SimonW)",
        repo: "simonw/Moebius-ONNX",
        files: &[
            ("unet.onnx", 906_698_976, "e3f90f52f72378339b990459fadb29a68d3c7b5c6851545ba42774f489160b08"),
            ("vae_decoder.onnx", 198_078_671, "d90ef0b7f6c8c8b7234459c8b449d70be0033bf1576c842e8b9991baf3934280"),
            ("vae_encoder.onnx", 136_757_093, "b8b81d41e757222a0707665ba9d826703987855e5bed056036b90b988968042f"),
        ],
        min_vram_mb: 2500,
        rank: 150,
        multiple: 8,
        dims: None,
        gpu: true,
        unzip: false,
    },
    Spec {
        id: "sd15-inpaint",
        kind: "generate",
        name: "Stable Diffusion 1.5 · Generative fill",
        blurb: "A diffusion model that paints new content into the hole: it can invent what was hidden, follow a short description (\"grass\", \"sky\"), and handles big holes. Slower than LaMa.",
        license: "CreativeML OpenRAIL-M (Stable Diffusion 1.5 Inpainting, RunwayML; ONNX conversion by RanaLLC)",
        repo: "RanaLLC/stable-diffusion-v1-5-inpainting-onnx-fp16",
        files: &[
            ("unet/model.onnx", 1720173976, "a3726e1944ec2fbc9596096cd001520aea789367f329360ee34eb520c751de16"),
            ("text_encoder/model.onnx", 246368629, "8af516fb184866f8caed192ca3bf0636ef9d40d4f7799c05600cae431d18d8d1"),
            ("vae_encoder/model.onnx", 68430178, "f0da9070d007def0d6a4e7c10a21462bb6172e460ef2587c3fe91191397b4cea"),
            ("vae_decoder/model.onnx", 99093889, "ce0ffca3fcfd0a2729a88d8d90dabf35c66d237404997e0461d52e46ac7a91bc"),
            ("tokenizer/vocab.json", 1059962, ""),
            ("tokenizer/merges.txt", 524619, ""),
        ],
        min_vram_mb: 4000,
        rank: 200,
        multiple: 8,
        dims: None,
        gpu: true,
        unzip: false,
    },
    Spec {
        id: "hyperswap-1a",
        kind: "faceswap",
        name: "HyperSwap 1A · Face swap",
        blurb: "Puts the face from another picture onto the faces in this photo, keeping their pose, expression and light. FaceFusion's default swapper, with its YOLOFace detector and ArcFace recogniser.",
        license: "FaceFusion models (research use; check FaceFusion's terms before commercial use)",
        repo: "facefusion/models-3.0.0",
        files: &[
            (crate::faceswap::DETECT, 12_659_761, "821cdbb1e65fbbabdde7dd0933f754797a343e56fd962729c61ffcefcd135929"),
            (crate::faceswap::RECOGNISE, 174_388_474, "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1"),
            ("@facefusion/models-3.3.0/hyperswap_1a_256.onnx", 402_742_682, "c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add"),
        ],
        min_vram_mb: 0,
        rank: 10,
        multiple: 1,
        dims: None,
        gpu: false,
        unzip: false,
    },
    Spec {
        id: "arcface-faces",
        kind: "faces",
        name: "YOLOFace + ArcFace · Face recognition",
        blurb: "Finds the faces in your photos and groups them into people (Library → People), so you can name them and search by person. FaceFusion's YOLOFace detector and the ArcFace w600k recogniser.",
        license: "FaceFusion models / InsightFace ArcFace (non-commercial research use)",
        repo: "facefusion/models-3.0.0",
        files: &[
            (crate::faceswap::DETECT, 12_659_761, "821cdbb1e65fbbabdde7dd0933f754797a343e56fd962729c61ffcefcd135929"),
            (crate::faceswap::RECOGNISE, 174_388_474, "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1"),
        ],
        min_vram_mb: 0,
        rank: 10,
        multiple: 1,
        dims: None,
        gpu: false,
        unzip: false,
    },
    Spec {
        id: "face-landmarks-2dfan4",
        kind: "landmarks",
        name: "2DFAN4 · Eye landmarks (closed eyes)",
        blurb: "68 landmarks per face found by the face scan: the eyelids' opening tells open eyes from a blink. Used by culling (reject closed eyes) and the focus panel. Needs YOLOFace + ArcFace too.",
        license: "FaceFusion models / FAN, Bulat & Tzimiropoulos (research use; check FaceFusion's terms before commercial use)",
        repo: "facefusion/models-3.0.0",
        files: &[("2dfan4.onnx", 97_904_803, "678c6fa539d52335a31c980feefdf4a6e02d781d83dce00af8a894f114557285")],
        min_vram_mb: 0,
        rank: 10,
        multiple: 1,
        dims: None,
        gpu: false,
        unzip: false,
    },
    Spec {
        id: "sam2.1-hiera-tiny",
        kind: "segment",
        name: "SAM 2.1 Hiera-Tiny · Click to select",
        blurb: "Meta's Segment Anything 2.1, smallest: click an object and it is selected, edges and all (Alt-click takes a part away). Fast enough to follow every click.",
        license: "Apache-2.0 (SAM 2.1, Meta; ONNX by onnx-community)",
        repo: "onnx-community/sam2.1-hiera-tiny-ONNX",
        files: &[
            ("onnx/vision_encoder.onnx", 354_238, "4f30aacd3aaefbca81a0b7fe4c1fc96345570ea0a6f80ced599493d1b3be2e8c"),
            ("onnx/vision_encoder.onnx_data", 134_084_864, "e83df9866a5afe68ea7f0f721f18f65137fc3acbf0da1c74e946d363e09c69cc"),
            ("onnx/prompt_encoder_mask_decoder.onnx", 213_114, "874414704c5d686db7d206a35f6e15d26563d50c8c4468fccc6739bd7e491dcf"),
            ("onnx/prompt_encoder_mask_decoder.onnx_data", 20_958_208, "e9874d900dd4134ed60eab1e97910327c2419e0b2954485d8fd6e7f1a1470f47"),
        ],
        min_vram_mb: 0,
        rank: 10,
        multiple: 1,
        dims: None,
        gpu: true,
        unzip: false,
    },
    Spec {
        id: "sam2.1-hiera-small",
        kind: "segment",
        name: "SAM 2.1 Hiera-Small · Click to select",
        blurb: "The same click selection with a larger image encoder: cleaner edges on fine detail (hair, branches), a little slower per photo.",
        license: "Apache-2.0 (SAM 2.1, Meta; ONNX by onnx-community)",
        repo: "onnx-community/sam2.1-hiera-small-ONNX",
        files: &[
            ("onnx/vision_encoder.onnx", 467_440, "aacf1f7137bb6fffcf6bf166abcfabe28f57a76059254f3fb611c4a64a208119"),
            ("onnx/vision_encoder.onnx_data", 162_476_288, "260fd1f0a34e72a3dc79a739e563b4facc0ba75504818b433a1f808e66637456"),
            ("onnx/prompt_encoder_mask_decoder.onnx", 213_114, "079c59b261f723ff5c6a125e69b0170a957b21c58738c28d2b0394ecd0587d7f"),
            ("onnx/prompt_encoder_mask_decoder.onnx_data", 20_958_208, "f9e59a584ab8ced21fa812c211bc01084204db1c9e92a5ef4fb3a49972b4e864"),
        ],
        min_vram_mb: 1000,
        rank: 20,
        multiple: 1,
        dims: None,
        gpu: true,
        unzip: false,
    },
    Spec {
        id: "sam3-fp16",
        kind: "concept",
        name: "SAM 3 · Select by name (full, FP16)",
        blurb: "Meta's Segment Anything 3: type what to select (\"sky\", \"person\", \"red car\") and it finds every instance; also what Auto masks use for subject, sky, people and background. The most reliable on vague words.",
        license: "SAM License (SAM 3, Meta; ONNX export by PIXL): read its terms, trade controls included",
        repo: "xuckless/pixl-sam3-onnx",
        files: &[
            ("sam3_image_encoder_fp16.onnx", 908_717_454, "709e68c56ba6c1e40c0cfa3381b06b29fc358b1281c706e02c1ff45d71dead8c"),
            ("sam3_text_encoder_fp16.onnx", 706_988_044, "2145c1a55137ea64076b441b48a79feaeebdcc6932154c9f702e54702a96eab9"),
            ("sam3_decoder.onnx", 129_868_227, "f6749e7b7a558ab3b16b0ba75075ebe7086003d86d129f8fa831dea727f01a28"),
            ("@openai/clip-vit-base-patch32/vocab.json", 862_328, ""),
            ("@openai/clip-vit-base-patch32/merges.txt", 524_657, ""),
        ],
        min_vram_mb: 4000,
        rank: 30,
        multiple: 1,
        dims: None,
        gpu: false,
        unzip: false,
    },
    Spec {
        id: "sam3-tiny",
        kind: "concept",
        name: "SAM 3 Tiny (EfficientSAM3) · Select by name",
        blurb: "SAM 3 distilled into a small EfficientViT encoder: the same select-by-name and Auto masks in a fifth of the size. Sure of clear nouns; can be confidently wrong on vague phrases where SAM 3 finds nothing.",
        license: "SAM License (distilled from SAM 3, Meta; EfficientSAM3, Zeng et al.; ONNX export by PIXL)",
        repo: "xuckless/pixl-efficientsam3-onnx",
        files: &[
            ("efficientsam3_ev_m_image_encoder.onnx", 85_436_582, "2ed08306579138120dbf9f1110fd31784d1c20fe1b67fa2187a0a2a6d7c82729"),
            ("efficientsam3_ev_m_text_encoder.onnx", 169_171_689, "3065ea6116f21f6e22bc70a0c041e5da9ae05fdec7b48df953394f5657609590"),
            ("efficientsam3_ev_m_decoder.onnx", 129_858_624, "16accc22a07ee99c73ef8a14e286ec567f94c116e0d8efa1a3a6c0bc9a2596a8"),
            ("@openai/clip-vit-base-patch32/vocab.json", 862_328, ""),
            ("@openai/clip-vit-base-patch32/merges.txt", 524_657, ""),
        ],
        min_vram_mb: 0,
        rank: 20,
        multiple: 1,
        dims: None,
        gpu: false,
        unzip: false,
    },
    Spec {
        id: "perspective-fields-gsv",
        kind: "align",
        name: "PerspectiveFields · Architecture & Street (GSV)",
        blurb: "Dense gravity & roll vector field + ParamNet camera calibration (SegFormer-B3 + ConvNeXt, ~0.1°–0.5° accuracy). Trained on Google Street View for architecture, building facades, and street scenes.",
        license: "MIT (PerspectiveFields, Jin et al. CVPR 2023)",
        repo: "spaces/jinlinyi/PerspectiveFields",
        files: &[
            ("models/paramnet_gsv_rpf.pth", 418_607_881, "3965db8523f573f8037b43696cd08d1e63a498218e28a4ee7ebd50b373bbd56d"),
        ],
        min_vram_mb: 1000,
        rank: 25,
        multiple: 32,
        dims: None,
        gpu: true,
        unzip: false,
    },
    Spec {
        id: "perspective-fields-edina",
        kind: "align",
        name: "PerspectiveFields · Room, Landscape & General (EDINA)",
        blurb: "Dense gravity & latitude vector fields + ParamNet camera calibration (SegFormer-B3 + ConvNeXt, ~0.1°–0.5° accuracy). Trained on 360Cities & EDINA for interior rooms, landscapes, and general photos.",
        license: "MIT (PerspectiveFields, Jin et al. CVPR 2023)",
        repo: "spaces/jinlinyi/PerspectiveFields",
        files: &[
            ("models/paramnet_360cities_edina_rpf.pth", 837_147_876, "58fe0285fe5d4592aec77e9ef57ac94273deb79bcb99f27d08bae68a2d1efc4a"),
        ],
        min_vram_mb: 1000,
        rank: 20,
        multiple: 32,
        dims: None,
        gpu: true,
        unzip: false,
    },
];

/// A file name may name its own repository, "@owner/repo/file", stored as "file": (repository, local name).
fn source<'a>(s: &'a Spec, name: &'a str) -> (&'a str, &'a str) {
    name.strip_prefix('@').and_then(|n| n.rsplit_once('/')).unwrap_or((s.repo, name))
}

fn spec(id: &str) -> Result<&'static Spec, String> {
    MODELS
        .iter()
        .find(|m| {
            m.id == id
                || (m.id == "moebius" && (id == "moebius-removal" || id == "Moebius" || id == "moebius-onnx"))
                || (m.id == "perspective-fields-gsv" && id == "perspective-fields")
        })
        .ok_or_else(|| format!("unknown AI model: {id}"))
}

fn dir(data: &Path, s: &Spec) -> PathBuf {
    data.join("models").join(s.id)
}

fn installed(data: &Path, s: &Spec) -> bool {
    if s.kind == "align" {
        return dir(data, s).join("perspective_fields.onnx").is_file();
    }
    if s.unzip {
        return dir(data, s).join(".unpacked").is_file();
    }
    s.files.iter().all(|(n, size, _)| fs::metadata(dir(data, s).join(source(s, n).1)).is_ok_and(|m| m.len() == *size))
}

/// Where model `id`'s `file` is, when the model is installed.
pub fn model_file(data: &Path, id: &str, file: &str) -> Option<PathBuf> {
    let s = spec(id).ok()?;
    installed(data, s).then(|| dir(data, s).join(file))
}

#[derive(Serialize, Clone)]
pub struct Hardware {
    pub gpu: Option<String>,
    pub vram_mb: Option<u32>,
    /// what inference will run on
    pub device: String,
}

/// ONNX Runtime execution provider: DirectML on Windows, CoreML (GPU + Neural Engine) on Apple Silicon.
#[cfg(target_os = "macos")]
pub const ACCEL_NAME: &str = "CoreML: GPU + Neural Engine";
#[cfg(windows)]
pub const ACCEL_NAME: &str = "DirectML";
#[cfg(not(any(windows, target_os = "macos")))]
pub const ACCEL_NAME: &str = "CPU";

#[cfg(target_os = "macos")]
pub fn accel(key: &str) -> ort::ep::ExecutionProviderDispatch {
    use ort::ep::coreml::{ComputeUnits, ModelFormat};
    // OPTIXEL_COREML=all|ane|gpu|cpu picks the compute units (default: all)
    let units = match std::env::var("OPTIXEL_COREML").as_deref() {
        Ok("ane") => ComputeUnits::CPUAndNeuralEngine,
        Ok("gpu") => ComputeUnits::CPUAndGPU,
        Ok("cpu") => ComputeUnits::CPUOnly,
        _ => ComputeUnits::All,
    };
    // the compiled model is cached, so only the first run pays the CoreML compile. CoreML keys the cache on the
    // model alone, so each input shape needs its own folder or it reuses a stale one ("output_features has no value")
    let cache = crate::data_dir().join("coreml-cache").join(key);
    ep::CoreML::default()
        .with_model_format(ModelFormat::MLProgram)
        .with_compute_units(units)
        .with_static_input_shapes(true)
        .with_model_cache_dir(cache.to_string_lossy())
        .build()
}
#[cfg(windows)]
pub fn accel(_key: &str) -> ort::ep::ExecutionProviderDispatch {
    ep::DirectML::default().build()
}
// ponytail: Linux runs on the CPU; add the CUDA EP when someone needs Linux GPU inference
#[cfg(not(any(windows, target_os = "macos")))]
pub fn accel(_key: &str) -> ort::ep::ExecutionProviderDispatch {
    ep::CPU::default().build()
}

/// Apple Silicon shares one memory pool between CPU, GPU and Neural Engine, so system RAM is the model budget.
#[cfg(target_os = "macos")]
fn unified_memory_mb() -> Option<u32> {
    let o = Command::new("sysctl").args(["-n", "hw.memsize"]).output().ok()?;
    let bytes: u64 = String::from_utf8_lossy(&o.stdout).trim().parse().ok()?;
    Some((bytes / 1_048_576) as u32)
}
#[cfg(not(target_os = "macos"))]
fn unified_memory_mb() -> Option<u32> {
    None
}

/// The GPU's name and memory. Memory comes from `nvidia-smi` (NVIDIA); other vendors report none, which
/// the recommendation treats as a small GPU.
pub fn hardware() -> Hardware {
    let smi = Command::new("nvidia-smi").args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]).output().ok().filter(|o| o.status.success());
    let (name, vram) = smi
        .and_then(|o| {
            let t = String::from_utf8_lossy(&o.stdout).lines().next()?.to_string();
            let (n, m) = t.rsplit_once(',')?;
            Some((n.trim().to_string(), m.trim().parse::<u32>().ok()))
        })
        .unzip();
    let gpu = name.or_else(crate::gpu::detect);
    let device = if gpu.is_some() { format!("GPU ({ACCEL_NAME})") } else { "CPU".to_string() };
    Hardware { gpu, vram_mb: vram.flatten().or_else(unified_memory_mb), device }
}

/// Tile edge for the GPU memory. Bigger tiles have fewer seams, but DirectML slows down 20x and returns
/// garbage above ~600 px for SCUNet (measured on an RTX 5070), so 512 is the ceiling whatever the memory.
fn tile_for(vram_mb: Option<u32>) -> usize {
    match vram_mb {
        Some(v) if v >= 6_000 => 512,
        _ => 384,
    }
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: &'static str,
    pub kind: &'static str,
    pub name: &'static str,
    pub blurb: &'static str,
    pub license: &'static str,
    pub source: String,
    pub size_mb: f32,
    pub installed: bool,
    pub fits: bool,
    pub recommended: bool,
    /// why it is or is not the pick for this machine
    pub note: String,
}

pub fn models(data: &Path, hw: &Hardware) -> Vec<ModelInfo> {
    // no GPU, or a GPU that does not report its memory: only what a small card runs well
    let vram = hw.vram_mb.unwrap_or(if hw.gpu.is_some() { 2000 } else { 0 });
    // the best model of each kind that the GPU memory allows
    let best_of = |kind: &str| MODELS.iter().filter(|m| m.kind == kind && m.min_vram_mb <= vram).max_by_key(|m| m.rank).or_else(|| MODELS.iter().filter(|m| m.kind == kind).min_by_key(|m| m.min_vram_mb));
    let mut out: Vec<ModelInfo> = MODELS
        .iter()
        .map(|m| {
            let fits = m.min_vram_mb <= vram;
            let recommended = best_of(m.kind).is_some_and(|b| b.id == m.id);
            let note = if recommended {
                match hw.vram_mb {
                    Some(v) if m.kind == "generate" => format!("Fits {} ({:.0} GB): about 10 seconds per removal on the GPU, and the most natural result of the removal models.", hw.gpu.as_deref().unwrap_or("your GPU"), v as f32 / 1024.0),
                    _ if m.kind == "generate" => "Best match for this machine (needs about 4 GB of GPU memory).".to_string(),
                    _ if m.id == "moebius" => "A fast 0.2B diffusion model running on GPU with learned conditioning for realistic object removal.".to_string(),
                    _ if m.kind == "faceswap" => "Runs on the CPU: about a second per face.".to_string(),
                    _ if m.kind == "faces" => "Runs on the CPU: a fraction of a second per photo, plus decoding it.".to_string(),
                    _ if m.kind == "align" => "PyTorch-to-ONNX exported on download (~200 MB FP16 ONNX). Sub-degree (~0.1°–0.5°) horizon & vertical alignment.".to_string(),
                    _ if m.kind == "inpaint" => "A fast alternative that runs on the CPU (about 2 seconds per removal) and only continues what is around the object.".to_string(),
                    Some(v) => format!("Best quality for {}: runs at {} px tiles in {:.0} GB.", hw.gpu.as_deref().unwrap_or("your GPU"), tile_for(Some(v)), v as f32 / 1024.0),
                    None => "Best match for this machine (GPU memory could not be read).".to_string(),
                }
            } else if fits {
                if m.id == "moebius" {
                    "Runs on your GPU; fast 0.2B diffusion inpainting with learned conditioning.".to_string()
                } else if m.kind == "align" {
                    "PyTorch-to-ONNX exported on download (~200 MB FP16 ONNX). Ideal for interior room scenes and landscapes.".to_string()
                } else {
                    "Runs on your GPU; lower quality than the recommended one.".to_string()
                }
            } else {
                format!("Needs about {:.0} GB of GPU memory.", m.min_vram_mb as f32 / 1024.0)
            };
            let size_mb = if m.kind == "align" {
                fs::metadata(dir(data, m).join("perspective_fields.onnx"))
                    .map(|meta| meta.len() as f32 / 1_048_576.0)
                    .unwrap_or(199.5)
            } else {
                m.files.iter().map(|f| f.1).sum::<u64>() as f32 / 1_048_576.0
            };
            ModelInfo {
                id: m.id,
                kind: m.kind,
                name: m.name,
                blurb: m.blurb,
                license: m.license,
                source: format!("huggingface.co/{}", m.repo),
                size_mb,
                installed: installed(data, m),
                fits,
                recommended,
                note,
            }
        })
        .collect();
    out.sort_by_key(|m| (m.kind, std::cmp::Reverse(MODELS.iter().find(|s| s.id == m.id).map_or(0, |s| s.rank))));
    out
}

/// The hole for a selection drawn on the preview frame (`alpha`, `pw` × `ph`), at the photo's own size `w` × `h`.
pub fn hole_from_alpha(alpha: &[f32], pw: usize, ph: usize, w: usize, h: usize) -> Option<Hole> {
    let (mut x0, mut y0, mut x1, mut y1) = (pw, ph, 0, 0);
    for y in 0..ph {
        for x in 0..pw {
            if alpha[y * pw + x] > 0.05 {
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            }
        }
    }
    if x1 <= x0 {
        return None;
    }
    let (sx, sy) = (w as f32 / pw as f32, h as f32 / ph as f32);
    let (fx0, fy0) = (((x0 as f32 - 1.0) * sx).max(0.0) as usize, ((y0 as f32 - 1.0) * sy).max(0.0) as usize);
    let (fx1, fy1) = ((((x1 + 1) as f32) * sx).ceil().min(w as f32) as usize, (((y1 + 1) as f32) * sy).ceil().min(h as f32) as usize);
    let (hw, hh) = (fx1 - fx0, fy1 - fy0);
    let mut a = Vec::with_capacity(hw * hh);
    for y in 0..hh {
        // bilinear, pixel centres
        let fy = (((fy0 + y) as f32 + 0.5) / sy - 0.5).clamp(0.0, ph as f32 - 1.0);
        let (iy, ty) = (fy as usize, fy.fract());
        for x in 0..hw {
            let fx = (((fx0 + x) as f32 + 0.5) / sx - 0.5).clamp(0.0, pw as f32 - 1.0);
            let (ix, tx) = (fx as usize, fx.fract());
            let (ix1, iy1) = ((ix + 1).min(pw - 1), (iy + 1).min(ph - 1));
            let top = alpha[iy * pw + ix] * (1.0 - tx) + alpha[iy * pw + ix1] * tx;
            let bot = alpha[iy1 * pw + ix] * (1.0 - tx) + alpha[iy1 * pw + ix1] * tx;
            a.push(top * (1.0 - ty) + bot * ty);
        }
    }
    Some(Hole { x0: fx0, y0: fy0, w: hw, h: hh, a })
}

pub fn remove(data: &Path, id: &str) -> Result<(), String> {
    let s = spec(id)?;
    let d = dir(data, s);
    if d.is_dir() {
        fs::remove_dir_all(d).map_err(es)?;
    }
    // Also clean up any CoreML cache entries matching this model
    let cache_dir = data.join("coreml-cache");
    if cache_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(id) || s.files.iter().any(|f| name.starts_with(Path::new(f.0).file_stem().and_then(|s| s.to_str()).unwrap_or(id))) {
                    let _ = fs::remove_dir_all(entry.path());
                }
            }
        }
    }
    *SESSION.lock().unwrap() = None;
    *SD.lock().unwrap() = None;
    crate::moebius::reset();
    crate::perspective::reset();
    crate::segment::reset();
    Ok(())
}

/// Download a model's files, checking each one's size and SHA-256. `progress(done, total)` in bytes.
pub fn download(data: &Path, id: &str, progress: &(dyn Fn(u64, u64) + Sync)) -> Result<(), String> {
    let s = spec(id)?;
    let d = dir(data, s);
    fs::create_dir_all(&d).map_err(es)?;
    let total: u64 = s.files.iter().map(|f| f.1).sum();
    let mut done = 0u64;
    if installed(data, s) {
        progress(total, total);
        return Ok(());
    }
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_connect(Some(Duration::from_secs(15))).timeout_per_read(Some(Duration::from_secs(60))).build().into();
    for (file, size, sha) in s.files {
        let (repo, name) = source(s, file);
        let out = d.join(name);
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent).map_err(es)?;
        }
        if fs::metadata(&out).is_ok_and(|m| m.len() == *size) {
            done += size;
            progress(done, total);
            continue;
        }
        let url = format!("https://huggingface.co/{repo}/resolve/main/{name}");
        let mut r = agent.get(&url).call().map_err(|e| format!("{name}: {e}"))?;
        let part = out.with_extension("part");
        let res = (|| -> Result<(), String> {
            let mut f = fs::File::create(&part).map_err(es)?;
            let mut ctx = ring::digest::Context::new(&ring::digest::SHA256);
            let mut reader = r.body_mut().as_reader();
            let mut buf = vec![0u8; 1 << 20];
            let mut got = 0u64;
            loop {
                let n = reader.read(&mut buf).map_err(es)?;
                if n == 0 {
                    break;
                }
                ctx.update(&buf[..n]);
                f.write_all(&buf[..n]).map_err(es)?;
                got += n as u64;
                progress(done + got, total);
            }
            let hex: String = ctx.finish().as_ref().iter().map(|b| format!("{b:02x}")).collect();
            if got != *size || (!sha.is_empty() && hex != *sha) {
                return Err(format!("{name} did not download correctly (size or checksum mismatch); try again"));
            }
            Ok(())
        })();
        if let Err(e) = res {
            let _ = fs::remove_file(&part);
            return Err(e);
        }
        fs::rename(&part, &out).map_err(es)?;
        done += size;
    }
    if s.kind == "align" {
        let pth_path = d.join(source(s, s.files[0].0).1);
        let onnx_path = d.join("perspective_fields.onnx");
        crate::perspective::convert_pth_to_onnx(&pth_path, &onnx_path)?;
        let _ = fs::remove_file(pth_path);
    }
    if s.unzip {
        let zip_path = d.join(s.files[0].0);
        let mut z = zip::ZipArchive::new(fs::File::open(&zip_path).map_err(es)?).map_err(es)?;
        for i in 0..z.len() {
            let mut f = z.by_index(i).map_err(es)?;
            // flat, by file name only: nothing in the archive can land outside the model folder
            let Some(name) = f.enclosed_name().and_then(|p| p.file_name().map(|n| n.to_owned())) else { continue };
            if f.is_dir() {
                continue;
            }
            let mut out = fs::File::create(d.join(name)).map_err(es)?;
            std::io::copy(&mut f, &mut out).map_err(es)?;
        }
        fs::write(d.join(".unpacked"), b"ok").map_err(es)?;
        let _ = fs::remove_file(zip_path);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/// The loaded model and the tile size its input shape was fixed to.
static SESSION: Mutex<Option<(String, usize, Session)>> = Mutex::new(None);

pub(crate) fn open_session(path: &Path, dims: Option<[&str; 3]>, tile: usize, gpu: bool) -> Result<Session, String> {
    let env = crate::sdfill::env()?;
    let fail = |e: String| format!("could not load the AI model: {e}");
    // DirectML needs sequential execution and no memory-pattern reuse; without a usable GPU it falls back to the CPU
    let b = Session::builder(env).map_err(|e| fail(e.to_string()))?;
    // OPTIXEL_AI_CPU=1 skips the GPU (to tell a GPU-driver problem from a model problem)
    let b = if !gpu || std::env::var_os("OPTIXEL_AI_CPU").is_some() {
        b.with_auto_device(ort::session::builder::AutoDevicePolicy::PreferCPU).map_err(|e| fail(e.to_string()))?
    } else {
        b.with_execution_providers([accel(&format!("{}-{tile}", path.file_stem().and_then(|f| f.to_str()).unwrap_or("model")))]).map_err(|e| fail(e.to_string()))?
    };
    let b = b.with_memory_pattern(false).map_err(|e| fail(e.to_string()))?;
    // on Apple Silicon the efficiency cores straggle statically-split work: use the performance cores only
    let b = match perf_cores() { Some(n) => b.with_intra_threads(n).map_err(|e| fail(e.to_string()))?, None => b };
    let mut b = b.with_parallel_execution(false).map_err(|e| fail(e.to_string()))?;
    if let Some([batch, height, width]) = dims {
        for (name, size) in [(batch, 1), (height, tile as i64), (width, tile as i64)].into_iter().filter(|(n, _)| !n.is_empty()) {
            b = b.with_dimension_override(name, size).map_err(|e| fail(e.to_string()))?;
        }
    }
    b.commit_from_file(path).map_err(|e| fail(e.to_string()))
}

#[cfg(target_os = "macos")]
pub(crate) fn perf_cores() -> Option<usize> {
    let o = Command::new("sysctl").args(["-n", "hw.perflevel0.physicalcpu"]).output().ok()?;
    String::from_utf8_lossy(&o.stdout).trim().parse().ok().filter(|&n| n > 0)
}
#[cfg(not(target_os = "macos"))]
pub(crate) fn perf_cores() -> Option<usize> {
    None
}

/// Periodic reflection of `i` into 0..n.
fn reflect(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let p = 2 * (n as isize - 1);
    let m = i.rem_euclid(p);
    (if m < n as isize { m } else { p - m }) as usize
}

pub(crate) fn encode(p: Px) -> Px {
    mul(&M2020_709, p).map(|v| srgb_oetf(v.clamp(0.0, 1.0)))
}

fn decode(e: Px) -> Px {
    srgb_to_working(e)
}

/// Denoise `img` with the model, a tile at a time. The network sees the clipped picture sRGB-encoded in
/// `space` (working → its linear RGB, and back; None: sRGB itself): for a RAW, its own camera RGB
/// (`libraw::camera_space`), where each channel's noise is still the sensor's own and not yet mixed by the
/// colour matrix. What it removes is subtracted from the original, so highlights above white and the colour
/// space are kept. `strength` 0-100 blends the result in. `tick(done, total)` returns false to cancel.
pub fn denoise_in(
    data: &Path,
    id: &str,
    img: Img,
    strength: f32,
    vram_mb: Option<u32>,
    space: Option<(crate::develop::M3, crate::develop::M3)>,
    tick: &dyn Fn(usize, usize) -> bool,
) -> Result<Img, String> {
    let (to, from) = space.unwrap_or((M2020_709, crate::develop::M709_2020));
    let encode = |p: Px| mul(&to, p).map(|v| srgb_oetf(v.clamp(0.0, 1.0)));
    let decode = |e: Px| mul(&from, e.map(crate::develop::srgb_eotf));
    let s = spec(id)?;
    if !installed(data, s) {
        return Err("download the AI model first (AI tab)".into());
    }
    let (w, h) = (img.w, img.h);
    let tile = tile_for(vram_mb).max(s.multiple * 4) / s.multiple * s.multiple;
    let mut guard = SESSION.lock().unwrap();
    if guard.as_ref().is_none_or(|(m, t, _)| m != id || *t != tile) {
        *guard = None; // free the old model's GPU memory first: only one model is ever loaded
        *SD.lock().unwrap() = None;
        crate::moebius::reset();
        *guard = Some((id.to_string(), tile, open_session(&dir(data, s).join(s.files[0].0), s.dims, tile, s.gpu)?));
    }
    let session = &mut guard.as_mut().unwrap().2;
    let (in_name, out_name) = (session.inputs()[0].name().to_string(), session.outputs()[0].name().to_string());

    let ov = s.multiple.max(32);
    let step = tile - 2 * ov;
    let amt = (strength / 100.0).clamp(0.0, 1.0);
    let (nx, ny) = (w.div_ceil(step), h.div_ceil(step));
    let mut out = img.px.clone();
    for ty in 0..ny {
        for tx in 0..nx {
            if !tick(ty * nx + tx, nx * ny) {
                return Err("cancelled".into());
            }
            let (x0, y0) = (tx * step, ty * step);
            // the tile's pixels, mirrored past the frame, as sRGB planes
            let mut enc_px: Vec<Px> = Vec::with_capacity(tile * tile);
            for y in 0..tile {
                let sy = reflect(y0 as isize + y as isize - ov as isize, h);
                for x in 0..tile {
                    let sx = reflect(x0 as isize + x as isize - ov as isize, w);
                    enc_px.push(encode(img.px[sy * w + sx]));
                }
            }
            let mut planes = vec![0f32; 3 * tile * tile];
            let area = tile * tile;
            for (i, p) in enc_px.iter().enumerate() {
                planes[i] = p[0];
                planes[area + i] = p[1];
                planes[2 * area + i] = p[2];
            }
            let input = Tensor::from_array(([1usize, 3, tile, tile], planes)).map_err(|e| format!("AI input: {e}"))?;
            let result = session.run(ort::inputs![in_name.as_str() => input]).map_err(|e| format!("AI model failed: {e}"))?;
            let (shape, den) = result[out_name.as_str()].try_extract_tensor::<f32>().map_err(|e| format!("AI output: {e}"))?;
            if shape.len() != 4 || shape[2] as usize != tile || shape[3] as usize != tile {
                return Err(format!("the AI model returned an unexpected shape {shape:?}"));
            }
            let (xe, ye) = ((x0 + step).min(w), (y0 + step).min(h));
            for y in y0..ye {
                let row = (y - y0 + ov) * tile;
                for x in x0..xe {
                    let i = row + (x - x0 + ov);
                    let d = decode([den[i], den[area + i], den[2 * area + i]]);
                    let e = decode(enc_px[i]);
                    let p = &mut out[y * w + x];
                    for c in 0..3 {
                        p[c] += amt * (d[c] - e[c]);
                    }
                }
            }
        }
    }
    Ok(Img { w, h, px: out })
}

// ---------------------------------------------------------------------------
// Object removal (inpainting)
// ---------------------------------------------------------------------------

/// LaMa: image and mask in, the filled picture out (interleaved 0-1 at 512).
fn lama(data: &Path, s: &Spec, rgb_in: &[f32], m_in: Vec<f32>) -> Result<Vec<f32>, String> {
    let id = s.id;
    let mut guard = SESSION.lock().unwrap();
    if guard.as_ref().is_none_or(|(m, t, _)| m != id || *t != LAMA) {
        *guard = None;
        *SD.lock().unwrap() = None;
        crate::moebius::reset();
        *guard = Some((id.to_string(), LAMA, open_session(&dir(data, s).join(s.files[0].0), s.dims, LAMA, s.gpu)?));
    }
    let session = &mut guard.as_mut().unwrap().2;
    let names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
    let out_name = session.outputs()[0].name().to_string();
    let area = LAMA * LAMA;
    let mut planes = vec![0f32; 3 * area];
    for i in 0..area {
        for c in 0..3 {
            planes[c * area + i] = rgb_in[i * 3 + c];
        }
    }
    let image_t = Tensor::from_array(([1usize, 3, LAMA, LAMA], planes)).map_err(|e| format!("inpaint input: {e}"))?;
    let mask_t = Tensor::from_array(([1usize, 1, LAMA, LAMA], m_in)).map_err(|e| format!("inpaint input: {e}"))?;
    let (first, second) = if names[0].to_lowercase().contains("mask") { (mask_t, image_t) } else { (image_t, mask_t) };
    let result = session.run(ort::inputs![names[0].as_str() => first, names[1].as_str() => second]).map_err(|e| format!("the removal model failed: {e}"))?;
    let (shape, out) = result[out_name.as_str()].try_extract_tensor::<f32>().map_err(|e| format!("inpaint output: {e}"))?;
    if shape.len() != 4 || shape[1] != 3 || shape[2] as usize != LAMA || shape[3] as usize != LAMA {
        return Err(format!("the removal model returned an unexpected shape {shape:?}"));
    }
    // the model answers in 0-255 or 0-1
    let scale = if out.iter().copied().fold(0.0f32, f32::max) > 2.0 { 1.0 / 255.0 } else { 1.0 };
    let mut filled = vec![0f32; area * 3];
    for i in 0..area {
        for c in 0..3 {
            filled[i * 3 + c] = (out[c * area + i] * scale).clamp(0.0, 1.0);
        }
    }
    drop(result);
    Ok(filled)
}

static SD: Mutex<Option<crate::sdfill::Sd>> = Mutex::new(None);

pub(crate) fn clear_other_sessions() {
    *SESSION.lock().unwrap() = None;
    *SD.lock().unwrap() = None;
}

/// Stable Diffusion (about 2 GB of GPU memory). Like every model it stays loaded until a different one is needed.
fn generate(data: &Path, s: &Spec, rgb_in: &[f32], m_in: &[f32], prompt: &str, tick: &dyn Fn(usize, usize) -> bool) -> Result<Vec<f32>, String> {
    let mut session = SESSION.lock().unwrap(); // always SESSION before SD, so jobs cannot deadlock
    let mut guard = SD.lock().unwrap();
    if guard.as_ref().is_none_or(|m| m.id() != s.id) {
        *guard = None;
        *session = None; // one model on the GPU at a time
        crate::moebius::reset();
        *guard = Some(crate::sdfill::Sd::load(crate::sdfill::env()?, &dir(data, s), s.id)?);
    }
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
    // Removing an object means painting plain background: without a prompt the model likes to invent new objects
    let prompt = if prompt.trim().is_empty() { "empty background, clean, seamless, natural, photo" } else { prompt };
    let negative = "person, people, object, animal, text, logo, watermark, pattern, ornament, frame, deformed, blurry, cartoon";
    guard.as_mut().unwrap().fill(rgb_in, m_in, prompt, negative, 25, seed, tick)
}

/// The object to remove: its matte (0-1, from `develop::object_matte`) inside a pixel window of the frame.
pub struct Hole {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
    pub a: Vec<f32>,
}

/// LaMa's fixed input size.
const LAMA: usize = 512;

/// Binary mask of the matte over the whole frame, grown by `grow` pixels (a square structuring element).
fn hole_mask(hole: &Hole, w: usize, h: usize, grow: usize) -> Vec<bool> {
    let mut m = vec![false; w * h];
    for y in 0..hole.h {
        for x in 0..hole.w {
            if hole.a[y * hole.w + x] >= 0.5 {
                m[(hole.y0 + y) * w + hole.x0 + x] = true;
            }
        }
    }
    if grow == 0 {
        return m;
    }
    // separable running maximum within the window grown by `grow`
    let (x0, y0) = (hole.x0.saturating_sub(grow), hole.y0.saturating_sub(grow));
    let (x1, y1) = ((hole.x0 + hole.w + grow).min(w), (hole.y0 + hole.h + grow).min(h));
    let mut tmp = m.clone();
    for y in y0..y1 {
        for x in x0..x1 {
            tmp[y * w + x] = (x.saturating_sub(grow)..(x + grow + 1).min(w)).any(|k| m[y * w + k]);
        }
    }
    for y in y0..y1 {
        for x in x0..x1 {
            m[y * w + x] = (y.saturating_sub(grow)..(y + grow + 1).min(h)).any(|k| tmp[k * w + x]);
        }
    }
    m
}

pub(crate) fn resized(plane: Vec<f32>, w: usize, h: usize, channels: usize, nw: usize, nh: usize) -> Vec<f32> {
    use image::{ImageBuffer, Rgb, imageops::{FilterType, resize}};
    if (w, h) == (nw, nh) {
        return plane;
    }
    if channels == 3 {
        let img: ImageBuffer<Rgb<f32>, Vec<f32>> = ImageBuffer::from_raw(w as u32, h as u32, plane).expect("rgb plane");
        resize(&img, nw as u32, nh as u32, FilterType::Triangle).into_raw()
    } else {
        let img: ImageBuffer<image::Luma<f32>, Vec<f32>> = ImageBuffer::from_raw(w as u32, h as u32, plane).expect("luma plane");
        resize(&img, nw as u32, nh as u32, FilterType::Triangle).into_raw()
    }
}

/// Inpaint the hole and return the resulting patch with original coordinates and soft alpha mask.
pub fn inpaint_patch(data: &Path, id: &str, img: &Img, hole: &Hole, grow: usize, prompt: &str, tick: &dyn Fn(usize, usize) -> bool) -> Result<crate::develop::Patch, String> {
    let s = spec(id)?;
    if !installed(data, s) {
        return Err("download the object removal model first (AI tab)".into());
    }
    let (w, h) = (img.w, img.h);
    let mask = hole_mask(hole, w, h, grow);
    let (mut bx0, mut by0, mut bx1, mut by1) = (w, h, 0, 0);
    for y in hole.y0.saturating_sub(grow)..(hole.y0 + hole.h + grow).min(h) {
        for x in hole.x0.saturating_sub(grow)..(hole.x0 + hole.w + grow).min(w) {
            if mask[y * w + x] {
                (bx0, by0, bx1, by1) = (bx0.min(x), by0.min(y), bx1.max(x + 1), by1.max(y + 1));
            }
        }
    }
    if bx1 <= bx0 {
        return Err("nothing is selected to remove".into());
    }
    let side = (((bx1 - bx0).max(by1 - by0) as f32 * 1.7) as usize).max(LAMA).min(w.min(h));
    let (cx, cy) = ((bx0 + bx1) / 2, (by0 + by1) / 2);
    let (sx, sy) = (cx.saturating_sub(side / 2).min(w - side), cy.saturating_sub(side / 2).min(h - side));

    // the crop as sRGB 0-1 and the hole mask, then both at 512
    let mut rgb = Vec::with_capacity(side * side * 3);
    let mut m = Vec::with_capacity(side * side);
    for y in 0..side {
        for x in 0..side {
            let i = (sy + y) * w + sx + x;
            rgb.extend(encode(img.px[i]));
            m.push(if mask[i] { 1.0 } else { 0.0 });
        }
    }
    let rgb_in = resized(rgb, side, side, 3, LAMA, LAMA);
    let m_in: Vec<f32> = resized(m, side, side, 1, LAMA, LAMA).into_iter().map(|v| if v >= 0.5 { 1.0 } else { 0.0 }).collect();

    let filled = if s.kind == "generate" {
        generate(data, s, &rgb_in, &m_in, prompt, tick)?
    } else if s.id == "moebius" || s.id.starts_with("moebius") {
        crate::moebius::fill(data, s.id, s.gpu, &rgb_in, &m_in, tick)?
    } else {
        if !tick(1, 3) {
            return Err("cancelled".into());
        }
        let res = lama(data, s, &rgb_in, m_in)?;
        if !tick(2, 3) {
            return Err("cancelled".into());
        }
        res
    };
    let filled = resized(filled, LAMA, LAMA, 3, side, side);
    let _ = tick(3, 3);

    // paste the hole back, feathered a little so the seam disappears
    let sigma = (side as f32 / LAMA as f32).max(1.0) * 1.5;
    let hard: Vec<f32> = (0..side * side).map(|k| if mask[(sy + k / side) * w + sx + k % side] { 1.0 } else { 0.0 }).collect();
    let soft = crate::develop::blur(&hard, side, side, sigma);
    let mut patch_data = Vec::with_capacity(side * side);
    for k in 0..side * side {
        let f = (soft[k] * 2.0).min(1.0).max(hard[k]);
        let new_px = decode([filled[k * 3], filled[k * 3 + 1], filled[k * 3 + 2]]);
        patch_data.push([new_px[0], new_px[1], new_px[2], f]);
    }
    Ok(crate::develop::Patch {
        sx,
        sy,
        side,
        orig_w: w,
        orig_h: h,
        data: patch_data,
    })
}

/// Face swap: the biggest face of `source` onto every face of `img`, one patch per face.
pub fn face_swap(data: &Path, id: &str, img: &Img, source: &Img, tick: &dyn Fn(&str, usize, usize) -> bool) -> Result<Vec<crate::develop::Patch>, String> {
    let s = spec(id)?;
    if s.kind != "faceswap" || !installed(data, s) {
        return Err("download the face swap model first (Settings → AI models)".into());
    }
    crate::faceswap::run(&dir(data, s), img, source, tick)
}

/// Where the face detector and ArcFace recogniser are: the People model, or the face swap one (which has both).
pub fn faces_dir(data: &Path) -> Result<PathBuf, String> {
    ["arcface-faces", "hyperswap-1a"]
        .iter()
        .filter_map(|id| spec(id).ok())
        .find(|s| installed(data, s))
        .map(|s| dir(data, s))
        .ok_or_else(|| "download the face recognition model first (Settings → AI models)".into())
}

/// Fill the hole with LaMa. The model works on a 512 px square, so a square around the object (with its
/// surroundings as context) is cut out at the photo's own resolution when it is small enough, and shrunk to 512
/// when not (big objects come back softer). Only the hole is replaced, with a feathered edge.
pub fn inpaint(data: &Path, id: &str, img: Img, hole: &Hole, grow: usize, prompt: &str, tick: &dyn Fn(usize, usize) -> bool) -> Result<Img, String> {
    let patch = inpaint_patch(data, id, &img, hole, grow, prompt, tick)?;
    let mut px = img.px;
    crate::develop::composite_patch(&mut px, img.w, img.h, &patch, [1.0, 1.0, 1.0]);
    Ok(Img { w: img.w, h: img.h, px })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs OPTIXEL_TEST_AI=1: downloads the model into the app's data folder and denoises a noisy picture.
    #[test]
    fn denoise_end_to_end() {
        if std::env::var_os("OPTIXEL_TEST_AI").is_none() {
            return;
        }
        let data = crate::data_dir();
        let id = std::env::var("OPTIXEL_TEST_MODEL").unwrap_or_else(|_| "scunet-real-psnr".into());
        let t0 = std::time::Instant::now();
        download(&data, &id, &|d, t| eprint!("\rdownload {d}/{t}")).unwrap();
        eprintln!("\ndownloaded/verified in {:?}", t0.elapsed());
        // a smooth sRGB picture with blocks, plus noise
        let (w, h) = (900usize, 700usize);
        let mut s = 7u32;
        let mut rnd = move || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32
        };
        let clean: Vec<f32> = (0..w * h).map(|i| 0.25 + 0.5 * ((i % w) as f32 / w as f32) + if (i / w / 100 + i % w / 100) % 2 == 0 { 0.1 } else { 0.0 }).collect();
        let noisy: Vec<Px> = clean
            .iter()
            .map(|&c| {
                let n = |r: &mut dyn FnMut() -> f32| (r() + r() + r() + r() - 2.0) * 0.17; // ~gaussian, sigma ~0.1 of 0.58 range
                srgb_to_working([(c + n(&mut rnd)).clamp(0.0, 1.0), (c + n(&mut rnd)).clamp(0.0, 1.0), (c + n(&mut rnd)).clamp(0.0, 1.0)])
            })
            .collect();
        let err = |px: &[Px]| -> f32 {
            let mut e = 0.0;
            for (p, &c) in px.iter().zip(&clean) {
                let g = encode(*p);
                e += ((g[0] - c).abs() + (g[1] - c).abs() + (g[2] - c).abs()) / 3.0;
            }
            e / px.len() as f32
        };
        let before = err(&noisy);
        let t0 = std::time::Instant::now();
        let out = denoise_in(&data, &id, Img { w, h, px: noisy }, 100.0, Some(12_000), None, &|d, t| {
            eprint!("\rtile {}/{t}", d + 1);
            true
        })
        .unwrap();
        let after = err(&out.px);
        eprintln!("\n{id}: mean error vs clean {before:.4} -> {after:.4} in {:?}", t0.elapsed());
        assert!(after < before * 0.6, "denoise did not reduce the noise enough");
    }

    /// Needs OPTIXEL_TEST_AI=1: downloads LaMa and removes a blob from a striped, gradient picture.
    #[test]
    fn inpaint_end_to_end() {
        if std::env::var_os("OPTIXEL_TEST_AI").is_none() {
            return;
        }
        let data = crate::data_dir();
        download(&data, "lama-big", &|d, t| eprint!("download {d}/{t}")).unwrap();
        for (w, h, r) in [(1000usize, 800usize, 90usize), (2400, 1800, 380)] {
            let bg = |x: usize, y: usize| 0.3 + 0.3 * (x as f32 / w as f32) + if (x / 40 + y / 40) % 2 == 0 { 0.08 } else { 0.0 };
            let (cx, cy) = (w / 2, h / 2);
            let inside = |x: usize, y: usize| ((x as f32 - cx as f32).powi(2) + (y as f32 - cy as f32).powi(2)).sqrt() < r as f32;
            let px: Vec<Px> = (0..w * h).map(|i| { let (x, y) = (i % w, i / w); let v = if inside(x, y) { 0.95 } else { bg(x, y) }; srgb_to_working([v, v * 0.4, 0.1]) }).collect();
            let a: Vec<f32> = (0..(2 * r) * (2 * r)).map(|k| if inside(cx - r + k % (2 * r), cy - r + k / (2 * r)) { 1.0 } else { 0.0 }).collect();
            let hole = Hole { x0: cx - r, y0: cy - r, w: 2 * r, h: 2 * r, a };
            let t0 = std::time::Instant::now();
            let out = inpaint(&data, "lama-big", Img { w, h, px }, &hole, 4, "", &|_, _| true).unwrap();
            let mut err = 0.0;
            let mut n = 0;
            for y in 0..h {
                for x in 0..w {
                    if inside(x, y) {
                        let g = encode(out.px[y * w + x]);
                        err += (g[0] - bg(x, y)).abs();
                        n += 1;
                    }
                }
            }
            let (err, base) = (err / n as f32, (0.95 - 0.45f32).abs());
            eprintln!("{w}x{h} hole r={r}: mean error in the hole {err:.3} (was {base:.3}) in {:?}", t0.elapsed());
            assert!(err < base * 0.35, "the hole was not filled convincingly");
        }
    }

    /// Needs OPTIXEL_TEST_AI=1: downloads Stable Diffusion inpainting into the app's data folder.
    #[test]
    fn sd_download() {
        if std::env::var_os("OPTIXEL_TEST_AI").is_none() {
            return;
        }
        let data = crate::data_dir();
        let t0 = std::time::Instant::now();
        download(&data, "sd15-inpaint", &|d, t| eprint!("downloading {} / {} MB   ", d >> 20, t >> 20)).unwrap();
        eprintln!("
downloaded and verified in {:?}", t0.elapsed());
    }

    /// Needs OPTIXEL_TEST_AI=1 and the model downloaded: Stable Diffusion removes a blob from a smooth picture.
    #[test]
    fn sd_end_to_end() {
        if std::env::var_os("OPTIXEL_TEST_AI").is_none() {
            return;
        }
        let data = crate::data_dir();
        let (w, h, r) = (900usize, 700usize, 100usize);
        let bg = |x: usize, y: usize| 0.25 + 0.4 * (x as f32 / w as f32) + 0.1 * (y as f32 / h as f32);
        let (cx, cy) = (w / 2, h / 2);
        let inside = |x: usize, y: usize| ((x as f32 - cx as f32).powi(2) + (y as f32 - cy as f32).powi(2)).sqrt() < r as f32;
        let px: Vec<Px> = (0..w * h).map(|i| { let (x, y) = (i % w, i / w); let v = if inside(x, y) { 0.95 } else { bg(x, y) }; srgb_to_working([v, v * 0.5, 0.2]) }).collect();
        let a: Vec<f32> = (0..(2 * r) * (2 * r)).map(|k| if inside(cx - r + k % (2 * r), cy - r + k / (2 * r)) { 1.0 } else { 0.0 }).collect();
        let hole = Hole { x0: cx - r, y0: cy - r, w: 2 * r, h: 2 * r, a };
        let t0 = std::time::Instant::now();
        let out = inpaint(&data, "sd15-inpaint", Img { w, h, px }, &hole, 6, "", &|d, t| { eprint!("step {}/{t} ", d + 1); true }).unwrap();
        let (mut err, mut n) = (0.0, 0);
        for y in 0..h {
            for x in 0..w {
                if inside(x, y) {
                    let g = encode(out.px[y * w + x]);
                    err += (g[0] - bg(x, y)).abs();
                    n += 1;
                }
            }
        }
        if let Some(p) = std::env::var_os("OPTIXEL_TEST_OUT") {
            let mut buf = image::RgbImage::new(w as u32, h as u32);
            for (i, p) in out.px.iter().enumerate() {
                let g = encode(*p);
                buf.put_pixel((i % w) as u32, (i / w) as u32, image::Rgb([(g[0] * 255.0) as u8, (g[1] * 255.0) as u8, (g[2] * 255.0) as u8]));
            }
            buf.save(p).unwrap();
        }
        let err = err / n as f32;
        eprintln!("
sd15-inpaint: mean error in the hole {err:.3} (was {:.3}) in {:?}", 0.95 - 0.45, t0.elapsed());
        assert!(err < 0.2, "the hole was not filled convincingly");
    }

    /// Real photo: OPTIXEL_TEST_RAW=<file> OPTIXEL_TEST_OUT=<png> OPTIXEL_TEST_HOLE=cx,cy,r (fractions of width).
    #[test]
    fn sd_real_photo() {
        let file = std::env::var_os("OPTIXEL_TEST_RAW");
        let out = std::env::var_os("OPTIXEL_TEST_OUT");
        let hole = std::env::var("OPTIXEL_TEST_HOLE").ok();
        let (Some(file), Some(out), Some(hole)) = (file, out, hole) else { return };
        let engine = std::env::var("OPTIXEL_TEST_MODEL").unwrap_or_else(|_| "sd15-inpaint".into());
        let img = crate::develop::load(Path::new(&file), Some(1600)).unwrap();
        let (w, h) = (img.w, img.h);
        let v: Vec<f32> = hole.split(',').map(|s| s.parse().unwrap()).collect();
        let (cx, cy, r) = ((v[0] * w as f32) as usize, (v[1] * h as f32) as usize, (v[2] * w as f32) as usize);
        let inside = |x: usize, y: usize| ((x as f32 - cx as f32).powi(2) + (y as f32 - cy as f32).powi(2)).sqrt() < r as f32;
        let (x0, y0) = (cx.saturating_sub(r), cy.saturating_sub(r));
        let (ww, wh) = ((2 * r).min(w - x0), (2 * r).min(h - y0));
        let a: Vec<f32> = (0..ww * wh).map(|k| if inside(x0 + k % ww, y0 + k / ww) { 1.0 } else { 0.0 }).collect();
        let before = img.px.clone();
        let t0 = std::time::Instant::now();
        let res = inpaint(&crate::data_dir(), &engine, img, &Hole { x0, y0, w: ww, h: wh, a }, 6, "", &|_, _| true).unwrap();
        eprintln!("{engine}: {:?}", t0.elapsed());
        let mut buf = image::RgbImage::new(2 * w as u32, h as u32);
        for i in 0..w * h {
            for (side, px) in [(0usize, &before), (1, &res.px)] {
                let g = encode(px[i]);
                buf.put_pixel((side * w + i % w) as u32, (i / w) as u32, image::Rgb([(g[0] * 255.0) as u8, (g[1] * 255.0) as u8, (g[2] * 255.0) as u8]));
            }
        }
        buf.save(out).unwrap();
    }

    /// Needs OPTIXEL_TEST_AI=1 and OPTIXEL_TEST_MODEL=<id>: downloads that model.
    #[test]
    fn download_model() {
        if std::env::var_os("OPTIXEL_TEST_AI").is_none() {
            return;
        }
        let Ok(id) = std::env::var("OPTIXEL_TEST_MODEL") else { return };
        let t0 = std::time::Instant::now();
        download(&crate::data_dir(), &id, &|d, t| eprint!("downloading {} / {} MB   ", d >> 20, t >> 20)).unwrap();
        eprintln!("\n{id} downloaded, verified and unpacked in {:?}", t0.elapsed());
    }

    #[test]
    fn moebius_spec() {
        let m = spec("moebius").expect("moebius should be registered");
        assert_eq!(m.kind, "inpaint");
        assert_eq!(m.repo, "simonw/Moebius-ONNX");
        assert_eq!(m.name, "Moebius (RemovalModel) · Object removal");
        assert_eq!(m.files.len(), 3);
        assert!(m.files.iter().any(|f| f.0 == "unet.onnx"));
        assert!(m.files.iter().any(|f| f.0 == "vae_encoder.onnx"));
        assert!(m.files.iter().any(|f| f.0 == "vae_decoder.onnx"));

        // Alias check
        assert_eq!(spec("moebius-removal").unwrap().id, "moebius");

        // Verify models() lists Moebius for the UI
        let hw = Hardware { gpu: Some("Apple M4 Max".into()), vram_mb: Some(128 * 1024), device: "CoreML".into() };
        let all = models(Path::new("/tmp"), &hw);
        assert!(all.iter().any(|m| m.id == "moebius"));
    }
}


