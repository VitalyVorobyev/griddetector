Short version: you already have most of the right pieces (pyramid, gradients, LSD-ish segments, VP, refinement scaffolding). What’s missing is (1) a clean layering / crate layout, (2) a deliberately small public API with a “pipeline builder” feel, and (3) tests + examples that exercise stages independently.

Below is a concrete proposal you can start implementing.

⸻

1. Where the project is now (from the outside)

From README + docs and exports in lib.rs, the current picture looks like this:
	•	Single crate griddetector, library name grid_detector, crate-type rlib + cdylib, plus binaries grid_demo and lsd_vp_demo.  ￼
	•	Modules documented in doc/: image, pyramid, edges, segments, lsd_vp, refine, detector, types, diagnostics, homography, config, angle.  ￼
	•	lib.rs re-exports a lot of stuff at the crate root: the detector, params, workspace, homography helpers, result types, and a big block of diagnostic stage types.  ￼
	•	Algorithmically (in docs):
	•	Pyramid: separable 5-tap Gaussian + 2× decimation.  ￼
	•	Edges: Sobel/Scharr gradients, simple NMS.  ￼
	•	Segments: region-grow on gradient orientation, PCA, line in normal form.  ￼
	•	LSD→VP: orientation histogram → 2 families → LS VP estimate → coarse homography H0 = [vpu | vpv | x0].  ￼
	•	Refinement: segment upsampling + homography IRLS over bundles across pyramid.  ￼
	•	Roadmap doc already lists high-level milestones: metric upgrade, grid indexing, robustness, performance, tests.  ￼

You also have demo binaries and python plotting tools wired around JSON diagnostics in tools/ and config/.  ￼

Given what you wrote (“image pyramid and line segment detection works; VP broken; many things untested”), I’ll treat the docs as aspirational and focus on architecture & testability, not on assuming every detail is already implemented.

⸻

2. Target crate & module structure

2.1 Short-term (within current crate)

Before you split into a workspace, I’d clean up inside the existing griddetector crate so it already looks like multiple crates:

src/
  lib.rs                 // slim, curated re-exports
  core/
    angle.rs
    homography.rs
    types.rs             // GridResult, Pose, etc.
    config.rs            // GridParams, stage configs
  image/
    mod.rs               // pub use {pyramid, edges, segments}
    image.rs             // ImageU8, ImageF32
    pyramid.rs
    edges.rs
    segments.rs
  vp/
    mod.rs
    lsd_vp.rs
    outlier.rs           // outlier gating, families
  refine/
    mod.rs
    segment.rs
    homography.rs
    anchor.rs
    families.rs
    irls.rs
    types.rs
  detector/
    mod.rs               // GridDetector, DetectorWorkspace
    pipeline.rs          // orchestration
    bundling.rs
    indexing.rs          // future: grid indexing/u,v lattice
  diagnostics/
    mod.rs
    builders.rs          // run_lsd_stage, run_outlier_stage, etc.
    trace_types.rs
  bin/
    grid_demo.rs
    coarse_edges.rs
    coarse_segments.rs
    lsd_vp_demo.rs
    vp_outlier_demo.rs

Key ideas:
	•	core: everything mathy and “global”: angles, homographies, result types, config structs, error types.
	•	image: low-level image/pyramid/edges/segments; pure functions and small structs, no grid-specific types.
	•	vp: all vanishing-point & coarse H0 logic.
	•	refine: segment refinement & homography IRLS; already reflected in your refine doc.  ￼
	•	detector: the high-level orchestrator + bundling/indexing logic.
	•	diagnostics: trace graph, timings, builders that run a single stage and yield JSON-friendly outputs.

This can be done purely by moving files and adjusting module paths; it doesn’t break semantics and sets you up nicely for a workspace split.

2.2 Medium-term: workspace split

Once the internal structure stabilizes, I’d move to a 3–4 crate workspace:

# Cargo.toml (workspace root)
[workspace]
members = [
  "crates/grid-core",
  "crates/grid-algo",
  "crates/grid-cli",
  # maybe later: "crates/grid-ffi"
]

Crates:
	1.	grid-core
	•	angle, homography, types (GridResult, Pose, error enums), config (per-stage params).
	•	No image or rayon dependency; only nalgebra and serde.
	•	This is what Python / C++ world can depend on for pure data structures and maybe offline analysis.
	2.	grid-algo
	•	All algorithmic code: image, pyramid, edges, segments, vp, refine, detector, diagnostics.
	•	Depends on grid-core.
	•	Exposes the main Rust API surface for detection and diagnostics.
	3.	grid-cli
	•	All binaries (moved from root crate): grid_demo, lsd_vp_demo, coarse_edges, etc.
	•	Depends on grid-algo.
	•	Reads JSON configs, writes JSON diagnostics/images.
	4.	(Optional later) grid-ffi
	•	Thin cdylib crate that exposes a stable C ABI around a tiny subset:
	•	create/destroy detector
	•	process one image
	•	return a simple C struct / JSON blob.
	•	This justifies having crate-type = ["cdylib"] somewhere; if you don’t need FFI soon, I’d remove cdylib from the core crate to avoid confusion.

⸻

3. Public API surface: tiers and pipeline configurability

Right now lib.rs is essentially a “mega prelude” that dumps most of the crate into public scope.  ￼ That’s convenient for hacking but makes it harder to evolve internals.

I’d move to 3 tiers of API:

3.1 Tier 1: “Just detect” API

This is for typical users / quick experiments:

use grid_detector::{GridDetector, GridParams, ImageU8, GridResult};
use nalgebra::Matrix3;

let (w, h) = (640usize, 480usize);
let gray = ImageU8 { w, h, stride: w, data: &buffer };

let params = GridParams {
    kmtx: Matrix3::identity(),       // or real intrinsics
    // anything else stays at Default::default()
    ..Default::default()
};

let mut det = GridDetector::new(params);
let res: GridResult = det.process(gray);

Variants:

impl GridDetector {
    pub fn process(&mut self, img: ImageU8<'_>) -> GridResult;

    pub fn process_with_trace(
        &mut self,
        img: ImageU8<'_>,
    ) -> (GridResult, PipelineTrace);
}

Where:
	•	GridParams is your “one struct to configure everything”, with nested stage configs:

pub struct GridParams {
    pub camera: CameraParams,
    pub pyramid: PyramidParams,
    pub lsd: LsdParams,
    pub vp: VpParams,
    pub refine: RefineParams,
    pub bundling: BundlingParams,
    pub indexing: IndexingParams,
    pub diagnostics: DiagnosticParams,
}

Tier-1 users mostly tweak lsd and pyramid and ignore the rest.

3.2 Tier 2: Pipeline customization API

Goal: allow you (and advanced users) to run or swap individual stages, but still within a guided structure.

You already have run_lsd_stage, run_outlier_stage, DetectorWorkspace, and a bunch of diagnostic types re-exported.  ￼ I’d formalize this a bit:

pub struct PipelineConfig {
    pub pyramid: PyramidParams,
    pub lsd: LsdParams,
    pub vp: VpParams,
    pub outlier: OutlierParams,
    pub bundling: BundlingParams,
    pub refine: RefineParams,
}

pub struct DetectorWorkspace {
    // preallocated buffers, per-level caches, etc.
}

impl DetectorWorkspace {
    pub fn new(max_w: usize, max_h: usize, max_levels: usize) -> Self;
}

Stage runners:

pub fn run_pyramid(
    img: ImageU8<'_>,
    cfg: &PyramidParams,
    ws: &mut DetectorWorkspace,
) -> PyramidStageOutput;

pub fn run_lsd_stage(
    pyr: &PyramidStageOutput,
    cfg: &LsdParams,
    ws: &mut DetectorWorkspace,
) -> LsdStageOutput;

pub fn run_vp_stage(
    lsd: &LsdStageOutput,
    cfg: &VpParams,
) -> VpStageOutput;

// etc – outlier filter, bundling, refine homography

All these *StageOutput types already exist in some form in diagnostics; they can be the backbone of the “pipeline customization” API.

This is what your Rust tests and Python visualizers should use: run the stage via Rust, serialize StageOutput as JSON, then inspect/plot in Python.

3.3 Tier 3: Low-level building blocks

Keep modules like image, pyramid, edges, segments, lsd_vp, refine, homography public or public-behind-a-module-namespace:

pub mod image;     // ImageU8, ImageF32
pub mod pyramid;   // Pyramid, PyramidOptions
pub mod segments;  // Segment, LsdOptions, etc.
pub mod vp;        // VpEstimate, etc.
pub mod refine;    // RefineParams, segment_refine(), homography_refine()

But don’t re-export every type at crate root; instead provide a small prelude:

pub mod prelude {
    pub use crate::image::ImageU8;
    pub use crate::{GridDetector, GridParams, GridResult, Pose};
}

That lets you:
	•	keep internals moving without constantly breaking external users, and
	•	have tests and examples rely on stable, documented types.

⸻

4. Roadmap (refined and testability-biased)

Building on your doc/roadmap.md, here’s a more implementation-oriented version.

Phase 0 – Structure & API cleanup
	1.	Refactor module layout like in §2.1, without changing behaviour.
	2.	Slim lib.rs:
	•	Export only high-level API + prelude + diagnostics module gateway.
	•	Stop glob-re-exporting all diagnostic types at root.
	3.	Add process_with_trace if not already wired fully:
	•	One entrypoint that returns PipelineTrace for Python scripts.
	4.	Stabilize GridParams and stage sub-configs:
	•	Separate config structs per stage (already implied in docs).
	•	Derive Serialize/Deserialize so they can be read/written as JSON.

Phase 1 – Synthetic tests and golden diagnostics
	1.	Synthetic grid generator (Rust dev-dependency or small helper module):
	•	Generates perfect & slightly noisy grid images + exact homographies.
	•	Use simple primitives (lines / boxes) so it doesn’t drag in imageproc.
	2.	Unit tests for working blocks:
	•	Pyramid: verify decimation, blur behaviour, value range.  ￼
	•	Edges: check Sobel/Scharr implementation and NMS patterns on toy images.  ￼
	•	Segments: feed a small synthetic line image, assert a single long segment with expected endpoints.  ￼
	3.	Integration tests for LSD→VP:
	•	On synthetic grids with known vanishing points (one family at infinity, one finite; both finite; both at infinity).
	•	Assert:
	•	correct family assignment counts,
	•	VP reasonably close to ground truth,
	•	H0 warped rectified points land where expected.  ￼
	4.	Golden JSON traces:
	•	For a couple of real sample images (put in tests/data/).
	•	Run pipeline once, dump PipelineTrace to tests/golden/…json.
	•	Tests compare new run vs golden (with small numeric tolerances) so you catch regressions.

Phase 2 – VP robustness & homography sanity

Here we tackle the thing you already know is broken.
	1.	Refactor VP estimation to explicit “cost”:
	•	Move from pure algebraic LS on line coefficients to an objective you can test:
	•	e.g. minimize angular residuals of line directions vs VP rays, using IRLS.
	•	Implement a separate vp::estimate_families_robust() with:
	•	Huber or Tukey loss on angular error,
	•	guard for near-singular configurations,
	•	clear behaviour when VP is inside vs outside image.
	2.	Unit tests for degenerate cases:
	•	All lines parallel (VP at infinity).
	•	Two nearly-coincident families.
	•	One family missing or very weak.
	3.	Homography refinement sanity checks:
	•	Use synthetic data to ensure that refining from a slightly perturbed H0 converges back to ground truth when noise is small.
	•	Add asserts that bail out when IRLS diverges or confidence is inconsistent (instead of blindly returning junk).  ￼

Phase 3 – Grid indexing & pose completion
	1.	Grid indexing:
	•	Implement indexing step: given refined homography, map bundles/segments into rectified space and snap to integer (u,v) indices.
	•	Expose per-line / per-cell occupancy and confidence in GridResult.
	2.	Metric upgrade:
	•	Enforce equal spacing in u and v in the rectified frame:
	•	estimate spacing and scale from bundle intersections,
	•	optionally refine homography under that constraint (affine/metric upgrade).
	3.	Pose output:
	•	Given intrinsics, turn homography into plane pose (R, t) with proper conventions; surface this as Pose (already exists).  ￼

Phase 4 – Examples & diagnostics for real work
	1.	Curate demo binaries:
	•	coarse_edges
	•	coarse_segments
	•	lsd_vp_demo
	•	vp_outlier_demo
	•	grid_demo (full pipeline)
	2.	For each binary:
	•	Take JSON config.
	•	Write:
	•	PNG(s) with overlays,
	•	JSON diagnostics with stage outputs.
	3.	Document the workflow (in the book / README):
	•	“How to debug LSD thresholds”
	•	“How to inspect VP families”
	•	“How to read PipelineTrace from Python”

This ties directly to “I am going to investigate the results using python visualization scripts”.

Phase 5 – Performance & parallelism

Once correctness + tests are in place:
	1.	Use DetectorWorkspace to pre-allocate everything and avoid per-frame allocations.
	2.	Turn on rayon for:
	•	pyramid construction,
	•	gradient computation,
	•	potentially segment extraction and refinement across tiles/lines.
	3.	Add Criterion benchmarks for core blocks and full pipeline; track them in CI.

⸻

5. Likely design issues and risks (given what we can see)

Because I can’t see the full Rust sources via this interface, I can’t point to exact lines. But from the docs and your description, I’d flag:
	1.	Over-wide lib.rs surface
	•	Root re-exports a lot of diagnostic types and internal helpers.  ￼
	•	This will make it painful to evolve internals. Tighten this to a small API + explicit modules.
	2.	VP / homography objectives may be too algebraic
	•	Docs emphasize solving ax+by+c≈0 in LS, which is numerically fragile when vanishing points approach infinity or live near the image centre.  ￼
	•	Given your observations (bad H, half-pixel artefacts in rectification, etc. from our other chats), this is a prime suspect.
	•	Moving to angular residuals + IRLS with guards should be a design goal in Phase 2.
	3.	Diagnostics entangled with core
	•	diagnostics types are re-exported at root; likely referenced from the detector itself.  ￼
	•	It’s great for tooling, but you might want to make diagnostics optional (e.g. behind a feature or just not computed unless asked via process_with_trace), to keep hot path lean.
	4.	cdylib without a defined FFI story
	•	Right now you declare crate-type = ["rlib", "cdylib"] but I don’t see README or docs describing a C API.  ￼
	•	Either:
	•	add a small, stable FFI surface in a dedicated module or crate, or
	•	drop cdylib until you actually need it (simpler builds, less confusion).
	5.	Roadmap/docs slightly ahead of code
	•	doc/refine.md describes a fairly advanced refinement pipeline, but you said a lot of that is still draft / untested.  ￼
	•	That’s fine, but I’d mark aspirational parts in docs explicitly (“planned”, “TODO”) so you don’t forget what’s actually wired vs. planned.

⸻

6. Suggested very next steps

If you’re okay with this direction, I’d suggest we do this next, concretely:
	1.	Design the cleaned-up public API in lib.rs and GridParams (signatures and struct fields only, no big rewrites yet).
	2.	Restructure modules into the “pseudo-workspace” layout under src/ without changing functionality.
	3.	Add the process_with_trace entrypoint and ensure diagnostics types form a coherent tree for JSON.

If you want, in the next step I can draft:
	•	a concrete lib.rs with the proposed prelude and public modules; and
	•	skeletons for GridParams, DetectorWorkspace, and PipelineConfig matching what you already have, so you can adapt the actual code with minimal churn.
