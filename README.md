# awesome-mlir-compiler-runtime-prep

> A curated, audited, interview-focused list for **MLIR / LLVM / compiler / runtime / Triton / GPU codegen** preparation.

This list is **not** just “awesome MLIR”. That name would be too narrow for the links here.

- IR design and compiler architecture
- MLIR dialects, rewrites, lowering, bufferization, codegen
- runtime concerns: dispatch, memory, dynamic shapes, scheduling
- GPU and Triton layout reasoning
- LLVM / SSA / backend / toolchain fundamentals
- low-precision deployment tradeoffs

So the repo name stays broad on purpose:

**`awesome-mlir-compiler-runtime-prep`**

## How to use this list

### Priority legend

- **P0 — Core**: highest ROI for most MLIR/compiler/runtime interviews
- **P1 — Role-dependent**: very strong if the role mentions runtime, GPU, Triton, LLVM backend, or performance
- **P2 — Stretch / adjacent**: useful for standing out, but not first-pass material

### Coverage tags

- **MLIR Core** — dialects, passes, conversion, IR structure
- **Runtime** — dispatch, host/device split, memory, execution model
- **GPU/Triton** — layout, kernels, SIMT/SIMD reasoning
- **LLVM/Backend** — SSA, LLVM IR, assembly, triples, linker/toolchain
- **Quantization** — FP8/FP4/PTQ deployment context
- **Adjacent** — helpful background, but not central interview prep

### Suggested order

1. Do **P0 / MLIR Core** first.
2. Add **P0 / Runtime** next.
3. Choose one specialization:
   - **GPU/Triton** for accelerator and kernel roles
   - **LLVM/Backend** for lower-level compiler roles
   - **Quantization** if deployment and datatype support matters
4. Use **P2** only after the above is solid.

---

## Fast study paths

### 5–7 day crash course

1. Chip Huyen — big picture
2. Stephen Diehl — MLIR introduction
3. Jeremy Kun MLIR tutorial repo
4. Lei — CodeGen Dialects, Linalg, Vector
5. Stephen Diehl — Memory in MLIR
6. Lei — Single-node ML Runtime Foundation
7. IREE — Data Tiling Walkthrough, Microkernels
8. Miguel Young — Why SSA?, LLVM IR, Target Triples

### 2–3 week interview track

Finish all P0 items, then choose one branch:

- **Runtime-heavy**: IREE posts, ApX course, Atomicless Concurrency
- **GPU-heavy**: Triton layout series, Gluon, tiny-gpu-compiler, SIMD article
- **Backend-heavy**: Assembly, linker scripts, pointers, target triples

### What to skip on a first pass

These are good resources, but **not** first-pass interview essentials:

- Neural Networks / Transformers explainers
- Typechecker Zoo
- Formatters
- Best C++ Library
- Curta
- Matrix post
- tuple / SAM closures unless the role is very language-implementation-heavy

---

## 1) Core mental model: what ML compilers and runtimes are doing

These are the best starting points for forming the right mental map.

- **[P0] [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)**  
  **Coverage:** MLIR Core, Runtime  
  **Why it matters:** Best big-picture intro in the set. Use this before learning dialect names.

- **[P0] [Introduction to MLIR](https://www.stephendiehl.com/posts/mlir_introduction/)**  
  **Coverage:** MLIR Core  
  **Why it matters:** Clear overview of LLVM, MLIR, modern compiler pipelines, and how MLIR fits as a multi-level IR.

- **[P0] [MLIR CodeGen Dialects for Machine Learning Compilers](https://www.lei.chat/posts/mlir-codegen-dialects-for-machine-learning-compilers/)**  
  **Coverage:** MLIR Core  
  **Why it matters:** One of the strongest “map of the territory” articles. Explains the dialect hierarchy from model-level representations down to kernel codegen.

- **[P1] [A Meticulous Guide to Advances in Deep Learning Efficiency over the Years](https://alexzhang13.github.io/blog/2024/efficient-dl/)**  
  **Coverage:** Adjacent, GPU/Triton, Quantization  
  **Why it matters:** Excellent macroscopic context for how kernels, compilers, libraries, hardware, and deployment evolved together. Great context, but broader than interview-core.

- **[P1] [How To Scale Your Model](https://jax-ml.github.io/scaling-book/)**  
  **Coverage:** Runtime, GPU/Triton, Adjacent  
  **Why it matters:** Strong systems/performance context: rooflines, TPUs, sharded matmuls, inference, serving, profiling, and JAX.

---

## 2) Hands-on MLIR onboarding

These are the best practical starting points if you want to build intuition by doing.

- **[P0] [j2kun/mlir-tutorial](https://github.com/j2kun/mlir-tutorial#mlir-tutorial)**  
  **Coverage:** MLIR Core  
  **Why it matters:** The best step-by-step beginner repo in this list. Covers setup, lowerings, passes, TableGen, dialect definition, traits, verifiers, canonicalization, rewrites, dialect conversion, LLVM lowering, and PDLL.

- **[P0] [MLIR Beginner-Friendly Tutorial: Part 1](https://www.youtube.com/watch?v=Uno_XhtkT5E)**  
  **Coverage:** MLIR Core  
  **Why it matters:** Beginner-friendly video counterpart to the written material; good first watch before diving into custom dialects.

- **[P1] [MLIR YouTube playlist](https://www.youtube.com/watch?v=tW8FVmQBYPk&list=PLlKj-4rp1Gz1Vt3onHU_SLjDwn51UsXrG)**  
  **Coverage:** MLIR Core  
  **Why it matters:** More advanced than the beginner video. Good reinforcement after you know the basics; includes setup, codegen, JIT, and API-oriented walkthroughs.

- **[P1] [tiny-gpu-compiler](https://gautam1858.github.io/tiny-gpu-compiler/)**  
  **Coverage:** MLIR Core, GPU/Triton  
  **Why it matters:** Excellent educational project showing a full MLIR-based path from a C-like GPU kernel language to binary instructions and a cycle-accurate simulator. Great for people who learn best from end-to-end systems.

---

## 3) Core MLIR mechanics: dialects, transformations, and lowering

This is the heart of the list.

### Must-read core

- **[P0] [MLIR Linalg Dialect and Patterns](https://www.lei.chat/posts/mlir-linalg-dialect-and-patterns/)**  
  **Coverage:** MLIR Core  
  **Why it matters:** Essential for structured ops, tiling, fusion, distribution, vectorization, and lowering to loops.

- **[P0] [MLIR Vector Dialect and Patterns](https://www.lei.chat/posts/mlir-vector-dialect-and-patterns/)**  
  **Coverage:** MLIR Core, GPU/Triton  
  **Why it matters:** Essential for understanding the vector level in progressive lowering: static tiles, vector ops, register-level reasoning, and lowering choices such as `vector.contract` strategies.

- **[P0] [Memory in MLIR](https://www.stephendiehl.com/posts/mlir_memory/)**  
  **Coverage:** MLIR Core, Runtime  
  **Why it matters:** High interview ROI. Explains tensors vs memrefs vs low-level memory forms — one of the most common places candidates are shaky.

### Strong follow-ups

- **[P1] [Affine Dialect and OpenMP](https://www.stephendiehl.com/posts/mlir_affine/)**  
  **Coverage:** MLIR Core  
  **Why it matters:** Valuable for loop nests, affine maps, polyhedral-style reasoning, parallelization, and dependency analysis.

- **[P1] [Linear Algebra in MLIR](https://www.stephendiehl.com/posts/mlir_linear_algebra/)**  
  **Coverage:** MLIR Core  
  **Why it matters:** Good second source for Linalg, especially `linalg.generic`, iterator types, indexing maps, reduction structure, and named ops.

- **[P1] [Specializing Python with E-graphs](https://www.stephendiehl.com/posts/mlir_egraphs/)**  
  **Coverage:** MLIR Core, Adjacent  
  **Why it matters:** Strong for rewrite systems and equality saturation. Not mandatory for most interviews, but very useful if the role mentions transformations or superoptimization-like reasoning.

- **[P1] [GPU Compilation with MLIR](https://www.stephendiehl.com/posts/mlir_gpu/)**  
  **Coverage:** MLIR Core, GPU/Triton  
  **Why it matters:** End-to-end toy compiler path that reaches PTX. Best used after basic MLIR familiarity.

### Adjacent, not first-pass core

- **[P2] [Neural Networks](https://www.stephendiehl.com/posts/mlir_neural_networks/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Helpful for people who want model-side intuition, but this is not directly compiler prep.

- **[P2] [Transformers](https://www.stephendiehl.com/posts/mlir_transformers/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Good model context, not compiler-core.

- **[P2] [Typechecker Zoo](https://www.stephendiehl.com/posts/typechecker_zoo/)**  
  **Coverage:** Adjacent, LLVM/Backend  
  **Why it matters:** Great language-implementation material, but it should not be ranked alongside core MLIR runtime prep.

- **[P2] [Stephen Diehl compiler tag page](https://www.stephendiehl.com/tags/compilers/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Useful as an index page, not as a primary resource itself.

---

## 4) Runtime systems and IREE-style deployment

This is the part many candidates under-prepare.

- **[P0] [Single-node ML Runtime Foundation](https://www.lei.chat/posts/single-node-ml-runtime-foundation/)**  
  **Coverage:** Runtime  
  **Why it matters:** Best runtime bridge in the list. Covers the single-node runtime problem, host/device separation, resource management, dispatching work to accelerators, and uses IREE as a concrete example.

- **[P0] [IREE Data-Tiling Walkthrough](https://github.com/iree-org/iree/blob/main/docs/website/docs/community/blog/posts/data-tiling-walkthrough.md?plain=1)**  
  **Coverage:** Runtime, MLIR Core  
  **Why it matters:** A very concrete real-world compiler walkthrough. Explains IREE’s encoding dialect, data tiling, fusion path, and how encodings are attached to matmul and propagated through the pipeline.

- **[P0] [IREE Microkernels](https://github.com/iree-org/iree/blob/main/docs/website/docs/community/blog/posts/microkernels.md)**  
  **Coverage:** Runtime, MLIR Core, LLVM/Backend  
  **Why it matters:** One of the best articles here for connecting compiler lowering to runtime execution. Explains rewriting `linalg.matmul` to `linalg.mmt4d`, then to a ukernel op, then LLVM IR, bitcode linking, and target-specific inner loops.

- **[P1] [ML Compiler & Runtime Optimization Techniques (ApX)](https://apxml.com/courses/compiler-runtime-optimization-ml)**  
  **Coverage:** Runtime, MLIR Core, Quantization  
  **Why it matters:** Broad guided course. Good as a structured syllabus, but not as canonical as the P0 reads above.

### Interview questions this section helps with

- Where should compilation stop and the runtime take over?
- How do host and device responsibilities split?
- What happens to layouts, encodings, and kernel selection after lowering?
- How do dynamic shapes, resource management, and scheduling affect performance?

---

## 5) GPU, Triton, layouts, and explicit performance control

This branch is extremely valuable for GPU compiler and accelerator roles.

### Best GPU/Triton sequence

1. Triton Linear Layout: Concept  
2. Triton Linear Layout: Examples  
3. Triton Bespoke Layouts  
4. Gluon: Explicit Performance  
5. tiny-gpu-compiler  
6. GPU Compilation with MLIR  
7. Designing a SIMD Algorithm from Scratch

### Core GPU/Triton resources

- **[P1] [Triton Linear Layout: Concept](https://www.lei.chat/posts/triton-linear-layout-concept/)**  
  **Coverage:** GPU/Triton  
  **Why it matters:** Best conceptual introduction to Triton linear layout. Focuses on GPU hierarchy, access patterns, and why a unified layout representation helps codegen.

- **[P1] [Triton Linear Layout: Examples](https://www.lei.chat/posts/triton-linear-layout-examples/)**  
  **Coverage:** GPU/Triton  
  **Why it matters:** Turns the conceptual article into concrete layout constructions and operations. Best read immediately after the Concept post.

- **[P1] [Triton Bespoke Layouts](https://www.lei.chat/posts/triton-bespoke-layouts/)**  
  **Coverage:** GPU/Triton  
  **Why it matters:** Useful for understanding blocked/shared/MMA-style layouts and how warp/thread arrangements are expressed.

- **[P1] [Gluon: Explicit Performance](https://www.lei.chat/posts/gluon-explicit-performance/)**  
  **Coverage:** GPU/Triton, Runtime  
  **Why it matters:** Best for reasoning about portability vs performance and why domain-specific systems sometimes expose explicit controls instead of hiding them behind generic abstractions.

### Useful companions

- **[P1] [GPU Compilation with MLIR](https://www.stephendiehl.com/posts/mlir_gpu/)**  
  **Coverage:** GPU/Triton, MLIR Core  
  **Why it matters:** Toy pipeline to PTX; useful for “how would you build a small GPU compiler?” conversations.

- **[P1] [tiny-gpu-compiler](https://gautam1858.github.io/tiny-gpu-compiler/)**  
  **Coverage:** GPU/Triton, MLIR Core  
  **Why it matters:** Especially strong for intuition-building.

---

## 6) LLVM, SSA, backend, and toolchain fundamentals

This is where strong candidates separate themselves from “I only know passes” candidates.

### Highest ROI backend reads

- **[P0] [Why SSA?](https://mcyoung.xyz/2025/10/21/ssa-1/)**  
  **Coverage:** LLVM/Backend  
  **Why it matters:** Outstanding explanation of SSA as a compiler architecture, including phi-style reasoning and CFG structure.

- **[P0] [A Gentle Introduction to LLVM IR](https://mcyoung.xyz/2023/08/01/llvm-ir/)**  
  **Coverage:** LLVM/Backend  
  **Why it matters:** Best practical introduction to reading LLVM IR without drowning in metadata.

- **[P1] [What the Hell Is a Target Triple?](https://mcyoung.xyz/2025/04/14/target-triples/)**  
  **Coverage:** LLVM/Backend  
  **Why it matters:** Very useful for backend and deployment interviews. Helps decode architecture / vendor / OS / ABI naming and what those strings really mean.

### Strong follow-ups

- **[P1] [Understanding Assembly, Part I: RISC-V](https://mcyoung.xyz/2021/11/29/assembly-1/)**  
  **Coverage:** LLVM/Backend  
  **Why it matters:** Excellent for control flow, branches, loops, and what IR eventually becomes.

- **[P1] [Designing a SIMD Algorithm from Scratch](https://mcyoung.xyz/2023/11/27/simd-base64/)**  
  **Coverage:** LLVM/Backend, GPU/Triton  
  **Why it matters:** Superb performance-thinking resource. Helps with vectorization, lane-wise reasoning, and “thinking like hardware”.

- **[P1] [The Taxonomy of Pointers](https://mcyoung.xyz/2021/05/24/ptr-taxonomy/)**  
  **Coverage:** LLVM/Backend  
  **Why it matters:** Good for low-level semantics, ownership, pointer representations, and memory model intuition.

- **[P2] [Everything You Never Wanted To Know About Linker Script](https://mcyoung.xyz/2021/06/01/linker-script/)**  
  **Coverage:** LLVM/Backend  
  **Why it matters:** Great if the role touches embedded, binaries, loaders, or deployment. Too deep for a first pass.

- **[P2] [Atomicless Concurrency](https://mcyoung.xyz/2023/03/29/rseq-checkout/)**  
  **Coverage:** Runtime, LLVM/Backend  
  **Why it matters:** Strong runtime-systems depth; good differentiator for systems-heavy roles.

### Adjacent / specialized backend reads

- **[P2] [What is a Matrix? A Miserable Pile of Coefficients!](https://mcyoung.xyz/2023/09/29/what-is-a-matrix/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Nice math intuition, but not interview-core.

- **[P2] [Single Abstract Method Traits](https://mcyoung.xyz/2023/05/11/sam-closures/)**  
  **Coverage:** Adjacent, LLVM/Backend  
  **Why it matters:** More language-design and closure-lowering flavor than MLIR/runtime prep.

- **[P2] [std::tuple the Hard Way](https://mcyoung.xyz/2022/07/13/tuples-the-hard-way/)**  
  **Coverage:** Adjacent, LLVM/Backend  
  **Why it matters:** Useful for advanced C++ metaprogramming fluency, not core interview prep.

- **[P2] [The Art of Formatting Code](https://mcyoung.xyz/2025/03/11/formatters/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Interesting tooling/front-end article, not central to MLIR compiler/runtime prep.

- **[P2] [The Best C++ Library](https://mcyoung.xyz/2025/07/14/best/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Good engineering/craftsmanship essay, but should not be treated as interview-core.

- **[P2] [3Hz Computer, Hold the Transistors](https://mcyoung.xyz/2022/07/24/curta/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Fun and insightful, but far from core prep.

---

## 7) Quantization and low precision

Important for deployment-aware roles, but not part of the absolute minimum MLIR/compiler core.

- **[P1] [Optimizing LLMs for Performance and Accuracy with Post-Training Quantization](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/)**  
  **Coverage:** Quantization  
  **Why it matters:** Practical PTQ overview. Good for calibration-aware discussions and modern schemes such as SmoothQuant, AWQ, AutoQuantize, and deployment-oriented tradeoffs.

- **[P1] [Floating-Point 8: An Introduction to Efficient, Lower-Precision AI Training](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)**  
  **Coverage:** Quantization  
  **Why it matters:** Best overview here for FP8 formats and where they fit in modern training stacks.

- **[P2] [Improving Quantized FP4 Weight Quality via Logit Distillation](https://dropbox.github.io/fp4_blogpost/)**  
  **Coverage:** Quantization  
  **Why it matters:** More advanced and more specialized. Strong if you want to discuss FP4 recovery methods and post-hoc correction, but not required for most interviews.

### Best use of this section

Read this section after the compiler/runtime fundamentals, not before.

---

## 8) Role-based reading tracks

### A. MLIR compiler engineer

- Chip Huyen intro
- Stephen Diehl intro
- Jeremy Kun tutorial repo
- Lei: CodeGen Dialects
- Lei: Linalg
- Lei: Vector
- Stephen Diehl: Memory in MLIR
- IREE Data Tiling
- Why SSA?
- LLVM IR

### B. MLIR + runtime / deployment role

- Lei: Single-node Runtime Foundation
- IREE Data Tiling
- IREE Microkernels
- Stephen Diehl: Memory in MLIR
- ApX course
- Target Triples
- Atomicless Concurrency

### C. GPU / Triton / kernel codegen role

- Lei: Vector
- Triton Linear Layout: Concept
- Triton Linear Layout: Examples
- Triton Bespoke Layouts
- Gluon: Explicit Performance
- tiny-gpu-compiler
- GPU Compilation with MLIR
- SIMD article

### D. LLVM / backend-heavy role

- Why SSA?
- LLVM IR
- Target Triples
- Assembly
- Pointers
- Linker Script
- IREE Microkernels

### E. Low-precision deployment-aware role

- NVIDIA PTQ post
- NVIDIA FP8 post
- Dropbox FP4 post
- IREE Microkernels
- Triton / GPU layout resources

---

## 9) Minimal shortlist

If you only have time for a very small subset, read these first:

- [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)
- [Introduction to MLIR](https://www.stephendiehl.com/posts/mlir_introduction/)
- [j2kun/mlir-tutorial](https://github.com/j2kun/mlir-tutorial#mlir-tutorial)
- [MLIR Beginner-Friendly Tutorial: Part 1](https://www.youtube.com/watch?v=Uno_XhtkT5E)
- [MLIR CodeGen Dialects for Machine Learning Compilers](https://www.lei.chat/posts/mlir-codegen-dialects-for-machine-learning-compilers/)
- [MLIR Linalg Dialect and Patterns](https://www.lei.chat/posts/mlir-linalg-dialect-and-patterns/)
- [MLIR Vector Dialect and Patterns](https://www.lei.chat/posts/mlir-vector-dialect-and-patterns/)
- [Memory in MLIR](https://www.stephendiehl.com/posts/mlir_memory/)
- [Single-node ML Runtime Foundation](https://www.lei.chat/posts/single-node-ml-runtime-foundation/)
- [IREE Data-Tiling Walkthrough](https://github.com/iree-org/iree/blob/main/docs/website/docs/community/blog/posts/data-tiling-walkthrough.md?plain=1)
- [IREE Microkernels](https://github.com/iree-org/iree/blob/main/docs/website/docs/community/blog/posts/microkernels.md)
- [Why SSA?](https://mcyoung.xyz/2025/10/21/ssa-1/)
- [A Gentle Introduction to LLVM IR](https://mcyoung.xyz/2023/08/01/llvm-ir/)
- [What the Hell Is a Target Triple?](https://mcyoung.xyz/2025/04/14/target-triples/)
