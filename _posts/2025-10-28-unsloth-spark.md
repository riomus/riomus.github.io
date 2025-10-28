---
layout: post
title: Unsloth your DGX Spark
excerpt_separator:  <!--more-->
tags: [DGX Spark, Unsloth, NVIDIA, PyTorch, LLM Tuning]
---
## Unsloth your DGX Spark


The NVIDIA DGX Spark is a powerful little devbox designed for local model development, particularly for those eager to experiment with the B10 chip. While it might not boast raw speed compared to its larger siblings, its 128GB of unified memory makes it surprisingly capable for large model fine-tuning. However, getting everything to play nicely, especially with tools like Unsloth, requires a few specific configurations. This post will guide you through optimizing your DGX Spark setup for efficient LLM fine-tuning using Unsloth.

### TL/DR

To efficiently use Unsloth on your DGX Spark, leverage NVIDIA's official PyTorch Docker image. Create a `uv` virtual environment with system site packages, block default `torch`, `triton`, and `torchvision` installations, and build `xformers` from a compatible source. Set `TORCH_CUDA_ARCH_LIST` to `12.0` to avoid build issues. Finally, integrate this `uv` environment into Jupyter Lab for seamless development.  [Here](https://github.com/riomus/dgx-spark-unsloth) you can find a complete repo.

### The DGX Spark Landscape

The DGX Spark is built around the B10 chip, featuring an impressive 128GB of unified memory. This makes it an excellent platform for experimenting with larger language models (LLMs) locally, where memory can often be the bottleneck.

NVIDIA strongly recommends using their optimized Docker images for PyTorch on the DGX Spark. Specifically, the [NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=25.09-py3) (`nvidia/pytorch:25.09-py3`) is tailored for NVIDIA hardware, including Blackwell (B10). This image comes pre-packaged with a special PyTorch build, Triton, and TorchVision, all optimized for NVIDIA's ecosystem.

A crucial point for the DGX Spark is its compute capability: `12.1`. While this is cutting-edge, some libraries like `xformers` and `fast-attention` are still catching up, with support often in active development (e.g., [xformers pull request #1344](https://github.com/facebookresearch/xformers/pull/1344)).

### Setting Up Your Environment for Unsloth

To get the best performance from Unsloth on your DGX Spark, we'll combine the benefits of NVIDIA's optimized PyTorch with `uv` for dependency management and custom `xformers` installation.

#### 1. Start with the NVIDIA Docker Image

Ensure you are running inside the official NVIDIA PyTorch Docker image. This provides the specially built PyTorch version that leverages the B10's capabilities.

#### 2. Create a `uv` Virtual Environment

Since the Docker image already provides an optimized PyTorch installation, we want our virtual environment to use it without reinstalling. This is achieved by creating the `uv` environment with `--system-site-packages`:

```bash
uv venv --system-site-packages
source .venv/bin/activate
```

#### 3. Pinning Dependencies and Blocking PyTorch

To prevent `uv` from trying to reinstall `torch`, `triton`, and `torchvision` (which are already present and optimized in the base image), we need to explicitly block them in our `pyproject.toml`. This ensures we rely on the system-level NVIDIA-optimized versions.

Create or update your `pyproject.toml` file with the following:

```toml
[project]
name = "dgx_spark_unsloth"
version = "0.1.0"
dependencies = [
    # Other dependencies 
]

[tool.uv]
override-dependencies = [
  "torch; python_version < '0'",
  "triton; python_version < '0'",
  "torchvision; python_version < '0'"
]
```

The `python_version < '0'` trick effectively makes these dependencies impossible to install by `uv`, forcing it to use the system-wide packages.

#### 4. Handling `TORCH_CUDA_ARCH_LIST`

When building packages like `xformers`, the `TORCH_CUDA_ARCH_LIST` environment variable is used to determine the target CUDA architectures. The NVIDIA base image often exposes a comprehensive list like `"8.0 8.6 9.0 10.0 11.0 12.0+PTX"`. However, this can sometimes cause parsing issues or lead to unnecessary builds for architectures not relevant to your DGX Spark.

The DGX Spark's compute capability is `12.1`. For `xformers` and similar libraries, targeting `12.0` is sufficient and avoids potential problems. You can set this explicitly:

```bash
export TORCH_CUDA_ARCH_LIST=12.0
```
Alternatively, if you want to avoid any complex parsing by `xformers`, you can simply unset it, letting the build system infer the current architecture.

```bash
unset TORCH_CUDA_ARCH_LIST
```

#### 5. Installing `xformers` from Source

As mentioned, full support for `CU130` and compute capability `12.1` in `xformers` might still be pending in official releases. To get around this, we can install a compatible version directly from a development branch. For example, using a branch with recent updates:

Update your `pyproject.toml` to include `xformers` from a specific Git repository and branch, and disable build isolation for it:

```toml
[project]
name = "dgx_spark_unsloth"
version = "0.1.0"
dependencies = [
    "unsloth",
    "xformers",
]

[tool.uv.sources]
xformers = {git="https://github.com/johnnynunez/xformers", branch="main"} # Using a known compatible branch

[tool.uv]
override-dependencies = [
  "torch; python_version < '0'",
  "triton; python_version < '0'",
  "torchvision; python_version < '0'"
]
no-build-isolation-package= ["xformers"]
```
Now, install your dependencies using `uv`:
```bash
uv sync
```
This will fetch and build `xformers` from the specified Git branch.

#### 6. Integrating with Jupyter Lab

To easily experiment with models and fine-tuning scripts, you'll want to use Jupyter Lab with your newly configured `uv` environment.

First, install `ipykernel` and create a custom kernel:

```bash
uv run python -m ipykernel install --user --name venv --display-name "Python (.venv)"
```

Then, launch Jupyter Lab:

```bash
uv run jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
```
You can now select the "Python (.venv)" kernel in your Jupyter notebooks to leverage your specialized environment.

### Takeaways

*   **NVIDIA Docker is Key:** Always start with the official NVIDIA PyTorch Docker image for optimized performance on DGX Spark.
*   **`uv` for Dependency Control:** `uv` is excellent for managing dependencies, especially when you need to combine system packages with custom builds.
*   **System Site Packages:** Use `--system-site-packages` and `override-dependencies` to leverage the base Docker image's PyTorch.
*   **`TORCH_CUDA_ARCH_LIST` Control:** Explicitly set or unset `TORCH_CUDA_ARCH_LIST` (e.g., `export TORCH_CUDA_ARCH_LIST=12.0`) to avoid build failures for libraries like `xformers`.
*   **Custom `xformers` Build:** Be prepared to install `xformers` from a compatible Git branch or source until official releases catch up with DGX Spark's `CU130` and `12.1` compute capabilities.

### References

*   **NVIDIA PyTorch Container:** [https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=25.09-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=25.09-py3)
*   **`xformers` GitHub:** [https://github.com/facebookresearch/xformers/](https://github.com/facebookresearch/xformers/)
*   **`uv` Documentation:** [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
*   **Complete Repo:** [https://github.com/riomus/dgx-spark-unsloth](https://github.com/riomus/dgx-spark-unsloth)

With these steps, your DGX Spark will be ready to unsloth your LLM fine-tuning experiments, leveraging its powerful unified memory and the optimized NVIDIA software stack. Happy experimenting!


### Prompt

Post was generated using AI - here is the prompt if you do not have time to read the story

```
Write Blog post using markdown for jekyll blog. 

title: Unsloth your DGX Spark

Add sections like
TL/DR
References
Take aways

Do not use to many emojis. Add links where possible, use code blocks for readibility. 

Data for the post
- DGX spark is a small devbox to play with local model development on B10 chip - it is not fast but has 128GB unified memory- so it is not bad for large model fine-tuning
- NVIDIA recommends usage of their docker image when using pytorch in dgx spark https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=25.09-py3  
- that image contains a special pytorch build optimised for nvidia + blackweell (dgx spark is b10 chip), triton, torchvision
- DGX spark requires CU130 and has compute capabilities 12.1 - but for some libs supporta is not yet merged (fast-attention, xformers - https://github.com/facebookresearch/xformers/pull/1344)
- To fully utilize performance of spark using unsloth - it is best to use official image + UV based env to pin dependncies and have all in correct version while  using official pytorch from nvidia (base image)
- To use torch from official image but with UV - we need to use system site packages and block torch installation - to do it we need to create venv using
uv venv --system-site-packages

and our pyproject.toml needs to block torch as a dependency using
[tool.uv]
override-dependencies = [
  "torch; python_version < '0'",
"triton; python_version <'0'",
"torchvision; python_version<'0'"
]


- for building inside of the base image it is important to override TORCH_CUDA_ARCH_LIST. Image exposes "8.0 8.6 9.0 10.0 11.0 12.0+PTX"  - and the xformers fails to parse it. DGX spark requires compute capabilities 12+ (it is 12.1 - but 12.0 works in xformers).
Because we are  building for local only we can simply
unset TORCH_CUDA_ARCH_LIST
or 
export TORCH_CUDA_ARCH_LIST=12.0

- because xformers support is not yet merged - we can install it from sources 
dependencies = [
    "xformers",
]
[tool.uv.sources]
xformers = {git="https://github.com/johnnynunez/xformers", branch="main"}
  
[tool.uv]
no-build-isolation-package= ["xformers"]

-After all those tricks - we can use unsloth using jupyter lab to experiment with models finetuning. 
- Install custom kernel thaw will use venv 
uv run python -m ipykernel install --user --name venv --display-name "Python (.venv)"
- Run jupyter lab
uv run jupyter lab


- Complete repo https://github.com/riomus/dgx-spark-unsloth

```