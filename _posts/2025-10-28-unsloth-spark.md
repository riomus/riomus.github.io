---
layout: post
title: Unsloth your DGX Spark
excerpt_separator:  <!--more-->
tags: [DGX Spark, Unsloth, NVIDIA, PyTorch, LLM Tuning]
---

## TL/DR

The NVIDIA DGX Spark is a powerful little devbox for local model development, boasting 128GB of unified memory despite its compact size. To truly unleash its potential with tools like Unsloth, you need to navigate a few key challenges:

*   **Official NVIDIA PyTorch Image is Key**: Leverage NVIDIA's optimized PyTorch Docker image for maximum performance on the B10 chip.
*   **UV for Dependency Management**: Use `uv` to create a virtual environment, allowing you to pin specific library versions while utilizing the optimized PyTorch from the base image.
*   **Block PyTorch with UV**: Prevent `uv` from reinstalling PyTorch by using `override-dependencies` in your `pyproject.toml`.
*   **`TORCH_CUDA_ARCH_LIST` Override**: Correctly set or unset `TORCH_CUDA_ARCH_LIST` to `12.0` for successful `xformers` compilation.
*   **Custom `xformers` Build**: Install `xformers` from a custom source branch that supports CUDA 12.1 until the official merge.
*   **Upgrades**:  When upgrading base image - virtual environment needs to be recreated
*   **Full repo with code**: [code is here](https://github.com/riomus/dgx-spark-unsloth)

## Unsloth Your DGX Spark: A Deep Dive

The NVIDIA DGX Spark, a compact yet capable development box, offers an enticing platform for local large model fine-tuning with its impressive 128GB unified memory on a B10 chip. While not a speed demon, its memory capacity makes it a serious contender for experimenting with substantial models. However, getting the most out of it, especially with performance-critical libraries like Unsloth, requires a careful setup.

### The Foundation: NVIDIA's Optimized PyTorch

NVIDIA strongly recommends using their official Docker image for PyTorch development on DGX Spark. This isn't just a suggestion; it's a performance imperative. The image, specifically [nvidia/pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch), contains a custom PyTorch build highly optimized for NVIDIA GPUs, including the Blackwell architecture found in the DGX Spark (B10 chip). It also includes pre-optimized versions of `triton` and `torchvision`. Using this image simplifies dependency management and ensures you're squeezing every bit of performance out of your hardware.

### The Challenge: Unsloth's Specific Requirements

Unsloth, a library designed to accelerate large language model fine-tuning, has specific version requirements for its dependencies, particularly those tied to CUDA and PyTorch versions. Mismatched versions can severely degrade performance or even prevent the library from working correctly.

The DGX Spark's B10 chip boasts CUDA compute capability 12.1, requiring CU130. However, some critical libraries like `fast-attention` and `xformers` don't yet have full, officially merged support for CUDA 12.1. This is a common hurdle with cutting-edge hardware. For instance, you can track `xformers` progress here: [facebookresearch/xformers pull/1344](https://github.com/facebookresearch/xformers/pull/1344).

### The Solution: `uv` and Strategic Overrides

To achieve optimal performance with Unsloth on DGX Spark, we need to combine the best of both worlds: the official NVIDIA PyTorch image and a flexible dependency manager like `uv` to pin specific versions and handle custom builds.

1.  **Creating a `uv` Virtual Environment**:
    We'll create a virtual environment using `uv` that utilizes the system's pre-installed packages, specifically the optimized PyTorch from the NVIDIA Docker image. We'll then instruct `uv` to *not* reinstall PyTorch, Triton, and Torchvision.

    ```bash
    uv venv --system-site-packages
    ```

2.  **Blocking PyTorch Installation with `pyproject.toml`**:
    To prevent `uv` from trying to install its own versions of PyTorch, Triton, and Torchvision (which would overwrite the optimized ones from the NVIDIA image), we use the `override-dependencies` feature in `pyproject.toml`. This effectively tells `uv` to ignore these packages when resolving dependencies.

    ```toml
    # pyproject.toml
    [tool.uv]
    override-dependencies = [
      "torch; python_version < '0'",
      "triton; python_version < '0'",
      "torchvision; python_version < '0'"
    ]
    ```

    The `python_version < '0'` trick is a way to make the dependency impossible to satisfy, effectively blocking its installation without causing an error.

3.  **Handling `TORCH_CUDA_ARCH_LIST`**:
    The NVIDIA Docker image exposes a broad `TORCH_CUDA_ARCH_LIST` (`"8.0 8.6 9.0 10.0 11.0 12.0+PTX"`). When compiling libraries like `xformers`, this extensive list can sometimes cause parsing issues or unnecessary compilation for architectures you don't need. Since the DGX Spark is primarily a development box and we're building for local use, we can simplify this. The DGX Spark's compute capability is 12.1, but `xformers` generally works well with `12.0`.

    You can either `unset` the variable or explicitly set it:

    ```bash
    # Option 1: Unset it (often sufficient if the build system defaults to a reasonable value)
    unset TORCH_CUDA_ARCH_LIST

    # Option 2: Explicitly set it to 12.0
    export TORCH_CUDA_ARCH_LIST=12.0
    ```

    It is recommended to use `export TORCH_CUDA_ARCH_LIST=12.0` for clarity and consistency during the build process.

4.  **Installing `xformers` from Source**:
    Given the ongoing work to fully support CUDA 12.1 in `xformers`, we'll install it from a specific branch that includes the necessary updates. This ensures we have the latest compatible version. We also need to tell `uv` to *not* use build isolation for `xformers` to allow it to pick up the existing CUDA environment correctly.

    ```toml
    # pyproject.toml
    dependencies = [
        "xformers",
        # ... other unsloth dependencies
    ]

    [tool.uv.sources]
    xformers = {git="https://github.com/johnnynunez/xformers", branch="main"}

    [tool.uv]
    no-build-isolation-package= ["xformers"]
    ```

### Ready for Fine-tuning with Jupyter Lab

With all these pieces in place, your DGX Spark is now primed for high-performance fine-tuning with Unsloth. The next step is to set up a Jupyter Lab environment to easily experiment with your models.

1.  **Install a Custom Jupyter Kernel**:
    Install an `ipykernel` that points to your newly created `uv` virtual environment. This ensures Jupyter Lab uses the correct Python interpreter and all your carefully installed dependencies.

    ```bash
    uv run python -m ipykernel install --user --name venv --display-name "Python (.venv)"
    ```

2.  **Run Jupyter Lab**:
    Finally, launch Jupyter Lab from within your `uv` environment.

    ```bash
    uv run jupyter lab
    ```

    You should now be able to select the "Python (.venv)" kernel in Jupyter Lab and begin fine-tuning large language models with Unsloth, leveraging the full power of your DGX Spark.


## References

*   **NVIDIA PyTorch Container**: [https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
*   **Xformers CUDA 12.1 Support Pull Request**: [https://github.com/facebookresearch/xformers/pull/1344](https://github.com/facebookresearch/xformers/pull/1344)
*   **UV Project**: [https://astral.sh/uv](https://astral.sh/uv)
*   **Complete Example Repository**: [https://github.com/riomus/dgx-spark-unsloth](https://github.com/riomus/dgx-spark-unsloth)

## Takeaways

*   **Prioritize NVIDIA's PyTorch Docker image** for unparalleled performance on DGX Spark.
*   **`uv` is your friend** for managing complex Python dependencies, especially when mixing system packages with custom builds.
*   **Be explicit with `pyproject.toml`**: Use `override-dependencies` to avoid conflicts and `no-build-isolation-package` for custom source installations when Torch is needed (like for `xformers`).
*   **Mind your `TORCH_CUDA_ARCH_LIST`**: Incorrect settings can lead to compilation failures or degraded performance.
*   **Stay updated on library support**: Especially for cutting-edge hardware, direct source installations might be necessary until official support is merged.
*   **DGX Spark + Unsloth is a powerful combo**: With the right setup, its 128GB of memory makes it excellent for local LLM fine-tuning experiments.
*   **Cross-dependencies**: Because base image provides some libs like PyTorch, Torchvision etc - when version of base image is changed - whole Python virtual environemtn needs to be recreated (things might work, but there is a chance native packages will get broken)


### Prompt

Post was generated using AI - here is the prompt if you do not have time to read the story

```
Write Blog post using markdown for jekyll blog. 

title: Unsloth your DGX Spark

Add sections like
TL/DR - key points of the post
<main body> - all the sections with main contnent
References - all the links at the end
Take aways - things that are worth to remember

Do not use to many emojis. Add links where possible, use code blocks for readibility. 

Data for the post
- DGX spark is a small devbox to play with local model development on B10 chip - it is not fast but has 128GB unified memory- so it is not bad for large model fine-tuning
- NVIDIA recommends usage of their docker image when using pytorch in dgx spark https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- using nvidia docker image simplifies management of dependencies that are crucial for performance - you will squeeeze out max performance from DGX Spark by using latest NVidia pytorch image
- that image contains a special pytorch build optimised for nvidia + blackweell (dgx spark is b10 chip), triton, torchvision
- unsloth requires multiple specific versions + libraries that depends on cuda and torch versions - missing bits will degrade performance
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