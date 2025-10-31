---
layout: post
title: Oh no 128GB is not enaught for 7B parameters!
excerpt_separator:  <!--more-->
tags: [DGX Spark, GPU]
---

## TL/DR

If you're working with large language models (LLMs) on systems like the DGX Spark, and encountering "out of memory" errors despite having seemingly ample RAM (e.g., 128GB for a 7B parameter model), the culprit might be your operating system's caching mechanisms. The solution is often as simple as dropping system caches.

*   **DGX Spark uses UMA (Unified Memory Architecture):** CPU and GPU share the same memory.
*   **OS Caching:** The OS aggressively uses memory for caches, which might not be visible to GPU tools.
*   **CUDA vs. Actual Usage:** DGX Dashboard's memory usage (via CUDA API) might show high usage even without a model loaded due to OS caches.
*   **The Fix:** Clear system caches with `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`.
*   **It is mentioned in NVidia docs** - [check it here](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html#memory-reporting-differences-with-unified-memory-architecture)
<!--more-->
## The Mystery of the Missing Memory

You've got a machine with quite a lot of DRAM/VRAM, like a DGX Spark, equipped with 128GB of RAM. You're excited to dive into the world of large language models, like a 7B parameter model. On paper, 128GB should be more than sufficient. A 7B model, even in a less quantized format, might comfortably fit within that. Yet, you hit a wall:

```
ValueError: Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules in 32-bit, you need to set ⁠ llm_int8_enable_fp32_cpu_offload=True ⁠ and pass a custom ⁠ device_map ⁠ to ⁠ from_pretrained ⁠. Check https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu for more details.
```

This error message from Hugging Face Transformers is a classic indicator that the system believes you don't have enough GPU RAM. But how can that be?

### Understanding Unified Memory Architecture (UMA)

The DGX Spark, leverages a Unified Memory Architecture (UMA). This means that the CPU and the powerful B10 GPU chip share the *same physical memory*. It's a dynamic architecture where the GPU has direct access to this shared, not-carved-out memory pool.

This design offers significant advantages in terms of data transfer efficiency, as data doesn't need to be copied between separate CPU and GPU memory spaces. However, it also introduces a layer of complexity when it comes to memory accounting.

### The Invisible Memory Hog: OS Caching

Here's where the plot thickens. Your operating system is a sophisticated piece of software designed for efficiency. One of its key strategies is aggressive caching. The OS will use available memory to cache frequently accessed files, disk blocks, and other system data. This is generally a good thing, as it speeds up subsequent operations.

The problem arises because the GPU, from its perspective, isn't fully aware of what parts of this shared memory are genuinely free and what parts are merely occupied by system caches that *could* be released. The OS can dynamically unswap and lazily free up this cached memory when an application (like your LLM) actually needs it, but there's a delay, and the initial allocation request might fail.

Furthermore, monitoring tools, such as the DGX Dashboard, often report memory usage based on the CUDA API. It might not fully account for the memory currently consumed by the operating system's internal caches. This can lead to a confusing situation where the dashboard shows high memory usage even when no AI model is explicitly loaded, making it seem like your precious 128GB is already gone!

### The Simple Solution: Drop Caches

Fortunately, the fix for this common issue is surprisingly straightforward. You can instruct the Linux kernel to drop its caches, freeing up that "invisible" memory for your applications.

To do this, simply run the following command in your terminal:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```


### Prompt

Post was generated using AI - here is the prompt if you do not have time to read the story

```
Write Blog post using markdown for jekyll blog. 

title:  Oh no 128GB is not enaught for 7B parameters!

Add sections like
TL/DR - key points of the post
<main body> - all the sections with main contnent

Do not use too many emojis. Add links where possible, use code blocks for readibility. 

Data for the post
- DGX spark is based on UMA - same memory is used by CPU and B10 GPU chip - what is more it is dynamich architecture where GPU has access to same not-carved out memory 
- OS is using memory for caching 
- OS can dynamically unswap and lazely free up memory
- GPU is not aware what part of memory is occupied and what part can be released by memory because it is a cache etc
- When using GPU you might encounter errors like


ValueError: Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules in 32-bit, you need to set ⁠ llm_int8_enable_fp32_cpu_offload=True ⁠ and pass a custom ⁠ device_map ⁠ to ⁠ from_pretrained ⁠. Check https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu for more details.



- Error is caused by memory being occupied by system caches etc
- To solve simply run 

sudo sh -c 'sync; echo 3 > ∕proc∕sys∕vm∕drop_caches'

- What is more - DGX Dashboard is showing memory usage using CUDA API - so without any model loaded - you might see almost whole memory occupied

```
