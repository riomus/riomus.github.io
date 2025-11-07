---
layout: post
title: Is it worht to use offcial Nvidia PyTorch image?
excerpt_separator:  <!--more-->
tags: [DGX Spark, GPU, Performance, Nvidia, PyTorch]
---

## TL/DR

*   **Significant Performance Boost:** Using the official NVidia PyTorch Docker image resulted in a 50% increase in TFLOPS for a specific matrix multiplication task compared to a native PyTorch installation.
*   **Essential for DGX Spark:** If you're running workloads on a DGX Spark (or similar NVidia hardware), the official image appears to be a must-have for maximizing performance.
*   **Custom Optimizations:** The performance gain likely stems from NVidia's custom PyTorch fork and optimized PTX code within their official image.
*   **Code:** [repo](https://github.com/riomus/dgx-spark-performance) with code of the experiment
*   **Important note:**  That is a simple check using mat-mul just to verify if there is any difference - and there is. Will try to do PyTorch bench in next post.

<!--more-->

Recently, I stumbled upon a fascinating code snippet by [@awnihannun](https://x.com/awnihannun/status/1982880363765768288) that piqued my interest. It prompted me to conduct a quick experiment on our DGX Spark to see if NVidia's official PyTorch Docker image truly offers a performance advantage over a standard PyTorch installation.

The official NVidia PyTorch images, available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch), are often touted for their optimized performance. But how much of a difference do they actually make in a real-world scenario?

### The Experiment Setup

To test this, I adapted Awni Hannun's code to perform a specific, compute-intensive operation: 8192x8192 bfloat16 square matrix multiplication, repeated 50 times per iteration, for 100 repetitions. The goal was to measure the average TFLOPS achieved in both environments.

**The Code:**

```python
import time
import torch

d = 8192
x = torch.randn(size=(d, d)).to(torch.bfloat16).to("cuda")
y = torch.randn(size=(d, d)).to(torch.bfloat16).to("cuda")

def fun(x):
    for _ in range(50):
        x = x @ y.T
    return x

# Warm-up
for _ in range(10):
    fun(x)
    torch.cuda.synchronize()

tic = time.time()
repetitions = 100
for _ in range(repetitions):
    fun(x)
    torch.cuda.synchronize()
toc = time.time()

s = (toc - tic)
msec = 1e3 * s
tf = (d**3) * 2 * 50 * repetitions / (1024**4) # Calculate theoretical TFLOPS
print(f"{msec=:.3f}")
tflops = tf / s
print(f"{tflops=:.3f}")
```

This code sets up two large bfloat16 tensors on the GPU and then performs a series of matrix multiplications. The `torch.cuda.synchronize()` calls ensure that all GPU operations are complete before timing.

### Native PyTorch Run

First, I ran the code with a native PyTorch installation on the DGX Spark.

**PyTorch Version:** `2.9.0+cu130`

**Results:**
`msec=87755.411`
`tflops=56.977`

### Dockerized PyTorch Run (NVidia Official Image)

Next, I executed the same code within the official NVidia PyTorch Docker image.

**PyTorch Version:** `2.9.0a0+145a3a7bda.nv25.10` (Note the `nv25.10` indicating NVidia's custom build)

**Results:**
`msec=57511.067`
`tflops=86.940`

### Analysis and Conclusion

During both runs, the DGX GPU was fully utilized, indicating that the differences observed are due to efficiency rather than resource contention.

Comparing the results:

*   **Native Run:** Approximately 57 TFLOPS
*   **Dockerized Run:** Approximately 87 TFLOPS

That's a **50% increase** in performance just by switching to the official NVidia PyTorch image! 


### Prompt

Post was generated using AI - here is the prompt if you do not have time to read the story

```
Write Blog post using markdown for jekyll blog. 

title:  Is it worht to use offcial NVidia PyTorch image?

Add sections like
TL/DR - key points of the post
<main body> - all the sections with main contnent


Based on [@awnihannun](https://x.com/awnihannun) [code](https://x.com/awnihannun/status/1982880363765768288) i have executed some small tests on DGX Spark to check if [NVidia official PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) image  allows you to run your code faster

Modified code to run  8192 bfloat16 square matrix multiplication 100 times and compute average tflops  


Code
import time
import torch
d = 8192
x = torch.randn(size=(d, d)).to(torch.bfloat16).to("cuda")
y = torch.randn(size=(d, d)).to(torch.bfloat16).to("cuda")

def fun(x):
    for _ in range(50):
        x = x @ y.T
    return x

for _ in range(10):
    fun(x)
    torch.cuda.synchronize()

tic = time.time()
repetitions = 100
for _ in range(repetitions):
    fun(x)
    torch.cuda.synchronize()
toc = time.time()
s = (toc - tic)
msec = 1e3 * s
tf = (d**3)  * 2 * 50 * repetitions / (1024 **4)
print(f"{msec=:.3f}")
tflops = tf / s
print(f"{tflops=:.3f}")


Results:
For native run:
2.9.0+cu130
msec=87755.411
tflops=56.977


For dockerized run
2.9.0a0+145a3a7bda.nv25.10
msec=57511.067
tflops=86.940

During both runs DGX GPU is fully utilized. 

Hard to say what differs inside of NVidia image - for sure PyTorch use PTX  + the pytorch code is a custom internal NVidia fork.

As shown by the results - IT IS A MUST to use official NVidia official PyTorch image to squeeze out more performance from the DGX Spark. 

That is a simple check just to verify if there is any difference - and there is - deeper and broader experiments could be done what are the differences for other operations and so one. 

[repo](https://github.com/riomus/dgx-spark-performance)
```
