# 06 — Project Task: Your First GPU Tensor

> The Week 1 deliverable. Small in scope, big in importance — every other week of the track assumes that you have a working Colab + GPU + PyTorch setup.

---

## The Task

> **Set up a Google Colab notebook with the GPU runtime enabled, verify the GPU is active in PyTorch, create a random tensor shaped like a galaxy image, and move it to the GPU.**

This is the bare-minimum smoke test that confirms you can run everything we'll build over the next five weeks.

---

## Starter and Solution Notebooks

You don't have to write the notebook from scratch — open the starter:

- 🟢 **Start here:** [`notebooks/week1_starter.ipynb`](notebooks/week1_starter.ipynb)
- 🟡 **Solution (only after attempting):** [`notebooks/week1_solution.ipynb`](notebooks/week1_solution.ipynb)

Both are designed to open in Colab directly. See the "Opening in Colab" section below.

---

## Step-by-Step Checklist

Each step maps to a cell (or a small group of cells) in the starter notebook. Tick them off as you go.

- [ ] **Step 0.** Open the starter notebook in Colab and switch to a **GPU runtime** (`Runtime → Change runtime type → GPU`).
- [ ] **Step 1.** Run `!nvidia-smi` and confirm an NVIDIA GPU appears in the output.
- [ ] **Step 2.** Import `torch`, `torchvision`, and `matplotlib.pyplot` and print their versions.
- [ ] **Step 3.** Print `torch.cuda.is_available()`. Expect `True`. Also print `torch.cuda.get_device_name(0)`.
- [ ] **Step 4.** Define `device = "cuda" if torch.cuda.is_available() else "cpu"` and print it.
- [ ] **Step 5.** Create a random tensor of shape `(3, 64, 64)` (one toy galaxy image). Print its `.shape`, `.dtype`, and `.device`.
- [ ] **Step 6.** Move the tensor to the GPU with `.to(device)`. Print its `.device` again and confirm it now reads `cuda:0`.
- [ ] **Step 7.** Perform a sanity matmul: take a `(64, 64)` slice of one channel and matrix-multiply it with itself. Print the result's shape.
- [ ] **Step 8.** Save a copy of the completed notebook to your fork of this repo (or to your Drive).

---

## Acceptance Criteria

Your submission counts as complete when, on a fresh Colab session:

1. **Runtime → Restart and run all** executes the entire notebook top-to-bottom **without errors**.
2. `torch.cuda.is_available()` prints `True`.
3. After moving the tensor to the GPU, its `.device` prints `cuda:0` (or similar).
4. The notebook is saved to your Drive *or* pushed to your fork of the CSoT repo so a mentor can review it.

---

## Stretch Goals (Optional but Recommended)

If you finish the core task with time to spare, try one or more of these — they directly preview things we'll do in later weeks.

### 1. Benchmark CPU vs GPU

Adapt the benchmark snippet from [`03-gpu-acceleration.md`](03-gpu-acceleration.md) and report the speedup for matmul of `(4096, 4096)` matrices.

> Bonus: plot CPU time vs GPU time for sizes `[256, 512, 1024, 2048, 4096]` using `matplotlib`. Does the speedup grow with size? Why might that be?

### 2. Play with `dtype`

Create the same `(3, 64, 64)` tensor with `dtype=torch.float16` and `dtype=torch.float64`. Compare their `.element_size()` and total memory footprint. Discuss when you might use each:

- `float16` (half precision) — much faster on modern GPUs, less memory, **but** less numerical range. Common in mixed-precision training.
- `float64` (double precision) — overkill for ML, standard in scientific computing.
- `float32` (single precision) — the default for ML, the right choice almost always.

### 3. Visualise a "Synthetic Galaxy"

Build a tensor that *looks* a bit like a galaxy and display it with `matplotlib`:

```python
import torch, math
import matplotlib.pyplot as plt

H, W = 128, 128
y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
r = torch.sqrt(x**2 + y**2)
theta = torch.atan2(y, x)

# An exponentially declining brightness profile (very loosely Sersic n=1)
disk = torch.exp(-r * 4)

# Add a faint two-arm spiral pattern
arms = 0.5 * torch.exp(-r * 4) * (1 + torch.cos(2 * theta + 8 * r))

galaxy = (disk + arms).clamp(0, 1).numpy()
plt.imshow(galaxy, cmap="inferno")
plt.axis("off")
plt.show()
```

This is *not* physics; it's a cartoon. But playing with the formula will give you intuition for what surface brightness profiles look like (which we'll formalise in Week 3 with **isophotes** and Week 4 with the **Sérsic profile**).

---

## Opening the Notebook in Colab

Three options, in order of how this track will scale over future weeks:

### Option A — From a GitHub fork (recommended)

1. Fork the [CSoT repo](../../../README.md) on GitHub.
2. In Colab: **File → Open notebook → GitHub** tab.
3. Paste your fork's URL or search for it, then open `ML-Astronomy/Week-1/notebooks/week1_starter.ipynb`.
4. Edits autosave back to Drive; when you want to push back to your fork, use **File → Save a copy in GitHub**.

### Option B — Direct GitHub link

If your fork lives at `github.com/YOUR_USER/csot-ml-astronomy`, paste this URL into Colab's "GitHub" tab:

```
https://github.com/YOUR_USER/csot-ml-astronomy/blob/main/ML-Astronomy/Week-1/notebooks/week1_starter.ipynb
```

### Option C — Upload

1. Download `week1_starter.ipynb` from this folder.
2. In Colab: **File → Upload notebook**.

---

## Submission

Submission logistics will be confirmed via the official CAIC channels. The default expectation is:

- Push your completed notebook into a `submissions/<your-name>/week1/` folder in your fork.
- Or share the Colab "share link" (with comment access) with your mentor.

When in doubt, ask in the mentor channel **before** the week ends.

---

## Reflection Questions

Once you finish, write 2–3 sentences in a Markdown cell at the bottom of your notebook reflecting on:

1. What was the most confusing part of getting Colab + GPU + PyTorch set up? How did you resolve it?
2. Pick one galaxy class (E / S / S0 / Irr) and describe in your own words what features a CNN might need to detect to recognise it.
3. Why do you think we're going to spend an entire week on **data pipelines** next, before any model training?

These reflections won't be graded — they exist so you can come back in week 6 and laugh at how far you've come.

---

⬅️ Previous: [`05-galaxy-morphologies.md`](05-galaxy-morphologies.md) | 📚 See also: [`resources.md`](resources.md)
