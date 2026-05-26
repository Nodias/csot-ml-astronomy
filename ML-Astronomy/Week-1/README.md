# Week 1 — Environment Setup & PyTorch Tensors

> Getting comfortable with Google Colab and learning the fundamental data structure of PyTorch: the **Tensor**. In parallel, we meet **Edwin Hubble's tuning fork** — the morphological classification scheme our model will eventually learn.

---

## At a Glance

- **Time estimate:** 4–6 hours total (split across two or three sittings).
- **Prerequisites:** Basic Python (variables, lists, functions). A Google account for Colab.
- **Deliverable:** A completed Colab notebook based on [`notebooks/week1_starter.ipynb`](notebooks/week1_starter.ipynb) that creates a tensor and runs it on a GPU.

---

## Learning Objectives

| ML / Computer Science | Astronomy / Physics |
|-----------------------|---------------------|
| Set up a Google Colab notebook and switch to a GPU runtime. | Recount the historical context of Hubble's 1926 classification. |
| Explain what a **tensor** is and how it generalises matrices. | Sketch the **Hubble tuning fork** and name its main branches. |
| Create tensors with `torch.tensor`, `torch.randn`, `torch.zeros`. | Distinguish **elliptical**, **spiral** (barred / unbarred), **lenticular**, and **irregular** galaxies. |
| Inspect `.shape`, `.dtype`, `.device`. | Connect morphology to underlying astrophysics (gas content, star formation, dynamics). |
| Check GPU availability with `torch.cuda.is_available()` and move tensors with `.to(device)`. | Explain why morphology is a meaningful first cut for understanding galaxies. |
| Benchmark a CPU vs. GPU matrix multiplication and reason about the speedup. | Identify a handful of well-known galaxies (M31, M87, NGC 1300, NGC 4449) by their morphology. |

---

## Suggested Reading Order

Work through the markdowns in this order. The first three are ML; the next two are astronomy; then the project task ties them together.

1. [`01-getting-started-with-colab.md`](01-getting-started-with-colab.md) — set up Colab and switch to GPU runtime.
2. [`02-pytorch-tensors.md`](02-pytorch-tensors.md) — meet the tensor and the operations you'll use all summer.
3. [`03-gpu-acceleration.md`](03-gpu-acceleration.md) — move computation onto the GPU and measure the speedup.
4. [`04-hubble-tuning-fork.md`](04-hubble-tuning-fork.md) — the classification scheme behind every label in our dataset.
5. [`05-galaxy-morphologies.md`](05-galaxy-morphologies.md) — what ellipticals, spirals, and irregulars actually look like (and why).
6. [`06-project-task.md`](06-project-task.md) — the deliverable for this week.

For deeper dives, see [`resources.md`](resources.md).

---

## Notebooks

Both notebooks live in [`notebooks/`](notebooks/) and are designed to open directly in Colab:

- [`notebooks/week1_starter.ipynb`](notebooks/week1_starter.ipynb) — scaffold with `TODO` cells. **Start here.**
- [`notebooks/week1_solution.ipynb`](notebooks/week1_solution.ipynb) — reference implementation. **Open only after attempting the starter.**

To open in Colab, you can either:

1. Push your fork of this repo to GitHub and use Colab's **File → Open notebook → GitHub** picker.
2. Or download the `.ipynb` and use **File → Upload notebook** in Colab.

---

## Definition of Done

You're ready for Week 2 when you can, without referring to the solution:

- [ ] Open a Colab notebook with a **T4 GPU runtime** active and confirm it via `!nvidia-smi`.
- [ ] Import `torch` and print `torch.cuda.is_available()` → `True`.
- [ ] Create a random tensor of shape `(3, 64, 64)` and move it to the GPU.
- [ ] Print its `.shape` and `.device` and explain in your own words what each means.
- [ ] Draw or describe the Hubble tuning fork and place an example galaxy on each prong.

---

## Where to Ask for Help

- Stuck on Colab? Re-read [`01-getting-started-with-colab.md`](01-getting-started-with-colab.md) and check the "Common Pitfalls" section.
- Stuck on tensors? Read the PyTorch ["Tensors" tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) (linked in [`resources.md`](resources.md)).
- Stuck on morphology? Try the [Galaxy Zoo classification tutorial](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/).
- Still stuck? Drop a message in the CAIC mentor channel — that's exactly what mentors are for.
