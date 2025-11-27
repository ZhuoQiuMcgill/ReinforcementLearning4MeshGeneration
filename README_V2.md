# RL Mesh Generation – v2 Quickstart

This document is for new users who want to **reproduce the paper results** or
**train their own models** using the **v2 RL subsystem** only.

You do **not** need to touch the legacy scripts under `rl/`, `general/`, or the
root `config` file. v2 is a self-contained project.

- v2 code: `v2/src/mesh_rl`
- v2 domains: `v2/data/domains/*.json`
- v2 outputs (logs, models, evaluation): `v2/outputs/...`
- v2 training configs: `v2/config/*.yaml`

---

## 1. Prerequisites

1. **Clone this repository** and move into its root directory.
2. Create the dedicated v2 RL environment from the provided conda env file
   (this installs **all dependencies except PyTorch**):

   ```bash
   conda env create -f environment_RLMeshV2.yml
   conda activate RLMeshV2
   ```

3. Install **PyTorch** (and optionally `torchvision`, `torchaudio`) by
   following the official instructions for **your CUDA / driver version**:

   - First, check your local CUDA / driver version (for example using
     `nvidia-smi`).
   - Then go to the official PyTorch site and use the **"Get Started" →
     "Start locally"** configurator to select:
       - OS: your system (e.g. Windows)
       - Package: `conda` or `pip` (both work inside `RLMeshV2`)
       - Language: `Python`
       - Compute Platform: the CUDA version that matches your driver, or
         `CPU` if you do not want GPU acceleration.
   - Copy the command it shows (for example a `pip install torch torchvision ...`
     line) and run it **inside the activated `RLMeshV2` environment**.

   This avoids CUDA / PyTorch mismatch issues and lets each user install the
   correct wheel for their own GPU.

> The conda env file already includes Gymnasium, Stable-Baselines3 2.x and all
> other v2 dependencies. Only the PyTorch stack is intentionally left out so
> that you can install the correct build for your hardware.

---

## 2. Directory layout (v2 only)

You only need to care about these paths for v2:

- `v2/src/mesh_rl` – v2 RL library (`mesh_rl` Python package).
- `v2/data/domains` – JSON domain files used by the environment.
- `v2/outputs` – where v2 writes logs, models and evaluation summaries.
- `v2/docs` – design docs and technical notes.

v2 **never** reads from `ui/domains` or the root `config` at runtime. All
runtime data is under `v2/`.

---

## 3. Step 1 – Choose a domain JSON

1. List the available v2 domains:

   ```bash
   ls v2/data/domains
   ```

   You should see files such as:

   - `dolphine3.json`
   - `random1_1.json`
   - ...

2. Pick one file. The **domain key** is the filename **without** `.json`.

   Examples:

   - `v2/data/domains/dolphine3.json` → domain key: `dolphine3`
   - `v2/data/domains/random1_1.json` → domain key: `random1_1`

You will pass this domain key to the training and evaluation commands.

---

## 4. Step 2 – Train on a domain (command-line)

From the **repository root**, ensure `v2/src` is on `PYTHONPATH` so that
Python can import the local `mesh_rl` package:

```bash
# From wherever you cloned the repo
git clone <this-repo-url>
cd ReinforcementLearning4MeshGeneration
conda activate RLMeshV2

# On PowerShell (from the repo root)
$env:PYTHONPATH = "$PWD/v2/src"
```

The training CLI supports **two ways** to specify hyperparameters:

1. Use only CLI flags (no config file).
2. Provide a YAML config via `--config` and optionally override some fields
   on the CLI.

### 4.1 Train with CLI flags only (no config)

If you do **not** pass `--config`, the CLI uses only its own defaults and
your explicit flags. The effective defaults are:

- `algo`: `sac`
- `domain`: `dolphine3`
- `steps` (total_timesteps): `1_500_000`
- `seed`: `999`
- `version`: `v2_run`
- `device`: `auto` (resolved to `cuda` if available, otherwise `cpu`)

So the **minimal command** (using all defaults) is:

```bash
python -m mesh_rl.cli.train_cli
```

A more explicit example on the `basic` domain with 100k steps is:

```bash
python -m mesh_rl.cli.train_cli --algo sac --domain basic --steps 100000 --seed 999 --version my_basic_run
```

Explanation of key arguments:

- `--algo` – RL algorithm: one of `sac`, `ppo`, `a2c`, `ddpg`, `td3`.
  - For reproducing the paper, `sac` is usually the default.
- `--domain` – domain key; must match a file `v2/data/domains/<key>.json`.
- `--steps` – total training timesteps (per curriculum stage).
- `--seed` – random seed for reproducibility.
- `--version` – an experiment tag; used to build output paths.

Internally, the library also applies **algorithm-specific defaults**
(network size, learning rate, etc.) that reproduce the legacy
`RL_Mesh.py` behaviour. You can override those via a YAML config (next
section).

### 4.2 Train with a YAML config (and optional overrides)

If you provide `--config <path>`, the CLI will:

1. Load the YAML file into a base config.
2. Override any fields that you also pass via CLI flags.

For example, with the provided
`v2/config/train_sac_basic_legacy.yaml`:

```bash
python -m mesh_rl.cli.train_cli --config v2/config/train_sac_basic_legacy.yaml
```

This uses **exactly** the hyperparameters from the YAML (including
`algo_kwargs`), which mirror the legacy `RL_Mesh.py` SAC settings for a
single-stage run.

You can still override individual fields from the YAML on the CLI. For
example, to change only the domain and version tag while keeping all
other settings:

```bash
python -m mesh_rl.cli.train_cli --config v2/config/train_sac_basic_legacy.yaml \
  --domain dolphine3 --version sac_dolphine3
```

### 4.3 Where does training write its outputs?

For the example above (SAC, `version=my_experiment`), v2 will write to:

- Logs & model:

  ```text
  v2/outputs/logs/sac/my_experiment/curriculum/0/mesh
  ```

  The underlying Stable-Baselines3 implementation will actually create
  `mesh.zip` at that location.

- TensorBoard logs:

  ```text
  v2/outputs/tensorboard/sac/my_experiment/curriculum/0/
  ```

You can inspect these folders after training finishes.

---

## 5. Step 3 – Evaluate a trained model (command-line)

After training, you can evaluate the model on one or more domains using the
v2 evaluation CLI.

Using the same example as above (SAC on `dolphine3`):

```bash
python -m mesh_rl.cli.eval_cli \
  --algo sac \
  --models v2/outputs/logs/sac/my_experiment/curriculum/0/mesh \
  --domains dolphine3 \
  --version my_experiment_eval
```

Explanation of arguments:

- `--algo` – algorithm name; must match how the model was trained.
- `--models` – comma-separated list of model paths.
  - Each path is the **base path** used when saving a model
    (e.g. `.../mesh`, which SB3 stores as `mesh.zip`).
- `--domains` – comma-separated list of domain keys to evaluate on.
- `--version` – evaluation tag; used to build evaluation output paths.

### 5.1 Evaluation outputs

The CLI runs a full episode per `(model, domain)` pair and collects:

- `completed` – whether the environment reported a complete mesh.
- `n_elements` – number of elements generated.

The summary is written to:

```text
v2/outputs/evaluation/my_experiment_eval/evaluation_summary.json
```

The JSON structure is:

```json
{
  "mesh": {
    "completed": [0 or 1 per domain],
    "n_elements": [int per domain]
  }
}
```

Here `"mesh"` is the stem of the model path (`.../mesh`).

---

## 5. Step 3 – (Optional) YAML config structure reference

YAML config files live under `v2/config`. The provided
`train_sac_basic_legacy.yaml` looks like this:

```yaml
algo: sac
domain: basic
total_timesteps: 4000000
seed: 999
version: "77"
device: auto
algo_kwargs:
  learning_rate: 0.0003
  learning_starts: 10000
  batch_size: 100
  gamma: 0.5
  policy_kwargs:
    activation_fn: relu
    net_arch: [128, 128, 128]
```

- Top-level keys (`algo`, `domain`, `total_timesteps`, `seed`,
  `version`, `device`) map directly to `RLConfig` fields.
- `algo_kwargs` is passed straight to the Stable-Baselines3 constructor
  for the chosen algorithm and lets you override anything that was
  hard-coded in the legacy script (learning rate, batch size, network
  architecture, etc.).

During startup, the training CLI always prints a short summary including
the resolved device, log directory and model path, followed by a live
progress bar (based on `tqdm`/`rich`) during training.

---

## 6. Step 4 – Use the Python API instead of CLIs (optional)

If you prefer to call v2 from your own scripts or notebooks, you can use the
Python API directly.

### 6.1 Single-environment training

```python
from pathlib import Path
from mesh_rl import RLConfig, make_default_paths, train_single_env

# Assume this file is located inside the repository tree
project_root = Path(__file__).resolve().parents[2]
paths = make_default_paths(project_root)

cfg = RLConfig(
    algo="sac",          # sac, ppo, a2c, ddpg, td3
    domain="dolphine3",  # must match v2/data/domains/dolphine3.json
    total_timesteps=100_000,
    seed=123,
    version="my_experiment",
)

model_path = train_single_env(cfg, paths=paths)
print("Model directory:", model_path.parent)
```

### 6.2 Curriculum training

```python
from pathlib import Path
from mesh_rl import (
    RLConfig,
    make_default_paths,
    CurriculumStage,
    train_curriculum,
)

project_root = Path(__file__).resolve().parents[2]
paths = make_default_paths(project_root)

cfg = RLConfig(
    algo="sac",
    domain="random1_1",
    total_timesteps=50,
    seed=123,
    version="curriculum_example",
)

stages = [
    CurriculumStage(index=0, domain="random1_1", timesteps=50),
    CurriculumStage(index=1, domain="random1_1", timesteps=50),
]

final_model = train_curriculum(cfg, paths=paths, stages=stages)
print("Final curriculum model:", final_model)
```

### 6.3 Evaluation

```python
from pathlib import Path
from mesh_rl import EvalConfig, evaluate_models

project_root = Path(__file__).resolve().parents[2]

model_paths = [
    project_root / "v2" / "outputs" / "logs" / "sac" / "my_experiment" / "curriculum" / "0" / "mesh",
]

eval_cfg = EvalConfig(
    algo="sac",
    model_paths=model_paths,
    domains=["dolphine3"],
    version="my_experiment_eval",
    deterministic=False,
    render=False,
)

results = evaluate_models(eval_cfg, project_root=project_root)
print("Evaluation results:", results)
```

---

## 7. Step 5 – Add your own domain JSON

You can train on your **own geometry** by adding a new domain file under
`v2/data/domains`.

1. Pick an existing JSON as a template, for example:

   ```text
   v2/data/domains/dolphine3.json
   ```

2. Copy it and rename, for example:

   ```text
   v2/data/domains/my_shape.json
   ```

3. Edit `my_shape.json` to describe your own polygon (boundary vertices and
   related metadata). Keep the overall structure consistent with the template.

4. Use `my_shape` as the domain key in training/evaluation commands:

   ```bash
   python -m mesh_rl.cli.train_cli --algo sac --domain my_shape --steps 100000 --seed 123 --version my_shape_exp
   ```

v2 will automatically resolve `my_shape` to `v2/data/domains/my_shape.json`.

---

## 8. Step 6 – Run the v2 test suite (optional)

v2 comes with a small pytest-based test suite to check that training and
evaluation are wired correctly and that behaviour matches the legacy
implementation.

From the repository root (with `RLMeshV2` activated and `PYTHONPATH`
set to include `v2/src`):

```bash
pytest v2/tests/mesh_rl
```

Some tests are smoke tests that train/evaluate very small models. They require
Stable-Baselines3, Gymnasium, PyTorch and TensorBoard; if these libraries
are missing, the tests will automatically skip.

---

## 9. Legacy vs v2 (what you can ignore)

- The original project uses:
  - `rl/boundary_env.py`, `rl/baselines/RL_Mesh.py`, `rl/baselines/testbed.py`,
  - geometry helpers under `general/`,
  - a root `config` file pointing to `ui/domains`.
- The **v2** subsystem:
  - Re-implements the same behaviour under `v2/src/mesh_rl`.
  - Reads domains only from `v2/data/domains`.
  - Writes outputs only under `v2/outputs`.

If you only care about using the method (training/evaluation) and not about the
historical code, you can safely treat the **v2** paths and commands in this
`README_V2.md` as the **canonical** way to use the project.
