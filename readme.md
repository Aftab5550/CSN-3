# Closeness in Syntactic Dependency Networks

Lab session: *Significance of network metrics* — **closeness group**.

Two independent C++ binaries:

| Binary | What it does | Output |
|---|---|---|
| `closeness_significance` | Monte Carlo p-values for mean closeness under (a) Erdős–Rényi `G(N,E)` and (b) the degree-preserving switching model | user-supplied path (default `results/table_2_significance.csv`) |
| `bounds` | Rigorous **analytical upper bounds** on `p(C_NH >= C_obs)` under `G(N,E)` only — Markov, higher-moment Markov, Cantelli, path-count Markov, path-count Chebyshev | `bounds_summary.csv`, `bounds_detail.csv`, `observed_closeness.csv` |

The MC pipeline answers *"is the metric significantly large?"*. The analytical pipeline gives an independent, rigorous confirmation that does not rely on sampling — useful as a cross-check and as an Advanced-exercise result.

---

## Repository structure

```
.
├── README.md
├── closeness_significance.cpp     # MC p-values (ER + switching)
├── bounds.cpp                     # analytical bounds (ER only)
│
├── data/                          # one file per language
│   ├── Arabic_dependency_network.txt
│   ├── Basque_dependency_network.txt
│   ├── Catalan_dependency_network.txt
│   ├── Chinese_dependency_network.txt
│   ├── Czech_dependency_network.txt
│   ├── English_dependency_network.txt
│   ├── Greek_dependency_network.txt
│   ├── Hungarian_dependency_network.txt
│   ├── Italian_dependency_network.txt
│   ├── Polish_dependency_network.txt
│   ├── Spanish_dependency_network.txt
│   └── Turkish_dependency_network.txt
│
├── results/                       # MUST exist before running closeness_significance
│   ├── table_2_significance.csv   # default output of the full MC run
│   ├── hungarian_T100_Q30.csv     # per-language runs, if any
│   ├── basque_T100_Q30.csv
│   └── ...
│
├── bounds_summary.csv             # produced by `bounds` (one row per language)
├── bounds_detail.csv              # produced by `bounds` (one row per bound tried)
└── observed_closeness.csv         # produced by `bounds`; reusable as --x-file
```

### Data file format

Each `<language>_dependency_network.txt` uses the **edge-list format** required by the assignment:

```
N E
word1 word2
word1 word3
...
```

- Header: `N` (number of vertices) then `E` (number of edges).
- Each subsequent line: the two word-strings that form an edge.
- Labels may be integers or strings; duplicates and self-loops are silently dropped by both binaries.
- **Undirected** — both loaders build a symmetric adjacency list.
- A file must correspond to a *single* language.

---

## Prerequisites

- A C++17 compiler (`g++ >= 7`, `clang++ >= 6`, or MSVC 2017+).
- **OpenMP** for multithreading (GCC's `libgomp`, Clang's `libomp`).
- Nothing else — no external libraries, no Python, no R.

---

## Building

From the repository root:

```bash
# MC pipeline
g++ -O3 -march=native -fopenmp -std=c++17 closeness_significance.cpp -o closeness_significance

# Analytical bounds
g++ -O3 -march=native -fopenmp -std=c++17 bounds.cpp -o bounds
```

Notes:

- Drop `-march=native` if you plan to ship the binaries elsewhere or run under virtualisation.
- Drop `-fopenmp` to compile single-threaded (slower, ~16× for the MC pipeline on a 16-core box). Both binaries have `#ifdef _OPENMP` guards, so they still build and run.
- For Clang on macOS with Homebrew `libomp`: `-L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp`.

**Pre-flight:**

```bash
mkdir -p results          # required by closeness_significance (it does not create the folder itself)
ls data/*.txt | wc -l     # should print 12
```

---

## Running the Monte Carlo pipeline

`closeness_significance` uses **positional arguments**, not flags:

```
closeness_significance  [T]  [Q]  [sample_ratio]  [language]  [out_path]
```

| Position | Default | Meaning | PDF recommendation |
|---|---|---|---|
| `T` | 25 | number of null graphs per model | ">20" |
| `Q` | 10 | switching trials = `Q*E` | ">=10, tune by coupon-collector" |
| `sample_ratio` | 0.05 | fraction of vertices sampled; **clamped to `[1000, 2500]`** | "10% or even smaller could work" |
| `language` | `ALL` | restrict to one language | — |
| `out_path` | `results/table_2_significance.csv` | where the CSV is written | — |

The **fifth argument** lets you send a per-language run to its own file — the master table is left untouched.

**Examples:**

```bash
# Full run, defaults — the report's baseline; writes results/table_2_significance.csv
./closeness_significance

# Fast exploration on the smallest language
./closeness_significance 5 3 0.02 Hungarian results/hungarian_quick.csv

# High-resolution run on one language, separate file
./closeness_significance 100 30 0.05 Basque results/basque_T100_Q30.csv
```

Each per-language file has its own header and single data row — easy to `tail -n1` into a master summary or to compare two configurations side by side.

---

## Running the analytical bounds

`bounds` uses **flags**:

```
bounds  [--data DIR]  [--x-file CSV]  [--save-x CSV]
        [--orders 1,2,3,4]  [--out CSV]  [--detail CSV]  [--threads N]
```

| Flag | Default | Purpose |
|---|---|---|
| `--data` | `data` | folder with `*_dependency_network.txt` |
| `--x-file` | — | CSV with `language` + `closeness` columns; **skips all BFS** |
| `--save-x` | `observed_closeness.csv` | where to cache computed closeness values |
| `--orders` | `1,2,3,4` | moment orders for the Markov bounds (order 1 always added) |
| `--out` | `bounds_summary.csv` | one row per language |
| `--detail` | `bounds_detail.csv` | one row per (language, bound) tried |
| `--threads` | 16 | OpenMP threads for the fallback BFS |

**Preferred workflow** — reuse the MC pipeline's `closeness` column:

```bash
./bounds --data data --x-file results/table_2_significance.csv
```

Runs in milliseconds per language. `bounds_detail.csv` shows which bound won (typically `path_chebyshev` with `L = 3` or `4`).

**Cold workflow** — no `--x-file`; BFS over all vertices with 16 threads:

```bash
./bounds --data data --threads 16
```

Writes `observed_closeness.csv` after *every* language, so an interrupted run resumes from the last completed line.

**CSV columns.**

`bounds_summary.csv`:

```
language,N,E,mean_k,delta,x_obs,U_old,p_old_markov,U_new,p_new_best,deterministic_max
```

`bounds_detail.csv`:

```
language,N,E,x_obs,bound,order,p_bound,value,note
```

The `bound` column contains one row per bound tried:
`old_markov`, `moment_markov` (one per requested order), `cantelli_m2`, `cantelli_range`, `path_chebyshev`, and `best`.



