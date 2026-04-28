"""MoMC wrapper for the maximum clique problem.

This module follows the shared GCO-HPIF solver-output contract used by the
Gurobi and CliSAT wrappers. The bundled third-party MoMC C source is compiled
locally and then invoked with the same run pattern used in the original
notebook workflow::

    gcc -O3 -DMOMC MOMC2016_1800sec_timeout_aware.c -o MoMC
    ./MoMC <graph_path.clq>

MoMC reads DIMACS ``.clq`` files with 1-based edge endpoints and reports clique
nodes using 1-based labels. Results returned by this module are converted back
to the repository's internal 0-based NetworkX node convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import shlex
import stat
import subprocess
import time
from typing import Sequence

import networkx as nx

from gco_hpif.solvers.common import is_valid_clique


DEFAULT_MOMC_SOURCE = Path("external/MOMC/src/MOMC2016_1800sec_timeout_aware.c")
DEFAULT_MOMC_EXECUTABLE = Path("external/MOMC/bin/MoMC")
DEFAULT_MOMC_CFLAGS = "-O3 -DMOMC"
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


@dataclass(frozen=True)
class ParsedMOMCOutput:
    """Parsed fields reported by MoMC for one graph instance."""

    instance_path: str | None
    max_clique_size: int | None
    clique_nodes_reported: list[int]
    solver_runtime_seconds: float | None
    prove_runtime_seconds: float | None
    branching_count: int | None
    timed_out: bool


@dataclass(frozen=True)
class MOMCCliqueResult:
    """Result returned by the MoMC maximum-clique wrapper."""

    status: str
    optimality_status: str
    runtime_seconds: float
    best_clique_size: int | None
    best_clique_nodes: list[int]
    clique_valid: bool
    return_code: int | None
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


def _clean_graph(graph: nx.Graph) -> nx.Graph:
    """Return a simple undirected graph with 0-based consecutive labels."""
    G = nx.Graph(graph)
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.convert_node_labels_to_integers(G, first_label=0, ordering="default")


def graph_to_dimacs_text(graph: nx.Graph, comment: str | None = None) -> str:
    """Convert a NetworkX graph to DIMACS ``.clq`` text for MoMC.

    GCO-HPIF stores graphs internally with 0-based node labels. DIMACS edge
    files are written with 1-based node ids, so node ``0`` becomes ``1``.
    """
    G = _clean_graph(graph)
    edges = sorted((min(int(u), int(v)) + 1, max(int(u), int(v)) + 1) for u, v in G.edges())

    lines = [
        "c",
        f"c {comment or 'Converted from GCO-HPIF NetworkX graph'}",
        "c",
        f"p edge {G.number_of_nodes()} {len(edges)}",
    ]
    lines.extend(f"e {u} {v}" for u, v in edges)
    return "\n".join(lines) + "\n"


def write_dimacs_file(graph: nx.Graph, output_path: str | Path, comment: str | None = None) -> Path:
    """Write one graph to a DIMACS ``.clq`` file and return the path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph_to_dimacs_text(graph, comment=comment), encoding="utf-8")
    return path


def _strip_ansi(text: str) -> str:
    """Remove terminal color/control codes from MoMC output before parsing."""
    return ANSI_ESCAPE_RE.sub("", text or "")


def parse_momc_stdout(stdout: str) -> ParsedMOMCOutput:
    """Parse MoMC stdout for clique size, clique nodes, and runtime.

    The parser accepts the output format used by the uploaded C source, e.g.::

        M 2 1
        s Instance path3.clq Max_CLQ 2 Branching 4 Time 0.01000000 \
          ProveBranching 0 ProveTime 0.00000000

    The original notebook extracted the maximum-clique size from the solver log,
    the clique node ids from the ``M`` line, and the runtime from the final
    ``Time ... ProveTime ...`` fields. This parser keeps the same information
    but stores it in the shared solver-output schema.
    """
    clean_stdout = _strip_ansi(stdout)

    summary_matches = re.findall(
        r"s\s+Instance\s+(?P<instance>.+?)\s+Max_CLQ\s+(?P<size>\d+)"
        r"(?:\s+Count\s+\d+)?\s+Branching\s+(?P<branching>\d+)"
        r"\s+Time\s+(?P<time>[0-9.eE+-]+)\s+ProveBranching\s+(?P<prove_branching>\d+)"
        r"\s+ProveTime\s+(?P<prove_time>[0-9.eE+-]+)",
        clean_stdout,
    )

    instance_path = None
    max_clique_size = None
    branching_count = None
    solver_runtime_seconds = None
    prove_runtime_seconds = None
    if summary_matches:
        instance_path, size, branching, runtime, _prove_branching, prove_runtime = summary_matches[-1]
        max_clique_size = int(size)
        branching_count = int(branching)
        solver_runtime_seconds = float(runtime)
        prove_runtime_seconds = float(prove_runtime)

    clique_nodes_reported: list[int] = []
    clique_lines = re.findall(r"^\s*M\s+(.+?)\s*$", clean_stdout, flags=re.MULTILINE)
    if clique_lines:
        clique_nodes_reported = [int(value) for value in clique_lines[-1].split()]

    timed_out = "Timeout reached" in clean_stdout

    return ParsedMOMCOutput(
        instance_path=instance_path,
        max_clique_size=max_clique_size,
        clique_nodes_reported=clique_nodes_reported,
        solver_runtime_seconds=solver_runtime_seconds,
        prove_runtime_seconds=prove_runtime_seconds,
        branching_count=branching_count,
        timed_out=timed_out,
    )


def _ensure_executable(path: Path) -> None:
    """Make a solver executable runnable if file permissions were lost."""
    if os.name == "nt":
        return
    mode = path.stat().st_mode
    if not (mode & stat.S_IXUSR):
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_momc_executable(
    source_path: str | Path = DEFAULT_MOMC_SOURCE,
    output_executable: str | Path = DEFAULT_MOMC_EXECUTABLE,
    cc: str = "gcc",
    cflags: str = DEFAULT_MOMC_CFLAGS,
    force: bool = False,
    compile_timeout_seconds: float = 900.0,
) -> Path:
    """Compile the bundled MoMC C source and return the executable path.

    By default this mirrors the notebook command, using ``gcc -O3 -DMOMC``.
    Tests may pass cheaper flags such as ``-O0 -DMOMC`` against tiny C fixtures.
    """
    source = Path(source_path)
    executable = Path(output_executable)

    if not source.exists():
        raise FileNotFoundError(f"MoMC source file not found: {source}")
    if executable.exists() and not force:
        _ensure_executable(executable)
        return executable

    executable.parent.mkdir(parents=True, exist_ok=True)
    command = [cc, *shlex.split(cflags), str(source), "-o", str(executable)]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=compile_timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "MoMC compilation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    _ensure_executable(executable)
    return executable


def ensure_momc_executable(
    momc_executable: str | Path = DEFAULT_MOMC_EXECUTABLE,
    source_path: str | Path = DEFAULT_MOMC_SOURCE,
    auto_compile: bool = True,
    force_compile: bool = False,
    cc: str = "gcc",
    cflags: str = DEFAULT_MOMC_CFLAGS,
    compile_timeout_seconds: float = 900.0,
) -> Path:
    """Return a usable MoMC executable, compiling from source when requested."""
    executable = Path(momc_executable)
    if executable.exists() and not force_compile:
        _ensure_executable(executable)
        return executable

    if not auto_compile:
        raise FileNotFoundError(
            f"MoMC executable not found: {executable}. "
            "Re-run with auto-compilation enabled or provide --momc-executable."
        )

    return build_momc_executable(
        source_path=source_path,
        output_executable=executable,
        cc=cc,
        cflags=cflags,
        force=True,
        compile_timeout_seconds=compile_timeout_seconds,
    )


def _run_momc_subprocess(
    momc_executable: str | Path,
    dimacs_path: str | Path,
    python_timeout_seconds: float | None = None,
) -> tuple[int | None, str, str, float, str]:
    """Run the external MoMC executable and return subprocess details.

    The command is equivalent to the original notebook command:

    ``./MoMC <graph_path>``
    """
    executable = Path(momc_executable)
    graph_file = Path(dimacs_path)

    if not executable.exists():
        raise FileNotFoundError(f"MoMC executable not found: {executable}")
    if not graph_file.exists():
        raise FileNotFoundError(f"DIMACS graph file not found: {graph_file}")

    _ensure_executable(executable)
    command = [str(executable), str(graph_file)]

    start_ns = time.perf_counter_ns()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=python_timeout_seconds,
        )
        elapsed = (time.perf_counter_ns() - start_ns) / 1_000_000_000
        return completed.returncode, completed.stdout, completed.stderr, elapsed, ""
    except subprocess.TimeoutExpired as exc:
        elapsed = (time.perf_counter_ns() - start_ns) / 1_000_000_000
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return None, stdout, stderr, elapsed, f"Python subprocess timeout after {python_timeout_seconds:.3f} seconds"


def _convert_reported_nodes_to_internal(nodes: Sequence[int], output_node_base: int) -> list[int]:
    if output_node_base not in {0, 1}:
        raise ValueError("output_node_base must be either 0 or 1.")
    return [int(node_id) - output_node_base for node_id in nodes]


def _runtime_from_solver_or_wall_clock(parsed: ParsedMOMCOutput, elapsed_seconds: float) -> float:
    """Return the runtime to store in the common solver output.

    MoMC can print ``Time 0.00000000`` for very fast instances because its
    internal timer rounds small values down to zero. A zero runtime is not
    useful for downstream comparisons, so the wrapper keeps MoMC's reported
    runtime when it is positive and falls back to a high-resolution wall-clock
    measurement otherwise.
    """
    if parsed.solver_runtime_seconds is not None and parsed.solver_runtime_seconds > 0:
        return float(parsed.solver_runtime_seconds)
    return max(float(elapsed_seconds), 1e-9)


def solve_momc_max_clique(
    graph: nx.Graph,
    momc_executable: str | Path = DEFAULT_MOMC_EXECUTABLE,
    dimacs_path: str | Path = "graph.clq",
    python_timeout_seconds: float | None = None,
    momc_output_node_base: int = 1,
) -> MOMCCliqueResult:
    """Solve maximum clique with MoMC for one NetworkX graph.

    ``best_clique_nodes`` are returned using the repository's internal 0-based
    node convention. The DIMACS input and MoMC stdout use 1-based node labels by
    default. If a local build reports 0-based ids, pass ``momc_output_node_base=0``.
    """
    G = _clean_graph(graph)

    if G.number_of_nodes() == 0:
        return MOMCCliqueResult(
            status="success",
            optimality_status="trivial_empty_graph",
            runtime_seconds=0.0,
            best_clique_size=0,
            best_clique_nodes=[],
            clique_valid=True,
            return_code=0,
        )

    write_dimacs_file(G, dimacs_path, comment=f"MoMC input for {Path(dimacs_path).stem}")

    return_code, stdout, stderr, elapsed, timeout_message = _run_momc_subprocess(
        momc_executable=momc_executable,
        dimacs_path=dimacs_path,
        python_timeout_seconds=python_timeout_seconds,
    )

    if timeout_message:
        parsed_timeout = parse_momc_stdout(stdout)
        return MOMCCliqueResult(
            status="error",
            optimality_status="python_timeout",
            runtime_seconds=elapsed,
            best_clique_size=parsed_timeout.max_clique_size,
            best_clique_nodes=[],
            clique_valid=False,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            error_message=timeout_message,
        )

    parsed = parse_momc_stdout(stdout)

    if return_code != 0:
        return MOMCCliqueResult(
            status="error",
            optimality_status="nonzero_exit",
            runtime_seconds=_runtime_from_solver_or_wall_clock(parsed, elapsed),
            best_clique_size=parsed.max_clique_size,
            best_clique_nodes=[],
            clique_valid=False,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            error_message=f"MoMC exited with return code {return_code}. stderr: {stderr.strip()}",
        )

    best_nodes = _convert_reported_nodes_to_internal(parsed.clique_nodes_reported, momc_output_node_base)
    best_size = parsed.max_clique_size

    if best_size is None:
        return MOMCCliqueResult(
            status="error",
            optimality_status="parse_error",
            runtime_seconds=elapsed,
            best_clique_size=None,
            best_clique_nodes=best_nodes,
            clique_valid=False,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            error_message="Could not parse Max_CLQ from MoMC stdout.",
        )

    clique_valid = len(best_nodes) == best_size and is_valid_clique(G, best_nodes)
    if not clique_valid:
        return MOMCCliqueResult(
            status="error",
            optimality_status="invalid_clique_output",
            runtime_seconds=_runtime_from_solver_or_wall_clock(parsed, elapsed),
            best_clique_size=best_size,
            best_clique_nodes=best_nodes,
            clique_valid=False,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            error_message=(
                "MoMC output did not parse into a valid clique under the repository's "
                "internal node convention. Check --momc-output-node-base if needed."
            ),
        )

    return MOMCCliqueResult(
        status="success",
        optimality_status="timeout_best_found" if parsed.timed_out else "reported_by_momc",
        runtime_seconds=_runtime_from_solver_or_wall_clock(parsed, elapsed),
        best_clique_size=best_size,
        best_clique_nodes=best_nodes,
        clique_valid=True,
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        error_message="",
    )


def write_raw_momc_output(output_dir: str | Path, graph_id: str, stdout: str, stderr: str) -> None:
    """Write per-graph MoMC stdout/stderr files for debugging and reproducibility."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{graph_id}.stdout.txt").write_text(stdout or "", encoding="utf-8")
    if stderr:
        (path / f"{graph_id}.stderr.txt").write_text(stderr, encoding="utf-8")


def append_notebook_style_momc_output(output_file: str | Path, graph_name: str, stdout: str, stderr: str = "") -> None:
    """Append one graph's MoMC raw output to a dataset-level text file.

    The original notebook concatenated stdout blocks into one text file. This
    function keeps that useful artifact while adding a graph marker and separator.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"Graph: {graph_name}\n")
        f.write(stdout or "")
        if stderr:
            f.write("\n[stderr]\n")
            f.write(stderr)
        f.write("\n" + "*" * 50 + "\n")
