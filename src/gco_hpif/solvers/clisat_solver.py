"""CliSAT wrapper for the maximum clique problem.

This module follows the same solver-output contract as the Gurobi wrapper:
solver-specific code returns a result dataclass, while the CLI writes the
common ``solver_runs.csv``, ``solver_errors.csv``, and ``run_summary.csv`` files.

The repository patch includes a Linux x86-64 CliSAT binary at
``external/CliSAT/bin/CliSAT``. The wrapper invokes it with the same argument
pattern used in the original notebook::

    CliSAT <graph_path.clq> <time_limit_seconds> <threads>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import stat
import subprocess
import time
from typing import Sequence

import networkx as nx

from gco_hpif.solvers.common import is_valid_clique


DEFAULT_CLISAT_EXECUTABLE = Path("external/CliSAT/bin/CliSAT")
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


@dataclass(frozen=True)
class ParsedCliSATOutput:
    """Parsed fields reported by CliSAT for one graph instance."""

    omega: int | None
    clique_nodes_reported: list[int]
    solver_runtime_seconds: float | None


@dataclass(frozen=True)
class CliSATCliqueResult:
    """Result returned by the CliSAT maximum-clique wrapper."""

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
    """Convert a NetworkX graph to DIMACS ``.clq`` text for CliSAT.

    GCO-HPIF stores graphs internally with 0-based node labels. DIMACS edge
    files are written with 1-based node ids, so node ``0`` becomes ``1`` in the
    generated file. The bundled CliSAT binary reports clique nodes using
    0-based ids in stdout; this is handled separately by ``solve_clisat_max_clique``.
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
    """Remove terminal color/control codes from CliSAT output before parsing."""
    return ANSI_ESCAPE_RE.sub("", text or "")


def _extract_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _extract_clique_nodes(clean_stdout: str) -> list[int]:
    """Extract the clique-node line from CliSAT stdout.

    The uploaded CliSAT binary prints lines like ``1 2  [2]``. Some related
    builds print a leading star, e.g. ``* 1 2 [2]`` or ``* 1 2 [``. This parser
    accepts both formats.
    """
    for line in clean_stdout.splitlines():
        stripped = line.strip()
        match = re.fullmatch(r"(?:\*\s*)?((?:\d+\s*)+)\[(?:\d+)?\]?", stripped)
        if match:
            return [int(value) for value in match.group(1).split()]
    return []


def parse_clisat_stdout(stdout: str) -> ParsedCliSATOutput:
    """Parse CliSAT stdout for clique size, clique nodes, and solver timing.

    The parser is intentionally tolerant because CliSAT builds may differ in
    surrounding log text. It looks for the fields used in the original notebook:

    - ``omega:<integer>``
    - a clique-node line ending in ``[omega]`` or ``[``
    - ``ts(s):...``, ``tp(s):...``, and ``tr(s):...``
    """
    clean_stdout = _strip_ansi(stdout)

    omega_match = re.search(r"omega:\s*(\d+)", clean_stdout)
    omega = int(omega_match.group(1)) if omega_match else None

    clique_nodes_reported = _extract_clique_nodes(clean_stdout)

    timing_parts = [
        _extract_float(r"ts\(s\):\s*([0-9.eE+-]+)", clean_stdout),
        _extract_float(r"tp\(s\):\s*([0-9.eE+-]+)", clean_stdout),
        _extract_float(r"tr\(s\):\s*([0-9.eE+-]+)", clean_stdout),
    ]
    solver_runtime_seconds = None
    if any(value is not None for value in timing_parts):
        solver_runtime_seconds = sum(value or 0.0 for value in timing_parts)

    return ParsedCliSATOutput(
        omega=omega,
        clique_nodes_reported=clique_nodes_reported,
        solver_runtime_seconds=solver_runtime_seconds,
    )


def _ensure_executable(path: Path) -> None:
    """Make the bundled binary executable if file permissions were lost on unzip."""
    if os.name == "nt":
        return
    mode = path.stat().st_mode
    if not (mode & stat.S_IXUSR):
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_clisat_subprocess(
    clisat_executable: str | Path,
    dimacs_path: str | Path,
    time_limit_seconds: float,
    threads: int = 1,
    python_timeout_seconds: float | None = None,
) -> tuple[int | None, str, str, float, str]:
    """Run the external CliSAT executable and return subprocess details.

    The command is equivalent to the original notebook command:

    ``CliSAT <graph_path> <time_limit_seconds> <threads>``
    """
    executable = Path(clisat_executable)
    graph_file = Path(dimacs_path)

    if not executable.exists():
        raise FileNotFoundError(f"CliSAT executable not found: {executable}")
    if not graph_file.exists():
        raise FileNotFoundError(f"DIMACS graph file not found: {graph_file}")

    _ensure_executable(executable)

    timeout = python_timeout_seconds
    if timeout is None:
        timeout = float(time_limit_seconds) + 60.0

    command = [str(executable), str(graph_file), str(int(time_limit_seconds)), str(int(threads))]

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        return completed.returncode, completed.stdout, completed.stderr, elapsed, ""
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return None, stdout, stderr, elapsed, f"Python subprocess timeout after {timeout:.3f} seconds"


def _convert_reported_nodes_to_internal(nodes: Sequence[int], output_node_base: int) -> list[int]:
    if output_node_base not in {0, 1}:
        raise ValueError("output_node_base must be either 0 or 1.")
    return [int(node_id) - output_node_base for node_id in nodes]


def solve_clisat_max_clique(
    graph: nx.Graph,
    clisat_executable: str | Path = DEFAULT_CLISAT_EXECUTABLE,
    dimacs_path: str | Path = "graph.clq",
    time_limit_seconds: float = 1800,
    threads: int = 1,
    python_timeout_seconds: float | None = None,
    clisat_output_node_base: int = 0,
) -> CliSATCliqueResult:
    """Solve maximum clique with CliSAT for one NetworkX graph.

    ``best_clique_nodes`` are returned using the repository's internal 0-based
    node convention. The DIMACS input file uses 1-based edge endpoints, but the
    uploaded CliSAT binary reports clique-node ids using 0-based labels. If a
    different CliSAT build reports 1-based node ids, pass
    ``clisat_output_node_base=1``.
    """
    G = _clean_graph(graph)

    if G.number_of_nodes() == 0:
        return CliSATCliqueResult(
            status="success",
            optimality_status="trivial_empty_graph",
            runtime_seconds=0.0,
            best_clique_size=0,
            best_clique_nodes=[],
            clique_valid=True,
            return_code=0,
        )

    write_dimacs_file(G, dimacs_path, comment=f"CliSAT input for {Path(dimacs_path).stem}")

    return_code, stdout, stderr, elapsed, timeout_message = _run_clisat_subprocess(
        clisat_executable=clisat_executable,
        dimacs_path=dimacs_path,
        time_limit_seconds=time_limit_seconds,
        threads=threads,
        python_timeout_seconds=python_timeout_seconds,
    )

    if timeout_message:
        return CliSATCliqueResult(
            status="timeout",
            optimality_status="python_timeout",
            runtime_seconds=elapsed,
            best_clique_size=None,
            best_clique_nodes=[],
            clique_valid=False,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            error_message=timeout_message,
        )

    if return_code != 0:
        return CliSATCliqueResult(
            status="error",
            optimality_status="nonzero_exit",
            runtime_seconds=elapsed,
            best_clique_size=None,
            best_clique_nodes=[],
            clique_valid=False,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            error_message=f"CliSAT exited with return code {return_code}.",
        )

    parsed = parse_clisat_stdout(stdout)
    runtime_seconds = parsed.solver_runtime_seconds if parsed.solver_runtime_seconds is not None else elapsed

    if parsed.omega is None:
        return CliSATCliqueResult(
            status="error",
            optimality_status="parse_error",
            runtime_seconds=runtime_seconds,
            best_clique_size=None,
            best_clique_nodes=[],
            clique_valid=False,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            error_message="CliSAT stdout did not contain an omega:<integer> field.",
        )

    best_clique_nodes = _convert_reported_nodes_to_internal(
        parsed.clique_nodes_reported,
        output_node_base=clisat_output_node_base,
    )
    clique_valid = is_valid_clique(G, best_clique_nodes) if best_clique_nodes else parsed.omega == 0

    warnings: list[str] = []
    if best_clique_nodes and len(best_clique_nodes) != parsed.omega:
        warnings.append(f"Parsed {len(best_clique_nodes)} clique nodes but omega={parsed.omega}.")
    if not best_clique_nodes and parsed.omega > 0:
        warnings.append("CliSAT reported omega but no clique node list was parsed.")
    if best_clique_nodes and not clique_valid:
        warnings.append(
            "Parsed CliSAT node list is not a valid clique. "
            "Check --clisat-output-node-base if using a different CliSAT build."
        )

    return CliSATCliqueResult(
        status="success",
        optimality_status="reported_by_clisat",
        runtime_seconds=runtime_seconds,
        best_clique_size=parsed.omega,
        best_clique_nodes=best_clique_nodes,
        clique_valid=bool(clique_valid),
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        error_message=" ".join(warnings),
    )


def write_raw_clisat_output(
    output_dir: str | Path,
    graph_id: str,
    stdout: str,
    stderr: str,
) -> tuple[Path, Path]:
    """Write per-graph raw stdout/stderr for debugging and reproducibility."""
    raw_dir = Path(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_graph_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", graph_id)
    stdout_path = raw_dir / f"{safe_graph_id}.stdout.txt"
    stderr_path = raw_dir / f"{safe_graph_id}.stderr.txt"
    stdout_path.write_text(stdout or "", encoding="utf-8")
    stderr_path.write_text(stderr or "", encoding="utf-8")
    return stdout_path, stderr_path


def append_notebook_style_clisat_output(
    output_file: str | Path,
    graph_name: str,
    stdout: str,
    stderr: str = "",
) -> Path:
    """Append raw output in the same block style as the original notebook.

    This file is only a reproducibility/debug artifact. The canonical results
    remain the common ``solver_runs.csv``, ``solver_errors.csv``, and
    ``run_summary.csv`` files.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"Graph: {graph_name}\n")
        f.write(stdout or "")
        if stdout and not stdout.endswith("\n"):
            f.write("\n")
        if stderr:
            f.write("\n--- STDERR ---\n")
            f.write(stderr)
            if not stderr.endswith("\n"):
                f.write("\n")
        f.write("\n" + "*" * 50 + "\n")
    return path
