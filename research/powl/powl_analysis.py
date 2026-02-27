"""Analysis utilities for POWL models.

Provides structural inspection of discovered POWL models, including node
counts, operator type distribution, and hierarchical depth analysis.

POWL operator types (pm4py):
    - StrictPartialOrder: Activities with partial ordering constraints
    - OperatorPOWL (LOOP): Loop construct
    - OperatorPOWL (XOR):  Exclusive choice
    - Transition:          Leaf activity node
    - SilentTransition:    Invisible/tau transition
"""

import sys
import os
from collections import Counter

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from pm4py.objects.powl.obj import (
        StrictPartialOrder,
        OperatorPOWL,
        Transition,
        SilentTransition,
        POWL,
    )
    _POWL_AVAILABLE = True
except ImportError:
    _POWL_AVAILABLE = False


def _walk_powl(node, depth=0):
    """Recursively walk a POWL model tree, yielding (node, depth) tuples."""
    if node is None:
        return
    yield node, depth

    if isinstance(node, StrictPartialOrder):
        for child in node.children:
            yield from _walk_powl(child, depth + 1)
    elif isinstance(node, OperatorPOWL):
        for child in node.children:
            yield from _walk_powl(child, depth + 1)


def get_powl_statistics(powl_model) -> dict:
    """Compute structural statistics for a POWL model.

    Parameters
    ----------
    powl_model : pm4py POWL model
        A discovered POWL model (root node of the POWL tree).

    Returns
    -------
    dict
        Statistics dictionary with keys:
        - ``node_count``: Total number of nodes in the POWL tree.
        - ``leaf_count``: Number of activity (Transition) nodes.
        - ``silent_count``: Number of silent/tau transitions.
        - ``operator_count``: Number of operator nodes (SPO, LOOP, XOR).
        - ``operator_types``: Counter of operator type names.
        - ``max_depth``: Maximum nesting depth of the model.
        - ``unique_activities``: Set of activity labels at leaf nodes.
        - ``partial_order_count``: Number of StrictPartialOrder nodes.
        - ``loop_count``: Number of LOOP operators.
        - ``xor_count``: Number of XOR (choice) operators.
    """
    if powl_model is None:
        return {
            "node_count": 0,
            "leaf_count": 0,
            "silent_count": 0,
            "operator_count": 0,
            "operator_types": Counter(),
            "max_depth": 0,
            "unique_activities": set(),
            "partial_order_count": 0,
            "loop_count": 0,
            "xor_count": 0,
        }

    if not _POWL_AVAILABLE:
        return {"error": "pm4py POWL classes not available; upgrade pm4py >= 2.7.11"}

    nodes = list(_walk_powl(powl_model))

    leaf_count = 0
    silent_count = 0
    operator_count = 0
    operator_types = Counter()
    unique_activities = set()
    max_depth = 0
    partial_order_count = 0
    loop_count = 0
    xor_count = 0

    for node, depth in nodes:
        max_depth = max(max_depth, depth)

        if isinstance(node, SilentTransition):
            silent_count += 1
        elif isinstance(node, Transition):
            leaf_count += 1
            label = getattr(node, "label", None)
            if label:
                unique_activities.add(label)
        elif isinstance(node, StrictPartialOrder):
            operator_count += 1
            operator_types["StrictPartialOrder"] += 1
            partial_order_count += 1
        elif isinstance(node, OperatorPOWL):
            operator_count += 1
            op = getattr(node, "operator", None)
            op_name = str(op) if op else "Unknown"
            # pm4py uses Operator.LOOP, Operator.XOR, etc.
            operator_types[op_name] += 1
            op_name_upper = op_name.upper()
            if "LOOP" in op_name_upper:
                loop_count += 1
            elif "XOR" in op_name_upper:
                xor_count += 1

    return {
        "node_count": len(nodes),
        "leaf_count": leaf_count,
        "silent_count": silent_count,
        "operator_count": operator_count,
        "operator_types": operator_types,
        "max_depth": max_depth,
        "unique_activities": unique_activities,
        "partial_order_count": partial_order_count,
        "loop_count": loop_count,
        "xor_count": xor_count,
    }


def get_partial_order_edges(powl_model) -> list:
    """Extract partial ordering edges from all StrictPartialOrder nodes.

    Returns a list of dicts, each with ``parent_depth``, ``from_node``,
    ``to_node`` describing an ordering constraint between children.

    Parameters
    ----------
    powl_model : pm4py POWL model

    Returns
    -------
    list[dict]
        List of ordering edge descriptors.
    """
    if powl_model is None or not _POWL_AVAILABLE:
        return []

    edges = []
    for node, depth in _walk_powl(powl_model):
        if isinstance(node, StrictPartialOrder):
            order = getattr(node, "order", None)
            if order is None:
                continue
            children = node.children
            # order is an adjacency matrix (order.get_matrix() or order itself)
            # In pm4py, StrictPartialOrder.order is a Matrix object
            try:
                matrix = order.get_matrix() if hasattr(order, "get_matrix") else order
            except Exception:
                continue

            for i in range(len(children)):
                for j in range(len(children)):
                    if i != j:
                        try:
                            if matrix[i][j]:
                                from_label = _node_label(children[i])
                                to_label = _node_label(children[j])
                                edges.append({
                                    "parent_depth": depth,
                                    "from_node": from_label,
                                    "to_node": to_label,
                                })
                        except (IndexError, TypeError):
                            continue
    return edges


def _node_label(node) -> str:
    """Get a human-readable label for a POWL node."""
    if not _POWL_AVAILABLE:
        return str(node)

    if isinstance(node, SilentTransition):
        return "(tau)"
    elif isinstance(node, Transition):
        return getattr(node, "label", None) or "(unlabeled)"
    elif isinstance(node, StrictPartialOrder):
        child_labels = [_node_label(c) for c in node.children]
        return f"SPO[{', '.join(child_labels)}]"
    elif isinstance(node, OperatorPOWL):
        op = getattr(node, "operator", "?")
        child_labels = [_node_label(c) for c in node.children]
        return f"{op}({', '.join(child_labels)})"
    return str(node)


def summarize_powl_model(powl_model) -> str:
    """Generate a human-readable summary of a POWL model.

    Parameters
    ----------
    powl_model : pm4py POWL model

    Returns
    -------
    str
        Multi-line summary suitable for logging or display.
    """
    stats = get_powl_statistics(powl_model)
    if "error" in stats:
        return f"POWL analysis unavailable: {stats['error']}"

    if stats["node_count"] == 0:
        return "Empty POWL model (no nodes)."

    lines = [
        "POWL Model Summary",
        "=" * 40,
        f"Total nodes:          {stats['node_count']}",
        f"Activity nodes:       {stats['leaf_count']}",
        f"Silent transitions:   {stats['silent_count']}",
        f"Operator nodes:       {stats['operator_count']}",
        f"Max nesting depth:    {stats['max_depth']}",
        f"Partial orders (SPO): {stats['partial_order_count']}",
        f"Loop operators:       {stats['loop_count']}",
        f"XOR operators:        {stats['xor_count']}",
        "",
        f"Unique activities ({len(stats['unique_activities'])}):",
    ]
    for act in sorted(stats["unique_activities"]):
        lines.append(f"  - {act}")

    if stats["operator_types"]:
        lines.append("")
        lines.append("Operator distribution:")
        for op_type, count in stats["operator_types"].most_common():
            lines.append(f"  {op_type}: {count}")

    edges = get_partial_order_edges(powl_model)
    if edges:
        lines.append("")
        lines.append(f"Partial ordering edges ({len(edges)}):")
        for edge in edges[:20]:  # Limit output for readability
            lines.append(f"  {edge['from_node']} -> {edge['to_node']}")
        if len(edges) > 20:
            lines.append(f"  ... and {len(edges) - 20} more")

    return "\n".join(lines)
