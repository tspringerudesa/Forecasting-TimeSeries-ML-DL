"""Reproducible, non-generative methodology diagrams for the thesis.

The figures summarize architectural information flow. Mathematical detail is
kept in ``paper.tex`` so the diagrams remain legible at page width.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

try:
    from thesis.reporting import IMAGES_DIR
except ModuleNotFoundError:  # Support direct execution from ``thesis/``.
    from reporting import IMAGES_DIR


FIGURE_DIR = IMAGES_DIR / "methodology_figures"
PALETTE = {
    "blue": "#355C7D",
    "light_blue": "#D8E4EA",
    "green": "#52796F",
    "light_green": "#DCE8E2",
    "orange": "#C17C54",
    "light_orange": "#EEDFD2",
    "red": "#A44A5B",
    "purple": "#725D7E",
    "light_purple": "#E5DDE9",
    "gold": "#B99055",
    "grey": "#64757D",
    "light_grey": "#E9ECE8",
    "cream": "#F7F3EC",
    "ink": "#233238",
}


def _save(fig, filename: str, output_dir=FIGURE_DIR, dpi: int = 320) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    return path


def _clean_axis(axis, xlim=(0, 1), ylim=(0, 1)):
    axis.set(xlim=xlim, ylim=ylim)
    axis.axis("off")


def _box(
    axis,
    xy,
    width,
    height,
    text,
    *,
    facecolor="white",
    edgecolor=None,
    fontsize=11,
    linewidth=1.6,
    radius=0.025,
):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor or PALETTE["ink"],
        linewidth=linewidth,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=PALETTE["ink"],
    )
    return patch


def _arrow(
    axis,
    start,
    end,
    *,
    color=None,
    linestyle="-",
    linewidth=1.5,
    connectionstyle="arc3",
    mutation_scale=12,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color or PALETTE["ink"],
        connectionstyle=connectionstyle,
    )
    axis.add_patch(patch)
    return patch


def _node(
    axis,
    x,
    y,
    label,
    *,
    facecolor,
    edgecolor=None,
    size=1300,
    fontsize=11,
):
    """Use display-coordinate markers so neurons remain circular."""
    axis.scatter(
        [x],
        [y],
        s=size,
        marker="o",
        facecolor=facecolor,
        edgecolor=edgecolor or PALETTE["ink"],
        linewidth=1.6,
        zorder=4,
    )
    axis.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=5,
    )


def neural_network_backpropagation_figure(
    filename="architecture_neural_network_backpropagation.jpg",
):
    """Generic feed-forward connectivity and reverse training signal."""
    fig, axis = plt.subplots(figsize=(15, 7))
    _clean_axis(axis)
    layers = [
        (
            "Input features",
            0.12,
            [0.27, 0.50, 0.73],
            ["$x_1$", "$x_2$", "$x_3$"],
            PALETTE["light_blue"],
        ),
        (
            "Hidden layer",
            0.45,
            [0.20, 0.40, 0.60, 0.80],
            ["$h_1$", "$h_2$", "$h_3$", "$h_4$"],
            PALETTE["light_green"],
        ),
        (
            "Output layer",
            0.77,
            [0.39, 0.61],
            ["$\\hat y_1$", "$\\hat y_2$"],
            PALETTE["light_orange"],
        ),
    ]
    coordinates = []
    for title, x, ys, labels, fill in layers:
        axis.text(
            x,
            0.93,
            title,
            ha="center",
            fontsize=15,
            weight="bold",
            color=PALETTE["ink"],
        )
        layer_coordinates = []
        for y, label in zip(ys, labels):
            _node(axis, x, y, label, facecolor=fill, size=1600, fontsize=12)
            layer_coordinates.append((x, y))
        coordinates.append(layer_coordinates)

    for left, right in zip(coordinates[:-1], coordinates[1:]):
        for start in left:
            for end in right:
                _arrow(
                    axis,
                    (start[0] + 0.035, start[1]),
                    (end[0] - 0.035, end[1]),
                    color=PALETTE["blue"],
                    linewidth=0.85,
                    mutation_scale=7,
                )

    _box(
        axis,
        (0.87, 0.42),
        0.10,
        0.16,
        "Loss",
        facecolor="#EAD8DB",
        edgecolor=PALETTE["red"],
        fontsize=13,
        linewidth=2,
    )
    for output in coordinates[-1]:
        _arrow(
            axis,
            (output[0] + 0.04, output[1]),
            (0.87, 0.50),
            color=PALETTE["orange"],
            linewidth=1.6,
        )
    axis.text(
        0.45,
        0.875,
        "Forward pass",
        ha="center",
        fontsize=13,
        weight="bold",
        color=PALETTE["blue"],
    )
    _arrow(
        axis,
        (0.92, 0.40),
        (0.92, 0.11),
        color=PALETTE["red"],
        linestyle="--",
        linewidth=2.4,
    )
    _arrow(
        axis,
        (0.92, 0.11),
        (0.16, 0.11),
        color=PALETTE["red"],
        linestyle="--",
        linewidth=2.4,
    )
    axis.text(
        0.52,
        0.045,
        "Backpropagation",
        ha="center",
        fontsize=12,
        weight="bold",
        color=PALETTE["red"],
    )
    axis.set_title(
        "Generic feed-forward neural network",
        fontsize=20,
        weight="bold",
        color=PALETTE["ink"],
        pad=18,
    )
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.08, top=0.88)
    return fig, _save(fig, filename)


def recurrent_family_figure(filename="architecture_recurrent_family.jpg"):
    """Qualitative comparison of RNN, LSTM, and GRU memory flow."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.8))
    panels = [
        ("Vanilla RNN", PALETTE["blue"], PALETTE["light_blue"]),
        ("Long short-term memory", PALETTE["green"], PALETTE["light_green"]),
        ("Gated recurrent unit", PALETTE["purple"], PALETTE["light_purple"]),
    ]
    for axis, (title, accent, fill) in zip(axes, panels):
        _clean_axis(axis)
        axis.set_title(title, fontsize=20, weight="bold", color=accent, pad=13)
        _box(
            axis,
            (0.16, 0.25),
            0.68,
            0.48,
            "",
            facecolor=fill,
            edgecolor=accent,
            linewidth=2,
        )
        _arrow(axis, (0.02, 0.48), (0.15, 0.48), color=PALETTE["blue"], linewidth=1.8)
        _arrow(axis, (0.85, 0.48), (0.98, 0.48), color=accent, linewidth=1.8)
        _arrow(axis, (0.50, 0.90), (0.50, 0.74), color=PALETTE["gold"], linewidth=2)
        axis.text(0.02, 0.53, "$x_t$", fontsize=17, color=PALETTE["blue"])
        axis.text(0.91, 0.53, "$h_t$", fontsize=17, color=accent)
        axis.text(0.50, 0.93, "$h_{t-1}$", fontsize=17, ha="center", color=PALETTE["gold"])

    _node(
        axes[0],
        0.50,
        0.50,
        "state\nupdate",
        facecolor="white",
        edgecolor=PALETTE["blue"],
        size=3900,
        fontsize=13,
    )
    axes[0].text(
        0.50,
        0.13,
        "one recurrent state",
        ha="center",
        fontsize=14,
        color=PALETTE["grey"],
    )

    # Cell-state path: the forget gate scales the previous cell, the input
    # gate contributes new content, and the output gate exposes the new state.
    _arrow(axes[1], (0.18, 0.79), (0.27, 0.79), color=PALETTE["orange"], linewidth=2.6)
    _node(
        axes[1], 0.31, 0.79, r"$\times$", facecolor="white",
        edgecolor=PALETTE["red"], size=950, fontsize=17,
    )
    _arrow(axes[1], (0.35, 0.79), (0.56, 0.79), color=PALETTE["orange"], linewidth=2.6)
    _node(
        axes[1], 0.61, 0.79, r"$+$", facecolor="white",
        edgecolor=PALETTE["orange"], size=950, fontsize=17,
    )
    _arrow(axes[1], (0.66, 0.79), (0.83, 0.79), color=PALETTE["orange"], linewidth=2.6)
    axes[1].text(0.17, 0.84, "$c_{t-1}$", fontsize=16, color=PALETTE["orange"])
    axes[1].text(0.76, 0.84, "$c_t$", fontsize=16, color=PALETTE["orange"])

    for x, label, color in (
        (0.31, "forget\ngate", PALETTE["red"]),
        (0.51, "input\ngate", PALETTE["gold"]),
        (0.74, "output\ngate", PALETTE["green"]),
    ):
        _node(
            axes[1],
            x,
            0.48,
            label,
            facecolor="white",
            edgecolor=color,
            size=2700,
            fontsize=11.5,
        )
    _arrow(axes[1], (0.31, 0.57), (0.31, 0.70), color=PALETTE["red"], linewidth=1.5)
    _arrow(axes[1], (0.51, 0.57), (0.59, 0.71), color=PALETTE["gold"], linewidth=1.5)
    axes[1].text(
        0.48, 0.66, r"candidate $\widetilde{c}_t$", ha="right",
        fontsize=10.5, color=PALETTE["gold"],
    )
    _arrow(axes[1], (0.74, 0.74), (0.74, 0.58), color=PALETTE["orange"], linewidth=1.5)
    _arrow(axes[1], (0.80, 0.48), (0.84, 0.48), color=PALETTE["green"], linewidth=1.5)
    axes[1].text(
        0.50,
        0.13,
        "separate cell-state highway",
        ha="center",
        fontsize=14,
        color=PALETTE["grey"],
    )

    for x, label, color in (
        (0.38, "reset", PALETTE["red"]),
        (0.62, "update", PALETTE["purple"]),
    ):
        _node(
            axes[2],
            x,
            0.50,
            "gate",
            facecolor="white",
            edgecolor=color,
            size=2600,
            fontsize=12,
        )
        axes[2].text(x, 0.32, label, ha="center", fontsize=13, color=color)
    axes[2].text(
        0.50,
        0.13,
        "gated state without a separate cell",
        ha="center",
        fontsize=14,
        color=PALETTE["grey"],
    )

    fig.suptitle(
        "Recurrent architectures: information retained at time $t$",
        fontsize=23,
        weight="bold",
        color=PALETTE["ink"],
        y=0.985,
    )
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.08, top=0.84, wspace=0.10)
    return fig, _save(fig, filename)


def kan_vs_mlp_figure(filename="architecture_kan_vs_mlp.jpg"):
    """Conceptual distinction between MLP node activations and KAN edges."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
    layer_x = [0.12, 0.50, 0.88]
    layer_y = [[0.38, 0.68], [0.30, 0.53, 0.76], [0.53]]
    for axis in axes:
        _clean_axis(axis)

    axes[0].set_title("Multilayer perceptron", fontsize=21, weight="bold", color=PALETTE["blue"])
    axes[1].set_title("Kolmogorov-Arnold network", fontsize=21, weight="bold", color=PALETTE["purple"])

    for panel in (0, 1):
        for layer, (x, ys) in enumerate(zip(layer_x, layer_y)):
            for y in ys:
                if panel == 0:
                    label = "$\\sigma$" if layer == 1 else "$x$" if layer == 0 else "$y$"
                    fill = PALETTE["light_green"] if layer == 1 else PALETTE["light_blue"]
                else:
                    label = "$\\sum$" if layer == 1 else "$x$" if layer == 0 else "$y$"
                    fill = PALETTE["light_orange"] if layer == 1 else PALETTE["cream"]
                _node(axes[panel], x, y, label, facecolor=fill, size=2300, fontsize=15)

    for left in range(2):
        for start in layer_y[left]:
            for end in layer_y[left + 1]:
                _arrow(
                    axes[0],
                    (layer_x[left] + 0.04, start),
                    (layer_x[left + 1] - 0.04, end),
                    color=PALETTE["grey"],
                    linewidth=1,
                    mutation_scale=8,
                )

    n_kan_edges = sum(
        len(layer_y[left]) * len(layer_y[left + 1])
        for left in range(2)
    )
    # Every KAN edge represents its own learned univariate function. Give each
    # edge a distinct categorical color so repetition cannot suggest parameter
    # sharing, equal functions, or a graph-theoretic relationship.
    edge_colors = plt.get_cmap("viridis")(
        np.linspace(0.08, 0.92, n_kan_edges)
    )
    edge_number = 0
    for left in range(2):
        for start in layer_y[left]:
            for end in layer_y[left + 1]:
                color = tuple(edge_colors[edge_number])
                edge_number += 1
                _arrow(
                    axes[1],
                    (layer_x[left] + 0.04, start),
                    (layer_x[left + 1] - 0.04, end),
                    color=color,
                    linewidth=1.8,
                    mutation_scale=8,
                )

    axes[0].text(
        0.50,
        0.13,
        "learned scalar connections\nfixed nonlinear activation at hidden nodes",
        ha="center",
        fontsize=15,
        color=PALETTE["blue"],
    )
    axes[1].text(
        0.50,
        0.13,
        "learned univariate functions on connections\nhidden nodes add incoming values",
        ha="center",
        fontsize=15,
        color=PALETTE["purple"],
    )
    fig.suptitle(
        "Where nonlinearity lives: MLP versus KAN",
        fontsize=23,
        weight="bold",
        color=PALETTE["ink"],
        y=0.985,
    )
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.07, top=0.85, wspace=0.08)
    return fig, _save(fig, filename)


def convolution_figure(filename="architecture_cnn_time_series.jpg"):
    """Compact temporal-convolution and weight-sharing schematic."""
    fig, axis = plt.subplots(figsize=(13, 6.2))
    _clean_axis(axis, xlim=(0, 12.5), ylim=(0, 6))

    for index in range(9):
        x = 0.55 + index * 0.78
        highlighted = 2 <= index <= 4
        axis.add_patch(Rectangle(
            (x, 3.35),
            0.65,
            0.65,
            facecolor=PALETTE["light_orange"] if highlighted else PALETTE["light_blue"],
            edgecolor=PALETTE["ink"],
            linewidth=1.2,
        ))
        lag = 8 - index
        label = "$x_t$" if lag == 0 else f"$x_{{t-{lag}}}$"
        axis.text(x + 0.325, 3.675, label, ha="center", va="center", fontsize=12)

    axis.add_patch(Rectangle(
        (2.00, 3.17),
        2.36,
        1.01,
        fill=False,
        edgecolor=PALETTE["orange"],
        linewidth=2.6,
    ))
    axis.text(
        3.18,
        4.42,
        "one local window",
        ha="center",
        fontsize=15,
        weight="bold",
        color=PALETTE["orange"],
    )
    _box(
        axis,
        (3.85, 1.55),
        1.65,
        0.82,
        "Shared\nkernel",
        facecolor=PALETTE["light_orange"],
        edgecolor=PALETTE["orange"],
        fontsize=15,
    )
    _arrow(axis, (3.18, 3.13), (4.55, 2.39), color=PALETTE["orange"], linewidth=2)

    for index in range(7):
        x = 6.35 + index * 0.78
        axis.add_patch(Rectangle(
            (x, 1.61),
            0.76,
            0.76,
            facecolor=PALETTE["light_green"],
            edgecolor=PALETTE["green"],
            linewidth=1.2,
        ))
        axis.text(x + 0.38, 1.99, f"$z_{index+1}$", ha="center", va="center", fontsize=14)
    _arrow(axis, (5.55, 1.97), (6.45, 1.97), color=PALETTE["green"], linewidth=2)
    axis.text(
        9.07,
        1.12,
        "temporal feature map",
        ha="center",
        fontsize=15,
        weight="bold",
        color=PALETTE["green"],
    )

    axis.text(
        3.85,
        5.05,
        "The same kernel slides across ordered observations",
        ha="center",
        fontsize=16,
        weight="bold",
        color=PALETTE["blue"],
    )
    axis.set_title(
        "One-dimensional convolution for time-series feature extraction",
        fontsize=22,
        weight="bold",
        color=PALETTE["ink"],
        pad=18,
    )
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.08, top=0.87)
    return fig, _save(fig, filename)


def attention_figure(filename="architecture_scaled_dot_product_attention.jpg"):
    """Attention flow and a compact interpretation of its weight matrix."""
    fig = plt.figure(figsize=(13, 7.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], wspace=0.22)

    axis = fig.add_subplot(grid[0, 0])
    _clean_axis(axis)
    _box(axis, (0.03, 0.67), 0.18, 0.12, "Queries\n$Q$", facecolor=PALETTE["light_blue"], fontsize=14)
    _box(axis, (0.03, 0.43), 0.18, 0.12, "Keys\n$K$", facecolor=PALETTE["light_green"], fontsize=14)
    _box(axis, (0.03, 0.19), 0.18, 0.12, "Values\n$V$", facecolor=PALETTE["light_orange"], fontsize=14)
    _box(axis, (0.31, 0.51), 0.20, 0.15, "Similarity\nscores", facecolor=PALETTE["light_purple"], fontsize=14)
    _box(axis, (0.61, 0.51), 0.18, 0.15, "Row-wise\nsoftmax", facecolor="#EAD8DB", fontsize=14)
    _box(axis, (0.59, 0.20), 0.22, 0.15, "Weighted\nvalue sum", facecolor=PALETTE["light_green"], fontsize=14)
    _box(axis, (0.83, 0.20), 0.14, 0.15, "Context", facecolor=PALETTE["light_blue"], fontsize=13)
    _arrow(axis, (0.21, 0.73), (0.31, 0.62), color=PALETTE["blue"], linewidth=1.8)
    _arrow(axis, (0.21, 0.49), (0.31, 0.56), color=PALETTE["green"], linewidth=1.8)
    _arrow(axis, (0.51, 0.59), (0.61, 0.59), color=PALETTE["purple"], linewidth=1.8)
    _arrow(axis, (0.70, 0.50), (0.70, 0.36), color=PALETTE["red"], linewidth=1.8)
    _arrow(axis, (0.21, 0.25), (0.59, 0.27), color=PALETTE["orange"], linewidth=1.8)
    _arrow(axis, (0.81, 0.27), (0.83, 0.27), color=PALETTE["green"], linewidth=1.8)
    axis.text(
        0.50,
        0.07,
        "scores to attention weights to contextual representation",
        ha="center",
        fontsize=14,
        weight="bold",
        color=PALETTE["purple"],
    )
    axis.set_title("Masked scaled dot-product attention", fontsize=19, weight="bold", color=PALETTE["purple"])

    heat_axis = fig.add_subplot(grid[0, 1])
    labels = ["$t-4$", "$t-3$", "$t-2$", "$t-1$", "$t$"]
    weights = np.array([
        [1.00, 0.00, 0.00, 0.00, 0.00],
        [0.30, 0.70, 0.00, 0.00, 0.00],
        [0.15, 0.25, 0.60, 0.00, 0.00],
        [0.08, 0.12, 0.20, 0.60, 0.00],
        [0.08, 0.32, 0.15, 0.18, 0.27],
    ])
    cmap = LinearSegmentedColormap.from_list(
        "attention",
        [PALETTE["cream"], "#C9D8D2", PALETTE["green"], "#274C47"],
    )
    image = heat_axis.imshow(weights, cmap=cmap, vmin=0, vmax=1.00)
    for row in range(5):
        for column in range(5):
            if column > row:
                heat_axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor=PALETTE["light_grey"],
                        edgecolor="#B9C0C2",
                        hatch="//",
                        linewidth=0.6,
                    )
                )
                continue
            heat_axis.text(
                column,
                row,
                f"{weights[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=12.5,
                color="white" if weights[row, column] >= 0.50 else PALETTE["ink"],
            )
    heat_axis.set(
        xlabel="Available key/value position",
        ylabel="Query position",
        title="Causally masked attention weights",
        xticks=np.arange(5),
        yticks=np.arange(5),
    )
    heat_axis.set_xticklabels(labels, fontsize=13)
    heat_axis.set_yticklabels(labels, fontsize=13)
    heat_axis.xaxis.label.set_size(13)
    heat_axis.yaxis.label.set_size(13)
    heat_axis.tick_params(length=0)
    heat_axis.title.set_fontsize(19)
    heat_axis.title.set_weight("bold")
    heat_axis.title.set_color(PALETTE["purple"])
    fig.colorbar(image, ax=heat_axis, fraction=0.046, pad=0.04, label="attention weight")

    fig.suptitle(
        "Attention converts pairwise relevance into weighted context",
        fontsize=23,
        weight="bold",
        color=PALETTE["ink"],
        y=0.985,
    )
    fig.subplots_adjust(left=0.035, right=0.97, bottom=0.18, top=0.84)
    return fig, _save(fig, filename)


def activation_functions_figure(filename="activation_functions_summary.jpg"):
    """Hidden and output activations discussed in the thesis."""
    x = np.linspace(-5, 5, 500)
    curves = [
        ("Sigmoid", 1 / (1 + np.exp(-x)), PALETTE["blue"]),
        ("Hyperbolic tangent", np.tanh(x), PALETTE["purple"]),
        ("Rectified linear unit", np.maximum(0, x), PALETTE["green"]),
        ("Leaky ReLU", np.where(x >= 0, x, 0.1 * x), PALETTE["orange"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for axis, (title, y, color) in zip(list(axes.flat)[:4], curves):
        axis.axhline(0, color=PALETTE["light_grey"], linewidth=1)
        axis.axvline(0, color=PALETTE["light_grey"], linewidth=1)
        axis.plot(x, y, color=color, linewidth=2.8)
        axis.set_title(title, fontsize=14, weight="bold", color=color)
        axis.set_xlim(-5, 5)
        axis.grid(alpha=0.13)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("$x$")
        axis.set_ylabel("activation")

    linear_axis = axes[1, 1]
    linear_axis.axhline(0, color=PALETTE["light_grey"], linewidth=1)
    linear_axis.axvline(0, color=PALETTE["light_grey"], linewidth=1)
    linear_axis.plot(x, x, color=PALETTE["blue"], linewidth=2.8)
    linear_axis.set(xlim=(-5, 5), ylim=(-5, 5), xlabel="$x$", ylabel="output")
    linear_axis.grid(alpha=0.13)
    linear_axis.spines[["top", "right"]].set_visible(False)
    linear_axis.set_title("Linear regression output", fontsize=14, weight="bold", color=PALETTE["blue"])

    softmax_axis = axes[1, 2]
    logits = np.array([0.5, 1.8, -0.2])
    probabilities = np.exp(logits) / np.exp(logits).sum()
    bars = softmax_axis.bar(
        ["class 1", "class 2", "class 3"],
        probabilities,
        color=[PALETTE["light_blue"], PALETTE["purple"], PALETTE["light_orange"]],
        edgecolor=PALETTE["ink"],
        linewidth=0.8,
    )
    softmax_axis.set_ylim(0, 1)
    softmax_axis.set_ylabel("probability")
    softmax_axis.grid(axis="y", alpha=0.13)
    softmax_axis.spines[["top", "right"]].set_visible(False)
    softmax_axis.set_title("Softmax multiclass output", fontsize=14, weight="bold", color=PALETTE["purple"])
    for bar, probability in zip(bars, probabilities):
        softmax_axis.text(
            bar.get_x() + bar.get_width() / 2,
            probability + 0.035,
            f"{probability:.2f}",
            ha="center",
            fontsize=10.5,
        )

    fig.text(
        0.50,
        0.93,
        "Hidden-layer nonlinearities",
        ha="center",
        fontsize=13,
        color=PALETTE["grey"],
        weight="bold",
    )
    fig.suptitle(
        "Activation functions used in hidden and output layers",
        fontsize=20,
        weight="bold",
        color=PALETTE["ink"],
        y=0.99,
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.07, top=0.87, hspace=0.32, wspace=0.22)
    return fig, _save(fig, filename)


def validation_strategies_figure(filename="validation_strategies_summary.jpg"):
    """Four gap-free validation strategies for ordered observations."""
    strategies = [
        ("Basic holdout", [[(0, 8, "train"), (8, 12, "validation")]]),
        ("Expanding Rep-Holdout-O", [
            [(0, 4, "train"), (4, 6, "validation")],
            [(0, 7, "train"), (7, 9, "validation")],
            [(0, 10, "train"), (10, 12, "validation")],
        ]),
        ("Rolling Rep-Holdout-O", [
            [(0, 4, "train"), (4, 6, "validation")],
            [(3, 7, "train"), (7, 9, "validation")],
            [(6, 10, "train"), (10, 12, "validation")],
        ]),
        ("Blocked cross-validation", [
            [(0, 3, "validation"), (3, 12, "train")],
            [(0, 3, "train"), (3, 6, "validation"), (6, 12, "train")],
            [(0, 6, "train"), (6, 9, "validation"), (9, 12, "train")],
        ]),
    ]
    colors = {"train": "#486F68", "validation": "#C07A56"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True)
    for axis, (name, folds) in zip(axes.flat, strategies):
        axis.set_facecolor("#FAF7F1")
        axis.set_xlim(0, 12)
        axis.set_ylim(-0.25, 3.35)
        axis.set_xticks(np.arange(0, 13, 2))
        axis.set_yticks([0.45, 1.45, 2.45])
        axis.grid(axis="x", color="#D9D2C7", linewidth=0.8, alpha=0.65)
        axis.set_title(name, loc="left", fontsize=14, weight="bold", color=PALETTE["ink"], pad=10)
        if len(folds) == 1:
            row_positions = [1.45]
            axis.set_yticklabels(["", "single split", ""])
        else:
            row_positions = [2.45, 1.45, 0.45]
            axis.set_yticklabels(["origin 3", "origin 2", "origin 1"])
        for row_y, segments in zip(row_positions, folds):
            for start, stop, kind in segments:
                axis.add_patch(Rectangle(
                    (start, row_y - 0.31),
                    stop - start,
                    0.62,
                    facecolor=colors[kind],
                    edgecolor="#FAF7F1",
                    linewidth=1.4,
                ))
                if stop - start >= 2:
                    axis.text(
                        (start + stop) / 2,
                        row_y,
                        "fit" if kind == "train" else "score",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white",
                        weight="bold",
                    )
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#AFA69A")
        axis.tick_params(axis="y", length=0, labelsize=9, colors=PALETTE["grey"])
        axis.tick_params(axis="x", length=0, labelsize=9, colors=PALETTE["grey"])
    for axis in axes[1, :]:
        axis.set_xlabel("ordered time index", fontsize=11)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=colors["train"], label="Training"),
        Rectangle((0, 0), 1, 1, facecolor=colors["validation"], label="Validation"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.015), frameon=False, fontsize=11)
    fig.suptitle(
        "Validation strategies for ordered time-series observations",
        fontsize=20,
        weight="bold",
        color=PALETTE["ink"],
        y=0.985,
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.90, hspace=0.32, wspace=0.16)
    return fig, _save(fig, filename)


def generate_all_methodology_figures():
    """Generate every architecture and validation figure."""
    generators = [
        neural_network_backpropagation_figure,
        recurrent_family_figure,
        kan_vs_mlp_figure,
        convolution_figure,
        attention_figure,
        activation_functions_figure,
        validation_strategies_figure,
    ]
    paths = []
    for generator in generators:
        fig, path = generator()
        paths.append(path)
        plt.close(fig)
    return paths


if __name__ == "__main__":
    for generated_path in generate_all_methodology_figures():
        print(generated_path)
