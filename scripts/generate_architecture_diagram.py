"""Generates reports/figures/architecture_diagram.png for the written report."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "reports" / "figures" / "architecture_diagram.png"


def box(ax, xy, w, h, text, color="#2a6f97"):
    rect = mpatches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5,
        edgecolor=color,
        facecolor=color,
        alpha=0.15,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color="#0b2545",
        wrap=True,
        fontweight="bold",
    )


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5),
    )


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Row 1: data -> training pipeline
    box(ax, (0.3, 5.3), 2.2, 1.0, "UCI Heart\nDisease Dataset")
    box(ax, (2.9, 5.3), 2.2, 1.0, "EDA +\nPreprocessing\nPipeline")
    box(ax, (5.5, 5.3), 2.2, 1.0, "Train LR + RF\n(GridSearchCV,\n5-fold CV)")
    box(ax, (8.1, 5.3), 2.2, 1.0, "MLflow Tracking\n(params/metrics/\nartifacts)")
    box(ax, (10.7, 5.3), 2.0, 1.0, "models/model.pkl\n(best pipeline)")

    arrow(ax, (2.5, 5.8), (2.9, 5.8))
    arrow(ax, (5.1, 5.8), (5.5, 5.8))
    arrow(ax, (7.7, 5.8), (8.1, 5.8))
    arrow(ax, (10.3, 5.8), (10.7, 5.8))

    # Row 2: CI/CD
    box(
        ax, (0.3, 3.6), 2.6, 1.0, "GitHub Actions:\nLint -> Unit Tests", color="#6a994e"
    )
    box(
        ax, (3.2, 3.6), 2.6, 1.0, "GitHub Actions:\nModel Training Job", color="#6a994e"
    )
    box(
        ax,
        (6.1, 3.6),
        2.6,
        1.0,
        "GitHub Actions:\nDocker Build +\nContainer Smoke Test",
        color="#6a994e",
    )
    arrow(ax, (2.9, 4.1), (3.2, 4.1))
    arrow(ax, (5.8, 4.1), (6.1, 4.1))
    arrow(ax, (11.7, 5.3), (7.4, 4.6))  # model artifact feeds docker build

    # Row 3: deployment
    box(ax, (0.3, 1.6), 2.4, 1.0, "Docker Image\nheart-disease-api", color="#bc4749")
    box(
        ax,
        (3.0, 1.6),
        2.6,
        1.0,
        "Kubernetes Deployment\n(2 replicas, probes)",
        color="#bc4749",
    )
    box(ax, (5.9, 1.6), 2.4, 1.0, "Kubernetes Service\n(LoadBalancer)", color="#bc4749")
    box(ax, (8.6, 1.6), 1.9, 1.0, "Client /\nPostman / curl", color="#bc4749")
    arrow(ax, (2.7, 2.1), (3.0, 2.1))
    arrow(ax, (5.6, 2.1), (5.9, 2.1))
    arrow(ax, (8.3, 2.1), (8.6, 2.1))
    arrow(ax, (1.5, 3.6), (1.5, 2.6))  # docker-build-test stage produces the image

    # Row 4: monitoring
    box(ax, (3.0, 0.1), 2.6, 1.0, "Prometheus\n(scrapes /metrics)", color="#e09f3e")
    box(ax, (5.9, 0.1), 2.4, 1.0, "Grafana Dashboard", color="#e09f3e")
    arrow(ax, (4.2, 1.6), (4.2, 1.1))
    arrow(ax, (5.6, 0.6), (5.9, 0.6))

    ax.set_title(
        "Heart Disease Risk API -- End-to-End MLOps Architecture",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=160)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
