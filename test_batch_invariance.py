import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from batch_invariant_ops import set_batch_invariant_mode

console = Console()

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)

# Just to get the logging out of the way haha
with set_batch_invariant_mode(True):
    pass


def test_batch_invariance(dtype=torch.float32):
    B, D = 2048, 4096
    a = torch.linspace(-1000, 1000, B * D, dtype=dtype).reshape(B, D)
    b = torch.linspace(-1000, 1000, D * D, dtype=dtype).reshape(D, D)

    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = torch.mm(a[:1], b)

    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out2 = torch.mm(a, b)[:1]

    # Check if results are identical
    diff = (out1 - out2).abs().max()
    return diff.item() == 0, diff


def run_iters(iters=100):
    table = Table(
        title="Batch Invariance Test Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Data Type", style="cyan", justify="center")
    table.add_column("Deterministic", justify="center")
    table.add_column("Max Diff", justify="right", style="yellow")
    table.add_column("Min Diff", justify="right", style="yellow")
    table.add_column("Range", justify="right", style="yellow")
    table.add_column("Iterations", justify="center", style="green")

    for dtype in [torch.float32, torch.bfloat16]:
        is_deterministic = True
        difflist = []
        for i in range(iters):
            isd, df = test_batch_invariance(dtype)
            is_deterministic = is_deterministic and isd
            difflist.append(df)

        deterministic_icon = "✓" if is_deterministic else "✗"
        deterministic_style = "green" if is_deterministic else "red"

        table.add_row(
            str(dtype),
            f"[{deterministic_style}]{deterministic_icon}[/{deterministic_style}]",
            f"{max(difflist):.2e}",
            f"{min(difflist):.2e}",
            f"{max(difflist)-min(difflist):.2e}",
            str(iters),
        )

    console.print(table)


# Test with standard PyTorch (likely to show differences)
console.print(Panel.fit("Standard PyTorch", style="bold red", border_style="red"))
with set_batch_invariant_mode(False):
    run_iters()

console.print()  # Add spacing

# Test with batch-invariant operations
console.print(Panel.fit("Batch-Invariant Mode", style="bold green", border_style="green"))
with set_batch_invariant_mode(True):
    run_iters()
