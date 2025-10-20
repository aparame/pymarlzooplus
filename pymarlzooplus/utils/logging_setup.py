from collections import defaultdict
import logging, re

import numpy as np
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from rich.table import Table
from rich.panel import Panel
from sacred.utils import apply_backspaces_and_linefeeds


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.tb_logger = None
        self._run_obj = None
        self.sacred_info = None

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

            self._run_obj.log_scalar(key, value, t)

    @staticmethod
    def _to_float(x):
        """Robustly convert numpy/torch scalars to float (NaN on failure)."""
        try:
            if hasattr(x, "item"):
                return float(x.item())
            return float(np.asarray(x).reshape(-1)[0])
        except Exception:
            return float("nan")

    def _get_rich_console(self):
        """Return the Rich Console from RichHandler if present, else None."""

        candidates = [self.console_logger, getattr(self.console_logger, "parent", None), logging.getLogger()]
        for lg in candidates:
            if not lg:
                continue
            for h in getattr(lg, "handlers", []):
                if h.__class__.__name__ == "RichHandler":
                    return getattr(h, "console", None)

        return None

    def print_recent_stats(self, default_window=5, epsilon_window=1):
        """
        Pretty logging of recent rolling stats.
        - default_window: rolling window for all metrics except 'epsilon'
        - epsilon_window: window for 'epsilon'
        - cols: columns in plain-text fallback
        - include_std: add a Std column (Rich only)
        """
        # Header values
        t_env, ep = self.stats["episode"][-1]

        # Collect (name, mean, std, window)
        items = []
        for k in sorted(self.stats.keys()):
            if k == "episode":
                continue
            window = epsilon_window if k.lower() == "epsilon" else default_window
            recent = self.stats[k][-window:]
            vals = [self._to_float(v) for _, v in recent] if recent else []
            mean_val = float(np.nanmean(vals)) if len(vals) else float("nan")
            std_val = float(np.nanstd(vals)) if len(vals) else float("nan")
            items.append((k, mean_val, std_val, window))

        # Render a nice table to the Rich console
        console = self._get_rich_console()
        assert console is not None, "Rich console not found in logger handlers."
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric", justify="left", no_wrap=True)
        table.add_column("Mean", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Average Window", justify="right")
        for name, mean_val, std_val, win in items:
            table.add_row(name, f"{mean_val:.4f}", f"{std_val:.4f}", f"{win}")
        header = f"Recent Stats  |  t_env: {t_env:,}  |  Episode: {ep}"
        console.print(Panel(table, title=header))


def get_logger(name="pymarlzooplus", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    install_rich_traceback(show_locals=False)
    rh = RichHandler(rich_tracebacks=True, markup=True, show_path=False, enable_link_path=False)
    rh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rh)
    logger.propagate = False

    return logger

# Use this for Sacred's captured_out_filter (cleans prints + Rich ANSI if they’re captured)
def captured_filter(text: str) -> str:
    text = apply_backspaces_and_linefeeds(text)
    return re.compile(r"\x1b\[[0-9;]*[mK]").sub("", text)





