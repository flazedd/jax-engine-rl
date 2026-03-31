"""Simple metrics logger."""


class Logger:
    def __init__(self):
        self.history = []
        self._header_printed = False

    def log(self, iteration: int, metrics: dict):
        entry = {"iteration": iteration}
        for k, v in metrics.items():
            entry[k] = float(v) if hasattr(v, 'item') else v
        self.history.append(entry)

    def print_summary(self, last_n=10):
        recent = self.history[-last_n:]
        if not recent:
            return
        keys = [k for k in recent[0] if k != "iteration"]
        if not self._header_printed:
            header = f"{'iter':>6} " + " ".join(f"{k:>12}" for k in keys)
            print(header)
            self._header_printed = True
        entry = recent[-1]
        vals = f"{entry['iteration']:>6} " + " ".join(
            f"{entry.get(k, 0.0):>12.4f}" for k in keys
        )
        print(vals)

    def get_means(self, key: str, last_n=None):
        entries = self.history if last_n is None else self.history[-last_n:]
        return [e[key] for e in entries if key in e]
