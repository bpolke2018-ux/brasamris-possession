"""Print the context model's pseudo-label diagnostics (aggregates only).

    python -m src.context_report
"""

from src import atd, context, features

if __name__ == "__main__":
    a = atd.load()
    f = features.compute(a)
    report = {}
    context.context_llr(a, f, report)
    for key, value in report.items():
        print("{:<16} {}".format(key, value))
