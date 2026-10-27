"""Convert an XES event log to a flat CSV (one row per event).

Dependency-free: the log is streamed with the standard-library
`xml.etree.ElementTree.iterparse`, so large logs are converted in constant
memory. Gzipped logs (`.xes.gz`) are read directly.

Two passes over the file:
    1. collect the union of trace-level and event-level attribute keys;
    2. write one row per event.

Trace attributes are written with a `case:` prefix (e.g. `concept:name` of the
trace becomes `case:concept:name`), event attributes keep their own key, which
matches the column conventions of the `create_*` dataset builders. Trace
attributes are expected before the trace's events, as in standard XES. Only
attributes that are direct children of a trace or event are exported; nested
(meta-)attributes are ignored. Values are written verbatim as strings; typing
is left to the dataset builders.

Usage:
    python Preprocessing/xes_to_csv.py <input.xes[.gz]> <output.csv>
"""

import csv
import gzip
import sys
import xml.etree.ElementTree as ET

ATTRIBUTE_TAGS = {"string", "date", "int", "float", "boolean", "id"}
CASE_PREFIX = "case:"

# Columns placed first, in this order, when present.
_LEADING_EVENT_COLUMNS = ("concept:name", "time:timestamp", "lifecycle:transition",
                          "org:resource")


def _local(tag):
    """Strip the XML namespace from a tag."""
    return tag.rsplit("}", 1)[-1]


def _open(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def _iter_attributes(path):
    """Yield ("trace" | "event" | "end_event" | "end_trace", key, value) items.

    Attribute items carry the owning element type ("trace" or "event") with the
    attribute key and value; "end_event" / "end_trace" items mark element ends.
    """
    stack = []
    with _open(path) as handle:
        for kind, elem in ET.iterparse(handle, events=("start", "end")):
            tag = _local(elem.tag)
            if kind == "start":
                parent = stack[-1] if stack else None
                if tag in ATTRIBUTE_TAGS and parent in ("trace", "event"):
                    key = elem.get("key")
                    if key is not None:
                        yield parent, key, elem.get("value")
                stack.append(tag)
            else:
                stack.pop()
                if tag == "event":
                    yield "end_event", None, None
                    elem.clear()
                elif tag == "trace":
                    yield "end_trace", None, None
                    elem.clear()


def collect_keys(path):
    """Return (trace_keys, event_keys) in order of first appearance."""
    trace_keys, event_keys = {}, {}
    for owner, key, _ in _iter_attributes(path):
        if owner == "trace":
            trace_keys.setdefault(key, None)
        elif owner == "event":
            event_keys.setdefault(key, None)
    return list(trace_keys), list(event_keys)


def _column_order(trace_keys, event_keys):
    case_cols = [CASE_PREFIX + k for k in trace_keys]
    case_id = CASE_PREFIX + "concept:name"
    lead_case = [case_id] if case_id in case_cols else []
    lead_event = [k for k in _LEADING_EVENT_COLUMNS if k in event_keys]
    rest_event = [k for k in event_keys if k not in lead_event]
    rest_case = [c for c in case_cols if c not in lead_case]
    return lead_case + lead_event + rest_event + rest_case


def convert(in_path, out_path):
    """Write `in_path` (XES) as a CSV with one row per event to `out_path`."""
    trace_keys, event_keys = collect_keys(in_path)
    columns = _column_order(trace_keys, event_keys)

    n_events = n_traces = 0
    trace, event = {}, {}
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=columns, restval="")
        writer.writeheader()
        for owner, key, value in _iter_attributes(in_path):
            if owner == "trace":
                trace[CASE_PREFIX + key] = value
            elif owner == "event":
                event[key] = value
            elif owner == "end_event":
                writer.writerow({**event, **trace})
                event = {}
                n_events += 1
            elif owner == "end_trace":
                trace = {}
                n_traces += 1

    print(f"wrote {n_events} events across {n_traces} traces "
          f"({len(columns)} columns) -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    convert(sys.argv[1], sys.argv[2])
