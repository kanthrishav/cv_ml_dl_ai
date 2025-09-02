# /ssd/agents/astro/api/app/cli.py  (NEW FILE)
from __future__ import annotations
import argparse, json, sys, time
import httpx

def main():
    p = argparse.ArgumentParser(description="astro-api CLI (submit + auto-stream)")
    p.add_argument("--api-url", default="http://127.0.0.1:8088")
    p.add_argument("--project", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--model", default="llama3.2:3b")
    p.add_argument("--max-iters", type=int, default=5)
    p.add_argument("--no-stream", action="store_true")
    args = p.parse_args()

    payload = {
        "problem": args.task,
        "project_name": args.project,
        "use_model": args.model,
        "max_iterations": args.max_iters,
    }
    with httpx.Client(timeout=None) as x:
        r = x.post(f"{args.api_url}/run", json=payload)
        r.raise_for_status()
        j = r.json()
        run_id = j.get("run_id")
        if not run_id:
            print("Failed to create run:", j, file=sys.stderr); sys.exit(1)
        print("run_id=", run_id)
        print("workspace=", j.get("workspace"))

        if args.no_stream:
            return

        with x.stream("GET", f"{args.api_url}/stream/{run_id}") as s:
            for line in s.iter_lines():
                if not line: continue
                if line.startswith("data: "):
                    print(line[6:])

if __name__ == "__main__":
    main()

