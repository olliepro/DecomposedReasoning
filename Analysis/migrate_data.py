import os
import subprocess
from pathlib import Path


def migrate_data():
    base_dir = Path("output")
    if not base_dir.exists():
        print("Output directory not found.")
        return

    # Find all run directories
    run_dirs = [
        d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("branch_")
    ]

    if not run_dirs:
        print("No run directories found.")
        return

    print(f"Found {len(run_dirs)} run directories.")

    # Construct command
    cmd = ["python", "build_report.py"]
    for run_dir in run_dirs:
        cmd.extend(["--run-dir", str(run_dir)])

    output_html = base_dir / "report.html"
    cmd.extend(["--output", str(output_html)])

    # Merge existing clustering caches to avoid re-clustering
    merged_cache = {}
    for run_dir in run_dirs:
        cache_file = run_dir / "cluster_prompt_cache.json"
        if cache_file.exists():
            try:
                import json

                cache_data = json.loads(cache_file.read_text())
                merged_cache.update(cache_data)
            except Exception as e:
                print(f"Warning: Failed to load cache from {cache_file}: {e}")

    cache_path = base_dir / "merged_cluster_cache.json"
    if merged_cache:
        import json

        cache_path.write_text(json.dumps(merged_cache))
        print(f"Merged {len(merged_cache)} clustering cache entries to {cache_path}")
        cmd.extend(["--cluster-cache", str(cache_path)])
    else:
        print("No existing clustering caches found. Clustering will run from scratch.")

    # Clustering is desired for the final report.
    # We remove --disable-clustering to allow the build script to use Gemini (or fallback) for clustering.
    # Note: This requires the user to have their environment set up for Gemini if they want semantic clustering.
    # cmd.append("--disable-clustering")

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Migration complete. Output at {output_html}")


if __name__ == "__main__":
    migrate_data()
