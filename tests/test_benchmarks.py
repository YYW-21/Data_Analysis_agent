from datapilot.benchmarks import run_benchmark


def test_run_small_benchmark(tmp_path) -> None:
    result = run_benchmark(output_dir=tmp_path, use_agent=False, max_cases=2)

    assert result["summary"]["total_cases"] == 2
    assert result["summary"]["passed_cases"] >= 1
    assert (tmp_path / "benchmark_results.json").exists()
    assert (tmp_path / "benchmark_results.csv").exists()
    assert (tmp_path / "benchmark_report.md").exists()
