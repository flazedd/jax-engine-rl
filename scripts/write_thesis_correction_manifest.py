"""Record the exact local code and saved results used by the corrected thesis."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from utils.paths import analysis_dir

ROOT = Path(__file__).resolve().parents[1]


def main():
    code = [p for directory in ('agents', 'beliefs', 'envs', 'training', 'evaluation',
                                'scripts', 'utils', 'plotting', 'oracles', 'tests')
            for p in (ROOT/directory).rglob('*.py')]
    code += list((ROOT/'experiments/configs').rglob('*.yaml'))
    code += [p for p in (ROOT/'reproduction').rglob('*') if p.is_file()]
    code += [ROOT/'uv.lock', ROOT/'pyproject.toml', ROOT/'REPRODUCE.md']
    outputs = [analysis_dir()/name for name in (
        'm5r_post_training_evaluation.json', 'm5r_posterior_vs_performance.json',
        'm5r_posterior_vs_performance_mlp.json', 'm5r_refresh_probe_returns_run.json',
        'm5r_hypothesis_tests.json', 'm5r_belief_swap.json',
        'm5r_belief_swap_run.json', 'm5r_diagnostic_tests.json',
        'm5r_action_distributions.json', 'm5r_belief_quality_tests.json',
        'm5r_belief_quality_mlp_tests.json', 'param_counts.json',
        'm5r_probe_confusion.json', 'm5r_method_return_tests.json',
        'm5r_time_to_threshold_tests.json', 'm5r_seed_block_sensitivity.json')]
    def hashes(paths):
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(paths)}
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'base_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'revision_description': 'Local audit corrections; file hashes identify uncommitted code precisely',
        'training_commit': '220b319f367e69a8f466c87d5346b1acf60574d6',
        'unchanged': ['trained checkpoints', 'fresh evaluation return values', 'fitted probe measurements'],
        'corrections': ['experiment/seed join to fresh evaluation returns',
                        'independent stratified run bootstrap, recentered per draw',
                        'actor-only RL2 substitution from the updated hidden state',
                        'unweighted inventory aggregation documented',
                        'unaligned exact-policy substitution reference removed',
                        'temporal curves smoothed before bootstrap percentile intervals',
                        'optimal-policy heatmap displays all numerical ties',
                        'module parameter counts checked against thesis table',
                        'executed KL normalization, filter model and omission screen documented',
                        'shared training randomness disclosed and all 18 seed-block sensitivities reported',
                        'isolated schema/render fixtures and content-hashed complete dependency graph',
                        'fail-closed full training identity and checkpoint integrity',
                        'complete required-artifact and comparison contract'],
        'code_sha256': hashes(code), 'analysis_sha256': hashes(outputs),
    }
    path=analysis_dir()/'thesis_correction_manifest.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(path)

if __name__ == '__main__':
    main()
