"""Package corrected source, thesis and the exact available evidence without rewriting it."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import io
import json
import platform
from pathlib import Path
import subprocess
import tarfile

from scripts.m5r_post_training_evaluation import METHODS
from utils.paths import results_root

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(root):
    # Include tracked files plus new reviewable source, never environments/caches.
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
    return [root/name for name in set(names) if name and (root/name).is_file()
            and '.git' not in Path(name).parts and not (root/name).is_symlink()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--thesis-root',required=True,type=Path)
    parser.add_argument('--pdf',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    thesis=args.thesis_root.resolve(); result=results_root()
    files={f'code/{p.relative_to(ROOT)}':p for p in source_files(ROOT)}
    files.update({f'thesis/{p.relative_to(thesis)}':p for p in source_files(thesis)
                  if p.suffix not in ('.pdf','.log','.aux','.fls','.fdb_latexmk','.out','.toc','.bbl','.blg')})
    # Generated tables may be ignored by a checkout, but are required build inputs.
    files.update({f'thesis/tables/{p.name}':p for p in (thesis/'tables').glob('*.tex')})
    files['thesis/main.pdf']=args.pdf.resolve()
    folders=[result/'analysis',result/'foundations']
    for name in ('thesis_contract.json','thesis_contract_run.json'):
        path=result/'audits'/name
        if path.is_file():files[f'code/results/audits/{name}']=path
    folders += [result/'medium'/experiment for _,experiment in METHODS]
    folders += sorted(result.glob('m5_factorial_*'))
    folders += sorted((result/'_archive').glob('m4_*'))
    for folder in folders:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.pkl','.png') and '.dummy.' not in p.name:
                files[f'code/results/{p.relative_to(result)}']=p
    evidence={}
    for method,experiment in METHODS:
        folder=result/'medium'/experiment
        config=json.loads((folder/'config.json').read_text())
        metrics=json.loads((folder/'metrics.json').read_text())
        checkpoints={p.name:sha(p) for p in sorted(folder.glob('checkpoint_seed_*.pkl'))}
        if set(checkpoints) != {f'checkpoint_seed_{s}.pkl' for s in range(20)}:
            raise ValueError(f'Incomplete checkpoint set: {experiment}')
        evidence[method]={'config_iterations':config['iterations'], 'metrics_iterations':metrics['iterations'],
            'per_seed_curve_lengths':[len(c) for c in metrics['per_seed_mean_return_per_iter']],
            'checkpoint_sha256':checkpoints,
            'interpretation':'Saved curves document 1500 iterations. Original metadata are preserved; no full historical retraining verification record is available.'}
    import jax
    environment={'recorded_at':datetime.now(timezone.utc).isoformat(),
                 'purpose':'Verification environment, not a reconstructed original training record',
                 'original_training_hardware':'Not recorded in the supplied evidence',
                 'platform':platform.platform(),'machine':platform.machine(),'python':platform.python_version(),
                 'devices':[str(d) for d in jax.devices()],
                 'packages':{name:importlib.metadata.version(name) for name in
                             ('jax','jaxlib','flax','optax','chex','numpy','pyyaml','matplotlib','scikit-learn','scipy','pytest')}}
    extras={'ENVIRONMENT.json':json.dumps(environment,indent=2).encode(),
            'TRAINING_EVIDENCE.json':json.dumps(evidence,indent=2).encode(),
            'README.txt':b'Verify with: python3 code/scripts/verify_reproduction_bundle.py .\nThen follow code/REPRODUCE.md.\nThis is a content-addressed local companion archive, not a public tagged release.\n'}
    hashes={name:sha(path) for name,path in sorted(files.items())}
    hashes.update({name:hashlib.sha256(raw).hexdigest() for name,raw in extras.items()})
    manifest={'schema_version':1,'description':'October 2026 corrected thesis companion archive',
              'code_base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'source_identity':'Exact files are identified by SHA-256, including uncommitted corrections',
              'files_sha256':hashes}
    extras['MANIFEST.json']=json.dumps(manifest,indent=2).encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    prefix=args.output.name.removesuffix('.tar.gz')
    with tarfile.open(args.output,'w:gz',compresslevel=6) as archive:
        for name,path in sorted(files.items()):
            archive.add(path,arcname=f'{prefix}/{name}',recursive=False)
        for name,raw in extras.items():
            info=tarfile.TarInfo(f'{prefix}/{name}');info.size=len(raw);info.mode=0o644
            archive.addfile(info,io.BytesIO(raw))
    checksum=sha(args.output)
    args.output.with_name(args.output.name+'.sha256').write_text(f'{checksum}  {args.output.name}\n')
    print(f'{args.output}: {args.output.stat().st_size:,} bytes; {len(hashes)} files; sha256={checksum}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
