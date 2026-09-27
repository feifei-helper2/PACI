"""Train PACI, extract features and evaluate clustering."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from .config import BYOLConfig, RelationConfig, GeometryConfig


def read_config(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    allowed = {'dataset', 'imbalance_factor', 'seed', 'byol', 'relation', 'geometry'}
    unknown = set(obj) - allowed
    if unknown:
        raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
    if obj['dataset'] not in {'cifar10', 'cifar20', 'cifar100', 'stl10', 'tinyimagenet'}:
        raise ValueError('Unsupported dataset')
    byol = BYOLConfig(**obj.get('byol', {}))
    relation = RelationConfig(**obj.get('relation', {}))
    geometry = GeometryConfig(**obj.get('geometry', {}))
    if byol.epochs != relation.total_epochs:
        raise ValueError('Optimizer horizon and total training epochs must agree')
    if not 0 < relation.shared_warmup_epochs == relation.relation_start_epoch < relation.total_epochs:
        raise ValueError('Require 0 < warm-up = relation start < total epochs')
    if byol.batch_size < 3 or byol.num_workers < 0 or byol.checkpoint_every < 1:
        raise ValueError('Require batch_size >= 3, num_workers >= 0, checkpoint_every >= 1')
    if relation.relation_refresh_epochs < 1 or relation.relation_temperature <= 0 or relation.relation_beta_max <= 0:
        raise ValueError('PACI requires positive relation weight, temperature and refresh interval')
    if relation.relation_ramp_epochs < 0 or geometry.pca_dim < 1:
        raise ValueError('Invalid ramp or PCA dimension')
    if not 1 <= float(obj['imbalance_factor']) < float('inf'):
        raise ValueError('Require finite imbalance_factor >= 1')
    return obj, byol, relation, geometry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train', help='BYOL warm-up followed by capacity-matched PACI')
    train.add_argument('--config', required=True)
    train.add_argument('--data-root', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--seed', type=int, help='Override the seed in the config')
    train.add_argument('--workers', type=int, help='Override data-loader workers')
    train.add_argument('--resume', action='store_true')
    train.add_argument('--no-download', action='store_true')
    train.add_argument('--stop-after', type=int, help='Stop at this epoch without shortening the 800-epoch LR horizon')
    train.add_argument('--dry-run', action='store_true', help='Validate and display configuration without loading data')
    ev = sub.add_parser('evaluate', help='Evaluate the saved feature bank; labels are used only for metrics')
    ev.add_argument('--run', required=True)
    extract = sub.add_parser('extract', help='Regenerate the target feature bank from the final checkpoint')
    extract.add_argument('--run', required=True)
    extract.add_argument('--data-root', required=True)
    args = parser.parse_args(argv)
    if args.command == 'train':
        obj, byol, relation, geometry = read_config(args.config)
        from dataclasses import replace
        seed = int(args.seed if args.seed is not None else obj.get('seed', 4801))
        if args.workers is not None:
            if args.workers < 0:
                parser.error('--workers must be >= 0')
            byol = replace(byol, num_workers=args.workers)
        resolved = {'dataset': obj['dataset'], 'imbalance_factor': float(obj['imbalance_factor']),
                    'seed': seed, 'byol': asdict(byol), 'relation': asdict(relation), 'geometry': asdict(geometry)}
        final = args.stop_after if args.stop_after is not None else relation.total_epochs
        if not 1 <= final <= relation.total_epochs:
            parser.error('--stop-after must lie between 1 and total_epochs')
        if args.dry_run:
            print(json.dumps(resolved, indent=2))
            return
        import numpy as np
        import torch
        from .data import build_datasets
        from .model import train_warmup, extract_target_projector_features
        from .training import train_paci, LocalPositionDataset
        from .utils import write_json
        if not torch.cuda.is_available():
            raise RuntimeError('Full PACI training requires a CUDA-enabled PyTorch installation and GPU')
        run = Path(args.output)
        if run.exists() and any(run.iterdir()) and not args.resume:
            raise FileExistsError('Output is not empty; use --resume or choose a new output directory')
        run.mkdir(parents=True, exist_ok=True)
        config_path = run / 'config.json'
        if config_path.exists():
            previous = json.loads(config_path.read_text(encoding='utf-8'))
            if previous != resolved:
                raise ValueError('Resume configuration differs from the saved run')
        ds = build_datasets(obj['dataset'], args.data_root, obj['imbalance_factor'], seed,
                            byol.crop_scale_min, byol.gaussian_blur, download=not args.no_download)
        digest = hashlib.sha256(ds.indices.tobytes()).hexdigest()
        subset_path = run / 'subset_indices.npy'
        if subset_path.exists() and not np.array_equal(np.load(subset_path), ds.indices):
            raise ValueError('Dataset subset differs from the saved run')
        write_json(config_path, resolved)
        np.save(subset_path, ds.indices)
        write_json(run / 'dataset.json', {'n': len(ds.labels), 'k': ds.num_classes,
                   'counts': ds.desired_counts.tolist(), 'subset_sha256': digest})
        warmup = run / 'warmup'
        checkpoint = warmup / 'checkpoint_last.pt'
        existing_epochs = []
        for cp in [checkpoint, run / 'training/checkpoint_last.pt']:
            if cp.exists():
                existing_epochs.append(int(torch.load(cp, map_location='cpu', weights_only=True)['epoch']))
        if existing_epochs and final < max(existing_epochs):
            raise ValueError('--stop-after precedes a saved checkpoint')
        model, _ = train_warmup(obj['dataset'], LocalPositionDataset(ds.train), byol, seed,
                                 warmup, resume=args.resume,
                                 epochs_override=min(final, relation.shared_warmup_epochs))
        if final <= relation.shared_warmup_epochs:
            print(f'Warm-up saved at epoch {final}; resume this output to finish PACI.')
            return
        del model
        torch.cuda.empty_cache()
        model, _ = train_paci(obj['dataset'], ds.train, ds.eval, ds.num_classes,
                              byol, relation, geometry, seed, checkpoint, run / 'training',
                              resume=args.resume, final_epoch_override=final)
        if final == relation.total_epochs:
            extract_target_projector_features(model, ds.eval, byol, run / 'features.npz', resolved)
            from .evaluation import evaluate_features
            write_json(run / 'metrics.json', evaluate_features(run / 'features.npz', ds.num_classes,
                       byol.byol_kmeans_n_init, byol.byol_kmeans_max_iter))
        print(f'PACI checkpoint saved at epoch {final}: {run}')
    else:
        from .utils import write_json
        run = Path(args.run)
        obj, byol, relation, geometry = read_config(run / 'config.json')
        info = json.loads((run / 'dataset.json').read_text(encoding='utf-8'))
        if args.command == 'extract':
            import numpy as np
            import torch
            from .data import build_datasets
            from .model import BYOLModel, extract_target_projector_features
            ds = build_datasets(obj['dataset'], args.data_root, obj['imbalance_factor'], obj['seed'],
                                byol.crop_scale_min, byol.gaussian_blur, download=False)
            if not np.array_equal(ds.indices, np.load(run / 'subset_indices.npy')):
                raise ValueError('Dataset subset does not match training')
            ckpt = torch.load(run / 'training/checkpoint_last.pt', map_location='cpu', weights_only=True)
            if int(ckpt['epoch']) != relation.total_epochs:
                raise ValueError('Extraction requires a completed training run')
            model = BYOLModel(obj['dataset'], byol).cuda()
            model.load_state_dict(ckpt['model'])
            extract_target_projector_features(model, ds.eval, byol, run / 'features.npz', obj)
        else:
            from .evaluation import evaluate_features
            metrics = evaluate_features(run / 'features.npz', info['k'],
                         byol.byol_kmeans_n_init, byol.byol_kmeans_max_iter)
            write_json(run / 'metrics.json', metrics)
            print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
