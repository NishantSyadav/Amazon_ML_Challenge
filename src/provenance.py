"""Bind inference and measured validation evidence to shipped production source."""
import hashlib
from pathlib import Path

PRODUCTION_FILES=['src/baseline_features.py','src/disk_candidates.py','src/features.py',
                  'src/inference.py','src/pipeline.py','src/provenance.py','requirements.txt']


def source_manifest(root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in PRODUCTION_FILES}


def verify_provenance(run,validation,root):
    if run.get('source_code')!=source_manifest(root):
        raise ValueError('Packaged inference source differs from code used for inference')
    if not validation or validation.get('validation_entities')!=1000 or validation.get('training_entities')!=4000 or validation.get('split_overlap')!=0:
        raise ValueError('Measured held-out validation evidence is required')
    if validation.get('source_code')!=run['source_code'] or validation.get('index_version')!=run['index_version'] or validation.get('model',{}).get('sha256')!=run['model']['sha256']:
        raise ValueError('Held-out validation source/index/model differs from final inference')
