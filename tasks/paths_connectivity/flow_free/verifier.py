"""flow_free verifier: re-run the board validator on the serialized paths."""
from visro import runners

TASK = 'flow_free'
CORRECTNESS = 'Run the production validator on the paths, lengths/turns, schedule and rendered image hash.'
UNIQUENESS = 'Path traversal determines the requested statistics and ordering.'


def _check(record, meta, original, image):
    from tasks.paths_connectivity.flow_free.generator import production
    production.validate_record(original, image, total=meta['audit']['total'])


def verify(record, image):
    return runners.run_checks(TASK, record, image, _check)
