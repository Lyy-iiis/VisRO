"""count_clusters verifier: recount the selected instances from the scene description."""
from visro import runners

TASK = 'count_clusters'
CORRECTNESS = 'Recount primitive attributes, region membership and overlap masks; check the schedule and the target IDs.'
UNIQUENESS = 'The question predicate selects a determinate set of instances.'


def _check(record, meta, original, image):
    from tasks.discrimination_grouping.count_clusters.generator import production, recount
    r = original['production']
    production.validate_record(original, r['generation_index'], r['split'], r['seed'],
                               {'sha256': r['source_fingerprint']})
    expected = recount.expected_target_ids(original, 448)
    if expected != set(original['meta']['target_ids']) or str(len(expected)) != record['answer']:
        raise ValueError('count predicate/mask recount mismatch')


def verify(record, image):
    return runners.run_checks(TASK, record, image, _check)
