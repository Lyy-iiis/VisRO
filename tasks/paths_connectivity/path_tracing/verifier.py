"""path_tracing verifier: recompute endpoints or dot ownership from the serialized curves."""
from visro import runners

TASK = 'path_tracing'
CORRECTNESS = 'Recompute endpoints or dot ownership from the serialized curves and targets; check geometry validity.'
UNIQUENESS = 'Each queried marker or dot belongs to its declared curve and endpoint mapping.'


def _tier(original):
    tier = runners.generator_tier(original)
    return original['path_tracing']['difficulty'] if tier is None else tier


def _check(record, meta, original, image):
    from tasks.paths_connectivity.path_tracing.generator.answers import structured_answer
    m = original.get('path_tracing', original.get('path_tracing_ood'))
    qa = m['qa_type']
    curves = m['curves']
    targets = m['target_indices']
    if not targets or len(set(targets)) != len(targets):
        raise ValueError('invalid path targets')
    selected = [curves[i] for i in targets]
    if qa == 'same_line':
        policy = m.get('same_line_query_policy') or m.get('same_line_query') or {}
        indices = policy.get('dot_curve_indices')
        if indices is None:
            indices = [dot['curve_index'] for dot in m['dot_markers']]
        if len(indices) != 2 or any(not 0 <= i < len(curves) for i in indices):
            raise ValueError('invalid path dot ownership')
        expected = indices[0] == indices[1]
    elif qa == 'one_marker':
        if len(selected) != 1:
            raise ValueError('one-marker query must select one path')
        curve = selected[0]
        if f"What is {curve['start_label']} connected to?" in original['question']:
            expected = str(curve['end_label']).upper()
        elif f"What is {curve['end_label']} connected to?" in original['question']:
            expected = str(curve['start_label']).upper()
        else:
            raise ValueError('one-marker question does not identify a curve endpoint')
    elif qa == 'ordered':
        expected = [str(c['end_label']).upper() for c in selected]
    elif qa == 'all_pairs':
        expected = sorted([[str(c['start_label']).upper(), str(c['end_label']).upper()] for c in selected])
    else:
        raise ValueError('unknown path question type')
    actual = structured_answer(qa, runners.answer(original), expected if qa == 'ordered' else None)
    if actual != expected:
        raise ValueError('path endpoint/ownership answer mismatch')
    if not (m.get('valid') or m.get('validity', {}).get('valid')):
        raise ValueError('path geometry validity failed')


def verify(record, image):
    return runners.run_checks(TASK, record, image, _check, tier_of=_tier)
