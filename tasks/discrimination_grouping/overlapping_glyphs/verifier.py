"""overlapping_glyphs verifier: re-read the queried characters from the glyph metadata."""
from visro import runners

TASK = 'overlapping_glyphs'
CORRECTNESS = 'Recount queried characters in color-box order and validate the partition/query metadata.'
UNIQUENESS = 'Distinct color identities and the ordered query determine one character sequence.'


def _check(record, meta, original, image):
    from tasks.discrimination_grouping.overlapping_glyphs.generator import glyph_runner

    m = original['meta']
    condition = {'tier': record['tier'], 'mode': m['mode'], 'outline': m['outline'], 'subset': m['proper_subset']}
    glyph_runner.validate_meta(m, condition)
    expected = ','.join(m['glyphs'][i]['char'] for i in m['query_ids'])
    if expected != record['answer']:
        raise ValueError('glyph query-order recount mismatch')


def verify(record, image):
    return runners.run_checks(TASK, record, image, _check)
