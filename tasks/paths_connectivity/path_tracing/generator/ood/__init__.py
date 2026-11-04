"""Constants of the Path Tracing out-of-distribution (xxhard) generator."""
from visro.seeding import seed_text

BASE_GENERATOR_VERSION = "path-tracing"
GENERATOR_VERSION = "path-tracing-ood"
GENERATOR_ENTRYPOINT = 'tasks.paths_connectivity.path_tracing.generator.ood.generate_eval'
DATASET_ROLE = "ood_eval"
DATASET_SPLIT = "eval"
# fixed RNG namespace (changing it changes every item)
RNG_NAMESPACE = seed_text(11835067211139708856667541050200742659226899637955148744019622113328)
ID_PREFIX = "path_ood_"
