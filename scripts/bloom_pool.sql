-- Labelled good/bad pool export for the bloom filter producer (scan-bloom-build).
-- Streams NDJSON to stdout: one {purl?, sha256, label} object per line.
--
-- Run against a local hopper replica, e.g.
--   psql postgres://hopper@localhost:5432/hopper -X -q -f scripts/bloom_pool.sql
-- (this is what `make publish-bloom` does by default).
--
-- Tiers, using the criticality scale hopper's samples_derive_cleave_cols trigger
-- materializes onto `max_crit` (a finding's `crit` >= 4 is suspicious, >= 5 is
-- hostile):
--   good = explicitly labelled good with NO hostile finding   (max_crit < 5)
--   bad  = labelled bad AND carrying a suspicious-or-hostile finding (max_crit >= 4)
--
-- Only top-level samples (parent = ''): archive members carry no independent
-- label. A versioned PURL is emitted only when both the version-less identity
-- (purl_base) and a concrete version are known, so a good bless is pinned to the
-- exact release the scanner will look up; otherwise the row contributes by sha256.
--
-- Freshness: only samples analyzed within `max_age_days` are included, so a
-- bless reflects the current model/ruleset rather than a stale verdict. Override
-- with -v max_age_days=N (the Makefile passes BLOOM_MAX_AGE_DAYS; default 90).

-- Default the age cutoff only when the caller didn't supply one via -v.
\if :{?max_age_days}
\else
\set max_age_days 90
\endif

\set FETCH_COUNT 50000
\pset format unaligned
\pset tuples_only on

SELECT json_build_object(
         'purl', CASE
                   WHEN purl_base <> '' AND version <> ''
                   THEN purl_base || '@' || version
                 END,
         'sha256', sha256,
         'label', label)
FROM samples
WHERE parent = ''
  AND analyzed_at >= now() - make_interval(days => :max_age_days)
  AND ( (label = 'good' AND max_crit < 5)
     OR (label = 'bad'  AND max_crit >= 4) );
