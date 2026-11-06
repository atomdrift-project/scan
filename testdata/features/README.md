# Keyword feature parity fixtures

The unit tests in `src/features/keywords.rs` read these files. Each file is a
JSON list of cases:

| Field      | Meaning                                                    |
|------------|------------------------------------------------------------|
| `name`     | Case label, printed when a case fails.                     |
| `spec`     | The `feature_spec.json` the case extracts with.            |
| `report`   | A cleave compact report: the extractor's input.            |
| `expected` | The feature vector the reference extractor produced.       |
| `error`    | (`keyword_parity.json` only) The expected rejection, if any. |

- `keyword_parity.json`: ABI 22 keyword sources, including the strict
  missing-description contract.
- `extended_keyword_parity.json`: ABI 23 sources (`fullword`, `ruletextword`,
  `contextword`).
- `keyword_combinations_parity.json`: each of the 31 non-empty combinations of
  the five keyword sources, over the nine reports in
  `extended_keyword_parity.json`.

The `expected` vectors come from the Python reference extractor that trains
the models. Regenerate them there whenever extraction changes; never edit
them by hand to make a test pass.
