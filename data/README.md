# Data

## `dilemmas.csv`

Master items in English, one row per item. UTF-8, comma-separated. Check the file with `python -m src.schema`.

| Column | Description |
|---|---|
| `item_id` | Stable, unique identifier |
| `type` | `dilemma` or `control` |
| `domain` | Life/death, privacy, promises, resources, honesty and so on. Required for dilemmas |
| `scenario_en` | Neutral text with no culturally marked names or countries |
| `option_1_en` | Option that favours `value_1`. For controls, the correct answer |
| `option_2_en` | Option that favours `value_2`. For controls, the wrong answer |
| `value_1`, `value_2` | The values in conflict. Required for dilemmas, empty for controls |
| `label_annotator2_v1`, `label_annotator2_v2` | Second annotator's labels, filled in at step 8 |
| `notes` | Ambiguities and source (original or adapted, with licence) |

## Option order

Option 1 always favours `value_1` in this file. Whether an option is shown as A or B is decided when the prompt is built, so the order can be swapped without touching the data.

## Other folders

- `translations/`: one file or folder per language
- `results/`: raw outputs, kept in the repository
