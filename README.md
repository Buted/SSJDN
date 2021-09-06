# Treasures Outside Contexts: Improving Event Detection via Global Statistics

## Preprocessing (for ACE data)

Generate `train.json`, `dev.json` and `test.json` according to [data_list.csv](./data_list.csv). Read  [ace2005-preprocessing](https://github.com/nlpcl-lab/ace2005-preprocessing) for details.

## Preprocessing (for all datasets)

Move `train.json`, `dev.json` and `test.json` to [ACE](./ACE) / [KBP14](./KBP14) / [KBP15](./KBP15) dir.


## Requirements

See [requirements.txt](./requirements.txt).

## SSJDN Code

Train the model:

~~~bash
bash scripts/train.sh
~~~

The result can be seen in `models/*/log`.

For more details, see [enet](./enet).