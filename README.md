# LLM ARC

## Setup conda environment

```bash
conda create --name lmm-arc python
conda activate lmm-arc
pip install -r requirements.txt
```

## Download data

- Download data from [https://lab42.global/arcathon/](https://lab42.global/arcathon/guide/) and store it in `data/` folder.
- Clone LARC repo: `git clone https://github.com/samacqua/LARC.git`

## Generate ARC dataset

Without additional descriptions:
```bash
python generate_arc_dataset.py
```

With additional descriptions:
```bash
python generate_arc_dataset.py --add_descriptions
```

Generate `n` samples of task `08ed6ac7`:
```bash
python simple_generate.py --n_samples <n>
```
