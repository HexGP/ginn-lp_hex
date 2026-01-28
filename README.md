# GINN-LP: A Growing Interpretable Neural Network for Discovering Multivariate Laurent Polynomial Equations

This code is an implementation of the AAAI 2024 paper "GINN-LP: A Growing Interpretable Neural Network for Discovering Multivariate Laurent Polynomial Equations".
[[ArXiv](https://arxiv.org/abs/2312.10913)]

GINN-LP is a an end-to-end differentiable interpretable neural network that can recover mathematical expressions that take the form of multivariate Laurent polynomial (LP) equations. This is made possible by a new type of interpretable neural network block, named the power-term approximator (PTA). 

<img src="./assets/GINN-LP PTA Block.png" width=480, alt="GINN-LP Architecture">

LPs produce equations important in physics and real-world systems, such as the Coulomb's law, the gravitational potential energy, and the kinetic energy. 

- $E_f = \frac{1}{4\pi\epsilon}\frac{q_1q_2}{r^2}$ 
- $K = \frac{m}{2}(u^2 + v^2 + w^2)$ 
- $U = Gm_1m_2(\frac{1}{r_2} - \frac{1}{r_1})$

GINN-LP can discover these equations from data, without any prior knowledge of the underlying equations.



<img src="./assets/GINN-LP Architecture.png" width=1080, alt="GINN-LP Architecture">

## Discovering an equation using GINN-LP

Below, we outline the steps to discover the Coulomb's law equation using GINN-LP. The data used for this example is available in the data folder.

1. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
   
2. Run the code
    ```bash
   python run.py --data data/feynman_I_12_2 --format tsv
    ```
Here, the data file should be a csv or tsv with any number of feature columns and a target column. The target column should be named as "target".

## Project scripts: ENB + Agriculture (comparison vs `codes`)

This repo includes extra scripts under the project root to run **ENB** and **Agriculture** in a way that matches the preprocessing used in `codes/`.

- `run_enb.py`: single-target ENB runs (Y1 Heating Load, Y2 Cooling Load)
- `run_agric.py`: single-target Agriculture runs using the **same raw CSV inputs and merge logic** as `codes/Agriculture/mtr_ginn_agric_sym.py`
- `STRATEGY.md`: step-by-step commands and how to build the comparison tables
- `extract_comparison.py`: extract metrics from JSON outputs and print a markdown comparison table
- `compare_results.py`: quick ENB-only compare against a `codes` JSON

### Quickstart (ENB)

```bash
conda activate ginn
cd /raid/hussein/project/ginn-lp

# Y1 (Heating Load)
python run_enb.py --data run_ENB/data/ENB2012_Heating_Load.csv --format csv --num_epochs 20000 --start_ln_blocks 2 --growth_steps 3 --output_dir run_ENB/outputs

# Y2 (Cooling Load)
python run_enb.py --data run_ENB/data/ENB2012_Cooling_Load.csv --format csv --num_epochs 20000 --start_ln_blocks 2 --growth_steps 3 --output_dir run_ENB/outputs
```

### Quickstart (Agriculture)

```bash
conda activate ginn
cd /raid/hussein/project/ginn-lp

# Y1 (Sustainability_Score)
python run_agric.py --data_dir run_AGRIC/data --target Y1 --sample_fraction 0.1 --num_epochs 20000 --start_ln_blocks 3 --growth_steps 1 --output_dir run_AGRIC/outputs

# Y2 (Consumer_Trend_Index)
python run_agric.py --data_dir run_AGRIC/data --target Y2 --sample_fraction 0.1 --num_epochs 20000 --start_ln_blocks 3 --growth_steps 1 --output_dir run_AGRIC/outputs
```

For details (what is matched, where metrics are stored in JSON, and comparison commands), see `STRATEGY.md`.

## Scikit-learn compatible API

We also provide a scikit-learn compatible API for GINN-LP. The GINN-LP package should first be installed using pip.

``` 
pip install -U ./ginn-lp
 ```

After installing the package, you can use the scikit-learn compatible API to fit the model and generate output predictions. The default hyperparameters are listed below.Here, train_x should be a pandas dataframe object or numpy array containing input features. train_y should be a pandas series, dataframe or a numpy array containing target values.

```python
from ginnlp.ginnlp import GINNLP
import pandas as pd

train_df = pd.read_csv("data/feynman_I_12_2.csv")
train_x = train_df.drop(columns=["target"])
train_y = train_df["target"]

est = GINNLP(reg_change=0.5,
        start_ln_blocks=1,
        growth_steps=3,
        l1_reg=1e-4,
        l2_reg=1e-4,
        num_epochs=500,
        round_digits=3,
        train_iter=4)
est.fit(train_x, train_y)
```

Once the model is trained, the discovered mathematical equation can be viewed.

```python
print(est.recovered_eq)
```

Here, the recovered_eq variable contains a SymPy expression.

## Citation
If our work is useful, please consider citing:

```
  @article{ranasinghe2023ginn,
    title={GINN-LP: A Growing Interpretable Neural Network for Discovering Multivariate Laurent Polynomial Equations},
    author={Ranasinghe, Nisal and Senanayake, Damith and Seneviratne, Sachith and Premaratne, Malin and Halgamuge, Saman},
    journal={arXiv preprint arXiv:2312.10913},
    year={2023}
  }
```

