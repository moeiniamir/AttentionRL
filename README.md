# Prerequisites
Clone and go into the repository:
```bash
git clone git@github.com:moeiniamir/AttentionRL.git
cd AttentionRL
```
Create Conda environment from `environment.yml` file:
```bash
conda env create -f environment.yml
```
Activate the environment:
```bash
conda activate rl39
```
# Usage
Agent1 which looks through the image is at `v2/agent1.ipynb`.

## Training
After setting the training configuration in the notebook, start the training:
```bash
cd v2
../run.sh agent1
```

## Testing
You can load the saved policy and test it in the notebook.