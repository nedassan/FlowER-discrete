# FlowER: Flow Matching for Electron Redistribution
_Joonyoung F. Joung*, Mun Hong Fong*, Nicholas Casetti, Jordan P. Liles, Ne S. Dassanayake, Connor W. Coley_

![Alt Text](FlowER.png)

FlowER uses flow matching to model chemical reaction as a process of electron redistribution, conceptually
aligns with arrow-pushing formalisms. It aims to capture the probabilistic nature of reactions with mass conservation
where multiple outcomes are reached through branching mechanistic networks evolving in time. 

## Environment Setup
### System requirements
**Ubuntu**: >= 16.04 <br>
**conda**: >= 4.0 <br>
**GPU**: at least 25GB Memory with CUDA >= 12.2

```bash
$ conda create -n flower python=3.10
$ conda activate flower
$ pip install -r requirements.txt
```

## Data/Model preparation
FlowER is trained on a combination of subset of USPTO-FULL (Dai et al.), RmechDB and PmechDB (Baldi et al.). <br>
To retrain/reproduce FlowER, download `data.zip` and `checkpoints.zip` folder from [this link](https://figshare.com/articles/dataset/FlowER_-_Mechanistic_datasets_and_model_checkpoint/28359407/2), and unzip them, and place under `FlowER/` <br>
The folder structure for the `data` folder is `data/{DATASET_NAME}/{train,val,test}.txt` and `checkpoints` folder is `checkpoints/{DATASET_NAME}/{EXPERIMENT_NAME}/model.{STEP}_{IDX}.pt`

## On how FlowER is structured
The workflow of FlowER revolves mainly around 2 files. `run.sh` and `settings.py`. <br> 
The main idea is to use comments `#` to turn on/off configurations when training/validating/inferencing FlowER. <br>
`run.sh` allows user to specify your data folder name, experiment name, gpu configuration and choose which scripts to run. <br>
`settings.py` allows user to specify different configurations for different workflows. 

## Training Pipeline
### 1. Train FlowER
Ensure that `data/` folder is populated accordingly and `settings.py` is pointing to the correct files.
```
    train_path = f"data/{DATA_NAME}/train.txt" 
    val_path = f"data/{DATA_NAME}/val.txt"
```
Check `run.sh` has `scripts/train.sh` uncommented. 
```bash
$ sh run.sh
```

### 2. Validate FlowER
You can validate FlowER on the validation set. Then, in `settings.py`, ensure these are uncommented
```
    # validation #
    do_validate = True
    steps2validate =  ["1050000", "1320000", "1500000", "930000", "1020000"]
```
`steps2validate` refers to the checkpoints that are selected based on train logs situated at the `/logs` folder. <br>
Check `run.sh` has `scripts/eval.sh` uncommented. 
```bash
$ sh run.sh
```

### 3. Test FlowER
You can validate FlowER on the test set. Then, in `settings.py`, specify your checkpoint at `MODEL_NAME` and ensure these are uncommented
```
    # inference #
    do_validate = False
```
Check `run.sh` has `scripts/eval.sh` uncommented. 
```bash
$ sh run.sh
```

### 4. Use FlowER for search
FlowER mainly uses beam search to seek for plausible mechanistic pathways. Users can input their smiles at `data/flower_dataset/beam.txt`
Ensure that in `settings.py`, `test_path` variable is pointing towards the corresponding file, and beam search configuration are uncommented.
```
    test_path = f"data/{DATA_NAME}/beam.txt"

    # beam-search #
    beam_size = 5
    nbest = 3
    max_depth = 15
    chunk_size = 50
```
Check `run.sh` has `scripts/search.sh` or `sh scripts/search_multiGPU.sh` uncommented. 
```bash
$ sh run.sh
```
Visualize your route at `examples/vis_network.ipynb`

## Citation
