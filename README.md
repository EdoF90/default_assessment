# Credit Risk Models

This library simulates credit risk environments. The are supported by two different articles that we recommend citing in case you use the library.

## Citing Us


```Bibtex
@article{fadda2024,
	author = {Edoardo Fadda, Elisa Luciano and Patrizia Semeraro},
	journal = {Submitted},
	publisher={...},
	pages = {...},
	title = {Machine Learning techniques in joint default assessment.},
	volume = {..},
	number = {..},
  	doi = {....},
	year = {2024}}

```

## Code Structure

```bash
|____cfgs
| |____model.py
| |____sim_settings.json

|____data

|____logs

|____models
| |______init__.py
| |____exchangeableLogitModel.py
| |____generateDatasets.py
| |____logitSimulator.py

|____results

|____utilities
| |____ato_Params.json
| |____instance_Params.json
| |____sampler_Params.json

```
The data folder is intended to contain the real datasets that you can download online.
Each main file outlines the primary steps to replicate the experiments and create new ones.
