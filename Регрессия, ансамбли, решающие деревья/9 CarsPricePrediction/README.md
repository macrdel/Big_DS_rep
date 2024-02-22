## The wheel of data

### Brief introduction​

​Comapany owns a considerable amount of corporate cars and it wants to be able to calculate their total value for accounting purposes. To do that the company wants to: 

1) have a model, which will predict the price of a car (single at a time) based on its input parameters; 
2) Estimate cars economical depreciation function (how do they lose value over time).​

It was decided to test this approach in Kazakhstan division at the first place.

### Problem description​

You are provided with already collected data in a format of csv file. In the first row there are column names; all the other rows contain data, each row corresponding to one publication on [Kolesa.kz](https://kolesa.kz/) website. The columns are separated with semicolon.​

​You are asked to do the following:​

- Build machine learning model, which will predict price of a single car based on its input parameters. You should choose parameter set yourself but make sure that the company have that data by asking the company about it;​

- Analyze how the value of a car changes with time. For different manufacturers (or even models of one manufacturer) those changes paces may be different. You may want to divide them into clusters with common behavior and illustrate some of those clusters properties.​

​You should provide a company with the results of your work in form of a presentation in PowerPoint or a Jupyter-Notebook. You should illustrate how you tested different models, which of them you chose (and why), what metrics you used, what parameters you decided to use and explain what conclusions you made about the depreciation.​

​​### Using models

- Baseline LinReg
- Baseline CatBoost
- Baseline LightGBM
- LightGMB + Optuna (best)