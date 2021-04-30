# Product Return Modelling Assignment

*by David Burghoff
Student ID: 614500*

Final Assignment of the course "Business Analytics and Data Science" (BADS) at Humboldt Universität Berlin.

## Introduction

Online Shopping has been on a rise for some time now, even more so due to the current epidemic. While local stores had to close down, online shops could continue to sell their goods throughout the past year. The Digital Commerce 360 report estimates that ecommerce sales increased by 44% in 2020, compared to 2019 ([source](https://www.digitalcommerce360.com/2021/02/15/ecommerce-during-coronavirus-pandemic-in-charts/)). However, online shops have to face a cost that regular retailers do not have to calculate: the costs emerging from returns. Free shipping and return has been established as a standard in the industry, and while it certainly leads to many benefits for consumers, it also leads to some issues, both ecologically and economically.

Sometimes it is more cost-efficient for companies to simply destroy returned items rather than sending them back to their warehouse. Amazon was accused in 2018 of throwing away "Goods worth thousands of euros each day" in the German Television Show Frontal 21 ([source](https://ecommercenews.eu/amazon-destroys-returned-items-and-new-products/#:~:text=Ecommerce%20giant%20Amazon%20is%20massively,rid%20of%20their%20unsold%20goods.)). This is quite alarming if you also factor in that these goods have been shipped twice, only to end up in a landfill.

Economically, high return rates also have devastating effects on ecommerce platforms. Especially for low-cost items, where margins are slim, high return rates can lead to high costs due to shipping expenses. We are used to and expect companies to offer free shipping which mitigates the risk for buyers of buying goods we do not want and having to pay high shipping costs.

These factors rise the question if it is possible to predict if an item will be returned and to impose measures that aim to reduce return rates.

### Scope and business setting

>For this assignment, you are provided with real-world data by an online retailer. Your task is to identify the items that are likely to be returned. When a customer is about to purchase an item, which is likely to be returned, the shop is planning to show a warning message. Pre-tests suggest that the warning message leads approx. 50% of customers to cancel their purchase. In case of a return, the shop calculates with shipping-related costs of 3 EUR plus 10% of the item value in loss of resale value. Your task is to build a targeting model to balance potential sales and return risk to optimize shop revenue. The data you receive and within the test set is artificially balanced (1:1 ratio between (non-)returns). Since the real business >situation contains substantially more non-returns than returns, the misclassification costs include a correction factor of 5.

  

#### Relevant Information

-  **Display warning** if user is about to purchase an item with a high likelihood of return

-  **50%** of users subsequently cancel their purchase!

-  **Costs for Returns** (Shipping) **3 € + 10% of item value** in loss of resale value

- Data is artificially balanced : **1:1 ratio between (non-)returns**

  We will discuss the cost matrix and its implications in depth  at the start of the Modelling Notebook.

## Installing Required Packages

All required Packages are listed inside `requirements.txt`.

Simply navigate to this project's root folder, open a terminal and run:
````
pip install -r requirements.txt
````

Anaconda Alternative:
````
while read requirement; do conda install --yes $requirement; done < requirements.txt
````

## Structure of this project

I used a project structure similar to the [Cookie Cutter](https://github.com/cookiecutter/cookiecutter) Template.

|Folder |Contents |
|--|--|
|data | Contains data in `.csv` or `.pkl` format. Further divided into subfolders according to different stages of the data science process. |
|docs |Official Task Description.|
|notebooks |Contains all Notebooks. One Notebook for each stage. |
|references |Relavent Literature |
|results |Final results, contains `predicitions.csv` used for grading |
|src |Python scripts and utility functions. Each notebook (except Data Visualization) has its own script to avoid having to run all cells in the notebook|

  

## Table of Contents (Notebooks)

In the following notebooks we will explore the given dataset, clean it, impute missing values, engineer features, select features, train, evaluate and compare models and finally make discrete predictions on the unlabelled dataset. Each notebook handles specific tasks of the Data Science Process.

|Notebook |Contents |
|--|--|
|Data Cleaning |Cleaning the data, imputing missing values, setting variable types|
|Feature Engineering and Selection |Engineer new Features, Select relevant Features, prepare data for modelling|
|Data Visualisation |Visualisation of the Data, containing various Graphs and Tables. Can be read alongside the Feature Engineering and Selection notebook, since it has been created simultaneously in an iterative manner. |
|Modelling |Building, Training, Evaluating and Comparing of different models. Making discrete Predictions for the unlabelled Dataset |
