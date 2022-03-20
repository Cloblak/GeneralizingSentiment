# GeneralizingSentiment
## Project introduction
### Motivation
- Sentiment give context to content. This is a useful tool for a analysis everything from response to a product to predicting political election outcomes. 
- Data is readily available(reviews, news, ratings)
- Practical and useful information for various applications
### Goal
- find data that can be uniformly labeled, while varying in content type and size and build a model that can take large and small test strings and produce sentiment percentage of the string being either negative, neutral, or positive. 
## Get started
- clone the repo to your colab notebook or local machine 
[demo notebook sample](https://colab.research.google.com/drive/1QidqeviCLWYfWmC9bNoexkHr6sdexzG9#scrollTo=acRL0cxcaiTs)
- since we have large files in repo, install git-lfs before cloning
```
import os

import getpass

!curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

!sudo apt-get install git-lfs

!git lfs install
```
- install all requirement package
`!make install`
- go to the scripts directory
`cd scripts`
- train the model and save the model under models
`!python main.py`
- if you work on local machine, jump to the final step; if you work on colab, you need to sign up a pyngrok account to get the authtoken and replace the token below by yours
- run the demo to check the result(load pretrained1 under models, which is a pretrained XLNET with 80% accuracy)
```
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
!get_ipython().system_raw('./ngrok http 8501 &')
!ngrok authtoken 26WJyNXXUSY34VVdvJeXnkGDO3g_xX5cnoALV1vAwq6K12F8
```
- get the demo link(in a new cell)
```
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    'import sys, json; print("Execute the next cell and the go to the following URL: " +json.load(sys.stdin)["tunnels"][0]["public_url"])'
```
- run the demo(in a new cell)
`!streamlit run app.py`
## Test result
- use 70 thousands reviews to train XLNET for 2 epochs
![21cafebb02ac33443b84c419809d295](https://user-images.githubusercontent.com/87921304/159173122-f3a40eec-d56e-44b2-8ad6-0b410513e998.png)
