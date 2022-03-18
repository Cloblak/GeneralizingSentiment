# GeneralizingSentiment
## Project introduction
### Motivation
-Sentiment give context to content. This is a useful tool for a analysis everything from response to a product to predicting political election outcomes. 
-Data is readily available(reviews, news, ratings)
-Practical and useful information for various applications
### Goal
-find data that can be uniformly labeled, while varying in content type and size and build a model that can take large and small test strings and produce sentiment percentage of the string being either negative, neutral, or positive. 
## Get started
-clone the repo to your colab notebook or local machine 
[demo notebook sample](https://colab.research.google.com/drive/1QidqeviCLWYfWmC9bNoexkHr6sdexzG9#scrollTo=acRL0cxcaiTs)
-since we have large files in repo, run the following code before cloning
'''
import os

import getpass

!curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

!sudo apt-get install git-lfs

!git lfs install
'''
-install all requirement package
'make install'
-train the model and save the model under models
'python main.py'
