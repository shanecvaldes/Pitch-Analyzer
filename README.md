# Pitch Analyzer
Analyze each pitch from Baseball Savant

## Background
The purpose of this project is to create a web scraper that scraps all of the pitching currently contained on [Baseball Savant](https://baseballsavant.mlb.com/). I want to use this data to run some ML algorithms and create data visualizations in a web app. For now, this code only scraps the pitches, and if you want, summarize/parse some statistics into their own files and run clustering/visualizations for the data.

## How to run:

### Scraper
Please download all of the requirements contained in the requirements.txt file. After that, please run the pitchers_getter.py file while in the stats directory. This file will scrap all the current pitches in Baseball Savant, which does take a long time. If you so desire, you can then run the summarize_pitchers.py and parse_pitches.py files, this will create more folders containing all of the data, most of which may be redundant. 

### Clustering
If you would like to try the data clustering, you can run the cluster_pitches.py file, but ONLY after running the summarize_pitchers.py file. The clustering is dependent on the summarized pitches folder. This process also takes a while, as many interactive html files are created from this process. 

## Disclaimer
If there are any comments, questions, or concerns in relation to this repository, please email me through the email on my profile. This project is not meant to harm Baseball Savant or any of its services. If you clone this repository for personal use or edit any of its functionality, please keep in mind the terms of services and robot.txt guidelines. At the time of writing this, there are no real guidelines that I can see in the robots.txt file.