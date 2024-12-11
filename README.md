# How To Use Software

## Video Walkthrough
https://mediaspace.illinois.edu/media/t/1_yxmcmxzx

## Install Dependencies
1.  Create a virtual environment with 
```python -m venv myenv  # Replace "myenv" with your desired environment name```
2. Activate the virtual environment: 
	a. On Windows
```myenv\Scripts\activate.bat```
	b. On macOS/Linux:
```source myenv/bin/activate```
3. Install requirements:
```pip install -r requirements.txt```

## Add New Documents (Optional As Documents Already Provided)
1. Add a new text file in the 'documents' folder in a .txt format and populate it with the article's content (see current documents in folder for example)
2. 15 documents are already populated into folder, so you can skip over to 'Run Program' section of the README.md

## Run Program
1. ```python main.py```
2. All output files are added to 'output_files' folder for reference.
2. Navigate over to http://127.0.0.1:8050/ to see dashboard


# How Is Software Implemented

## Folders
### ./documents

Contains all articles that will be used in the program, in a txt format. Currently contains 15 articles that can be referenced in the last section of the README.md. If a user wants to add additional article, they just need to make a new text file, they add it here. This folder is used by the select_articles.py script.


### ./documents_filtered

Contains the documents from the documents folder that have passed the BM25 threshold filter. These documents will then be used by the find_common_phrases.py script.
 

### ./output_files

Contains all output reference files. Specifically it contains the common_words_filtered.txt file outputted by the find_common_phrases.py file, the document_similarity.txt file also from the find_common_phrases.py script, and the ndcg_score.txt file created by the select_articles.py script.


## Scripts
### display_dashboard.py

Dependencies: dash, pandas
1. Dashboard Setup: Dash is used to create an interactive web application. A dropdown menu lists all document pairs, allowing users to select one and view the results.
2. Interactivity: The @app.callback function updates the output (common phrases and cosine similarity) when a document pair is selected.
3. User Interface: Displays document pair details, cosine similarity, and a list of common phrases.
4. Ranked Output: The phrases are sorted by frequency in descending order.
5. Dropdown for Display Limit: The dropdown allows users to choose how many top phrases to display (e.g., Top 10, Top 20).
6. Dashboard Display: Displays the top-ranked phrases and their frequencies in a list.


### find_common_phrases.py

Dependencies: os, re, collections, sklearn
1. Data Processing: 
	a. ```load_documents```: Reads .txt files from the documents_filtered folder.
	b. ```find_common_phrases_across_documents```: Processes all document pairs, extracting common phrases and cosine similarity scores.
2. Single-Word Focus: 
	a. Updated extract_common_words to handle individual words only. 
	b. It uses Counter objects to find word frequencies and their intersection between documents.
3. Updated File Saving: 
	a. The save_common_words_to_file function writes each word and its frequency into a text file called common_words_filtered. 
	b. The document similarity is written to the document_similarity.txt file.
	

### main.py

Used as the main function which the user executes. It imports in the other three python scripts (display_dashboard, select_articles, find_common_phrases) and runs them in order. It also allows a user to pass custom folder paths for the program to use if they don't want to use the default options


### select_articles.py: 

Dependencies: os, math, collections, shutil
1. Initialization: The BM25 class initializes with a list of documents and calculates document frequencies for all unique terms.
2. IDF Calculation: The IDF component ensures that terms that occur in fewer documents are weighted higher.
3. BM25 Scoring:For each query term, the algorithm calculates the term frequency and combines it with IDF and normalization components to compute the BM25 score.
4. Ranking:The documents are scored and ranked based on their relevance to the query.
5. nDCG@10 Function: Calculates nDCG@10 and writes the calculated nDCG@10 score to a text file named ndcg_score.txt.

# Articles Prepopulated For Example
1. https://www.who.int/news-room/fact-sheets/detail/healthy-diet
2. https://www.nhs.uk/live-well/eat-well/how-to-eat-a-balanced-diet/eight-tips-for-healthy-eating/
3. https://www.mayoclinic.org/healthy-lifestyle/weight-loss/in-depth/weight-loss/art-20048466
4. https://www.healthline.com/nutrition/how-to-start-exercising#The-bottom-line
5. https://www.healthline.com/health/how-to-maintain-a-healthy-lifestyle#takeaway
6. https://www.forbes.com/health/weight-loss/workout-schedule/
7. https://nutritionsource.hsph.harvard.edu/healthy-eating-plate/
8. https://www.cnet.com/health/fitness/best-beginner-workouts/
9. https://kaynutrition.com/healthy-daily-habits/
10. https://www.lifehack.org/677367/powerful-daily-routine
11. https://www.ktpress.rw/2024/12/the-fish-revolution-how-rwanda-is-embracing-healthy-diets-for-a-healthier-future/
12. https://www.heartandstroke.ca/healthy-living/healthy-eating/healthy-eating-basics
13. https://www.nature.com/articles/s41582-024-01036-9
14. https://www.eatingwell.com/article/289245/7-day-heart-healthy-meal-plan-1200-calories/
15. https://www.heart.org/en/healthy-living/healthy-lifestyle/mental-health-and-wellbeing/meditation-to-boost-health-and-wellbeing
