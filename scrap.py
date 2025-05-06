#Load Libraries
from google_play_scraper import app
from google_play_scraper import Sort, reviews
import pandas as pd
import numpy as np

#Scrape Play Store Reviews
result, continuation_token = reviews(
    'com.codeway.aitutor',
    lang='en',
    country='us',
    sort=Sort.MOST_RELEVANT,
    count=50000,
    filter_score_with=None
)

#Format
df = pd.DataFrame(np.array(result),columns=['review'])
df = df.join(pd.DataFrame(df.pop('review').tolist()))

#Retrive Data
comment = df['content']
print(comment.head())

#Save Dataset
comment.to_csv('raw_dataset.csv',index=False)

