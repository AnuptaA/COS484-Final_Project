import openai
import requests
import pandas as pd
import os
from dotenv import load_dotenv
import time

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=api_key)

CATEGORIES = ["quantity","location","object","gender-number","gender","full"]

def make_caption_incorrect(caption, category):
    prompt = ""
    if category == CATEGORIES[0]:
        prompt = f"""Given this caption: "{caption}"
Alter the caption by changing the quantity of the subject of the sentence. Only output the altered caption and nothing else.
For example, for the sentence "Three women are having a picnic at the park.", \
the sentence might change to "Five women are having a picnic at the park." """
    elif category == CATEGORIES[1]:
        prompt = f"""Given this caption: "{caption}"
Alter the caption by changing the location of the sentence. Only output the altered caption and nothing else.
For example, for the sentence "Three women are having a picnic at the park.", \
the sentence might change to "Three women are having a picnic at the house." """
    elif category == CATEGORIES[2]:
        prompt = f"""Given this caption: "{caption}"
Alter the caption by changing the object of the sentence. Only output the altered caption and nothing else.
For example, for the sentence "Three women are having a picnic at the park.", \
the sentence might change to "Three women are having a basketball game at the park." """
    elif category == CATEGORIES[3]:
        prompt = f"""Given this caption: "{caption}"
Alter the caption by changing the gender and quantity of the subject of the sentence. Only output the altered caption and nothing else.
For example, for the sentence "Three women are having a picnic at the park.", \
the sentence might change to "Four men are having a picnic at the park." """
    elif category == CATEGORIES[4]:
        prompt = f"""Given this caption: "{caption}"
Alter the caption by changing the gender of the subject of the sentence. Only output the altered caption and nothing else.
For example, for the sentence "Three women are having a picnic at the park.", \
the sentence might change to "Three men are having a picnic at the park." """
    elif category == CATEGORIES[5]:
        prompt = f"""Given this caption: "{caption}"
Alter the caption by changing the gender and quantity of the subject, the location, and the object of the sentence. Only output the altered caption and nothing else.
For example, for the sentence "Three women are having a picnic at the park.", \
the sentence might change to "Four men are having a basketball game at the house." """
    else:
        print('Invalid Category: "' + category + '"')
        return False

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

# train data pre-processing
df = pd.read_csv('sunglass/clip_UND_scores_100_samples.csv')
inc = df[["imageURL","imageID","original"]]

# Iterate over each row in the DataFrame with a delay
for index, row in inc.iterrows():
    for category in CATEGORIES:
        if type(df.loc[index, ("und_" + category)]) is float:
            inc.loc[index, ("inc_" + category)] = "NA"
        else:
            inc.loc[index, ("inc_" + category)] = make_caption_incorrect(inc.loc[index, ("original")], category)
            print(inc.loc[index, ("inc_" + category)])
            time.sleep(1)  # Introduce a delay of 1 second

inc.to_csv('init_inc_100_samples.csv', index=False)