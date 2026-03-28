# ReadRadar
An ML-powered web app for literary recommendations, reviews, and themes.

### Setup
1. Download datasets from the [UCSD Book Graph GitHub](https://github.com/MengtingWan/goodreads). 
2. Modify path names as needed.
3. Run:
`pip install -r requirements.txt
streamlit run app.py`

### Planning
The intended final result of this project is to create a web app that performs the following:

Users use the web app to discover books they are looking for, receive personalized recommendations for new reads, and learn how to understand the discourse around literary works. When using the web app, users can either: 
1. Input themes, titles, or subjects. Receive books that match.
2. Input books they've liked OR their GoodReads account. Receive book recommendations. (Extension: Input ratings (1-5 stars) for works, receive what they're predicted to rate other works)
3. Input title of work or author. Find the work. Receive a controversy score for the work, pros and cons based on readers' reviews.

The intended data sources are as follows:
1. UCSD Book Graph
2. Open Library
3. WikiData

To do:
- Improve thematic search
- Make recommendations hybrid, instead of interaction-only
- Make the controversy feature more complex and accurate. Ideas: rating variance, proportion of extreme ratings (1-star + 5-star), disagreement index across sentiment in reviews, review-topic disagreement if possible
- Improve the current pros/cons extraction for reviews. This gets shallow real fast
- Bring Open Library into actual web data, specifically its subjects and descriptions.
- Add filters for search, e.g. minimum ratings count, rating threshold, publication year range, language, genre if available.
- Results are currently ranked by similarity alone, but make sure to consider: ratings count, average rating, relevance score, metadata completeness
- Could we add book detail pages for each book, or is this too much?



