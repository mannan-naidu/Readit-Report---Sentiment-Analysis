# The Readit Report 📈

Readit is an interactive web application built with Streamlit that allows you to scrape and analyze discussions from Reddit. It goes beyond simple sentiment analysis by incorporating conversational context, sarcasm detection, and a state-of-the-art AI model to provide nuanced insights into public opinion.



Features

 Dynamic Topic Search**: Search for any topic across all of Reddit or within a specific subreddit.
 Multi-Post Analysis**: Select multiple posts with checkboxes to analyze comments from all of them in a single batch.
 Advanced Sentiment Analysis**: Utilizes a pre-trained **RoBERTa model** from Hugging Face for highly accurate, context-aware sentiment classification (Positive, Negative, Neutral).
 Contextual Analysis**: Fetches and displays the **parent comment** for each reply to show the conversational context.
 Sarcasm Detection**: Automatically identifies comments ending with `/s`, flags them as sarcastic, and reverses the model's literal sentiment analysis to reveal the true intended meaning.
 Interactive Dashboard**: View results with summary metrics, a bar chart, and a detailed comment viewer.
 Comment Filtering**: Dynamically filter the detailed comment view by sentiment classification (e.g., show only "Negative" comments).

-----

Tech Stack

 Backend: Python
 Frontend: Streamlit
 Web Scraping: PRAW (Python Reddit API Wrapper)
 NLP / AI: Hugging Face `transformers` (using `cardiffnlp/twitter-roberta-base-sentiment-latest`), PyTorch
 Data Handling: Pandas

