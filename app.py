import streamlit as st
import praw
import pandas as pd
from transformers import pipeline

# --- Page Configuration and Title ---
st.set_page_config(page_title="Readit Report", layout="wide")
st.title("The Readit Report 📈")

# --- Reddit API Credentials ---
try:
    reddit = praw.Reddit(
        client_id=st.secrets["CLIENT_ID"],
        client_secret=st.secrets["CLIENT_SECRET"],
        user_agent=st.secrets["USER_AGENT"],
        read_only=True,
    )
except Exception:
    st.error("Reddit credentials not found. Please add them to your Streamlit secrets.")
    st.stop()

# --- Load Advanced Model (cached for performance) ---
@st.cache_resource
def load_model():
    """Loads the pre-trained sentiment analysis pipeline."""
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

sentiment_pipeline = load_model()


# --- Helper Functions ---

@st.cache_data(show_spinner="Searching for posts...")
def search_posts(topic, subreddit):
    """Searches Reddit for posts and returns a list of them."""
    if not subreddit:
        subreddit = "all"
    subreddit_instance = reddit.subreddit(subreddit)
    search_results = subreddit_instance.search(topic, sort='relevance', limit=15)
    
    posts = []
    for post in search_results:
        posts.append({
            'id': post.id,
            'title': post.title,
            'subreddit': post.subreddit.display_name,
            'num_comments': post.num_comments,
            'url': post.url
        })
    return posts

@st.cache_data(show_spinner="Analyzing comments with advanced model... this may take a moment.")
def analyze_comments(post_ids):
    """Scrapes and analyzes comments using the Hugging Face model."""
    comments_to_analyze = []
    structured_comments = []

    for post_id in post_ids:
        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=50) 
        for comment in submission.comments.list():
            comment_body = comment.body.strip()
            parent_text = ""
            parent = comment.parent()
            
            if isinstance(parent, praw.models.Comment):
                parent_text = parent.body
            elif isinstance(parent, praw.models.Submission):
                parent_text = parent.title
            
            comments_to_analyze.append(comment_body)
            structured_comments.append({
                'text': comment_body,
                'parent_text': parent_text
            })

    if not comments_to_analyze:
        return None

    results = sentiment_pipeline(comments_to_analyze, truncation=True, max_length=512)
    
    analyzed_comments = []
    for i, comment_data in enumerate(structured_comments):
        comment_body = comment_data['text']
        
        if comment_body.lower().endswith("/s"):
            sentiment_category = "Sarcastic"
            clean_text = comment_body[:-2].strip()
            
            original_result = sentiment_pipeline(clean_text, truncation=True, max_length=512)[0]
            original_sentiment = original_result['label'].capitalize()
            
            if original_sentiment == "Positive":
                reversed_sentiment = "Negative"
            elif original_sentiment == "Negative":
                reversed_sentiment = "Positive"
            else: # Neutral
                reversed_sentiment = "Negative"
            
            display_label = f"Sarcastic (Reversed to {reversed_sentiment})"
            text_to_display = clean_text
        else:
            prediction = results[i]
            sentiment_category = prediction['label'].capitalize()
            display_label = sentiment_category
            text_to_display = comment_body
        
        analyzed_comments.append({
            'text': text_to_display, 
            'sentiment_category': sentiment_category,
            'display_label': display_label,
            'parent_text': comment_data['parent_text']
        })

    sentiments = [comment['sentiment_category'] for comment in analyzed_comments]
    
    return {
        'counts': {
            'total_comments': len(analyzed_comments),
            'positive': sentiments.count("Positive"),
            'negative': sentiments.count("Negative"),
            'neutral': sentiments.count("Neutral"),
            'sarcastic': sentiments.count("Sarcastic")
        },
        'comments': analyzed_comments
    }

# --- Main App UI ---

col1, col2 = st.columns(2)
with col1:
    topic = st.text_input("Enter a topic to search for:", "AI ethics")
with col2:
    subreddit = st.text_input("Enter a subreddit (optional):", "technology")

if st.button("Search for Posts", type="primary"):
    st.session_state.posts = search_posts(topic, subreddit)
    if 'analysis_results' in st.session_state:
        del st.session_state['analysis_results']

if 'posts' in st.session_state:
    posts = st.session_state.posts
    st.markdown("---")
    st.subheader("Search Results")

    if not posts:
        st.warning("No posts found for this topic.")
    else:
        selected_posts = []
        if st.button("Analyze Selected Posts", type="primary"):
            for post in posts:
                if st.session_state.get(f"cb_{post['id']}", False):
                    selected_posts.append(post['id'])
            
            if selected_posts:
                st.session_state.analysis_results = analyze_comments(selected_posts)
            else:
                st.warning("Please select at least one post to analyze.")
        
        for post in posts:
            st.checkbox(
                f"**{post['title']}** (r/{post['subreddit']} • {post['num_comments']} comments)", 
                key=f"cb_{post['id']}"
            )

if 'analysis_results' in st.session_state:
    results = st.session_state.analysis_results
    st.markdown("---")
    st.subheader("Analysis Results")
    
    if results is None:
        st.error("Could not retrieve or analyze comments.")
    else:
        counts = results['counts']
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Comments", counts['total_comments'])
        col2.metric("🟢 Positive", counts['positive'])
        col3.metric("🔴 Negative", counts['negative'])
        col4.metric("⚪ Neutral", counts['neutral'])
        col5.metric("🎭 Sarcastic", counts['sarcastic'])

        chart_data = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral", "Sarcastic"],
            "Count": [counts['positive'], counts['negative'], counts['neutral'], counts['sarcastic']]
        })
        st.bar_chart(chart_data, x="Sentiment", y="Count")
        
        # NEW: Filter selection box
        st.markdown("---")
        filter_choice = st.selectbox(
            "Filter comments by classification:",
            ("All", "Positive", "Negative", "Neutral", "Sarcastic")
        )
        
        with st.expander("View Analyzed Comments", expanded=True):
            all_comments = results['comments']
            
            # NEW: Filtering logic
            if filter_choice == "All":
                comments_to_display = all_comments
            else:
                comments_to_display = [c for c in all_comments if c['sentiment_category'] == filter_choice]
            
            st.write(f"**Showing {len(comments_to_display)} of {len(all_comments)} comments.**")
            
            if comments_to_display:
                sentiment_map = {
                    "Positive": ("🟢", "green"),
                    "Negative": ("🔴", "red"),
                    "Neutral": ("⚪", "gray"),
                    "Sarcastic": ("🎭", "orange")
                }
                for i, comment in enumerate(comments_to_display):
                    if comment['parent_text']:
                        st.markdown(f"> **In reply to:** *{comment['parent_text'][:200]}...*")

                    primary_sentiment = comment['sentiment_category']
                    display_label = comment['display_label']
                    emoji, color = sentiment_map[primary_sentiment]
                    
                    st.markdown(f"**Comment {i+1}** | Sentiment: {emoji} <span style='color:{color};'>{display_label}</span>", unsafe_allow_html=True)
                    st.info(comment['text'])
            else:
                st.write("No comments found for this filter.")