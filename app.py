import streamlit as st
import pickle
import pandas as pd

# ---- Load Precomputed Artifacts ----
book_names = pickle.load(open('notebook/artifacts/book_names.pkl', 'rb'))
book_pivot = pickle.load(open('notebook/artifacts/book_pivot.pkl', 'rb'))
final_rating = pickle.load(open('notebook/artifacts/final_rating.pkl', 'rb'))
model = pickle.load(open('notebook/artifacts/model.pkl', 'rb'))

# ---- Streamlit UI Setup ----
st.set_page_config(page_title="Book Recommender", layout="wide")
st.title("📚 Book Recommendation System")
st.markdown("Get similar book suggestions based on your favorite one!")

# ---- Book Selector ----
selected_book = st.selectbox("Choose a book", book_names)

# ---- Recommendation Function ----
def recommend(book_name):
    index = book_pivot.index.get_loc(book_name)
    distances, suggestions = model.kneighbors(
        book_pivot.iloc[index, :].values.reshape(1, -1), n_neighbors=6
    )

    recommended = []
    for i in range(1, len(suggestions[0])):
        book_title = book_pivot.index[suggestions[0][i]]
        data = final_rating[final_rating['title'] == book_title].drop_duplicates('title')
        if not data.empty:
            recommended.append({
                'title': book_title,
                'author': data['author'].values[0],
                'image': data['image_url'].values[0]
            })
    return recommended

# ---- Recommend Button ----
if st.button("🔍 Recommend"):
    recommendations = recommend(selected_book)

    st.subheader("📖 Recommended Books:")
    if not recommendations:
        st.write("No recommendations found for this book.")
    else:
        st.write(f"Based on your choice of **{selected_book}**, we recommend the following books:")
    
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        if idx < len(recommendations):
            col.image(recommendations[idx]['image'], use_column_width=True)
            col.markdown(f"**{recommendations[idx]['title']}**")
            col.markdown(f"*{recommendations[idx]['author']}*")
