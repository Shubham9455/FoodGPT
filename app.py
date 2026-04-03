import streamlit as st
from rag_engine import ask

st.set_page_config(page_title="FoodGPT", page_icon="🍽️")
st.title("🍽️ FoodGPT")
st.caption("AI-powered restaurant recommendations powered by Zomato data")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("What are you craving? (e.g. 'cheap North Indian in Koramangala')")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Finding the best options..."):
            answer, results = ask(query)
        st.subheader("🍽️ Top Recommendations")

        for r in results:
            st.markdown(f"""
        **{r['name']}**  
        📍 {r['location']}  
        🍜 {r['cuisines']}  
        ⭐ {r['rate']}
        ---
        """)
    
    st.session_state.history.append({"role": "assistant", "content": answer})