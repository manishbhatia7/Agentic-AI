import uuid
import streamlit as st

from backend import (
    conversation_chain,
    get_session_history,
    parser
)

st.set_page_config(
    page_title="AI Review Analyzer",
    page_icon="🤖"
)

st.title("AI Product Review Analyzer")

# Session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Select task
task = st.radio(
    "Choose operation",
    ["Analyze Review", "Ask Question"],
    horizontal=True
)

if task == "Analyze Review":

    user_input = st.text_area("Enter product review")
    selected_task = "analyze"

else:

    user_input = st.text_area("Ask about previous reviews")
    selected_task = "question"


# Submit
if st.button("Submit"):

    if not user_input.strip():

        st.warning("Please enter text.")

    else:

        try:

            with st.spinner("Processing..."):

                result = conversation_chain.invoke(

                    {
                        "task": selected_task,
                        "input": user_input,
                        "format_instructions":
                            parser.get_format_instructions()
                    },

                    config={
                        "configurable": {
                            "session_id":
                                st.session_state.session_id
                        }
                    }

                )

            st.session_state.last_result = result

        except Exception as e:

            st.error(str(e))


# Display results
result = st.session_state.last_result

if result:

    analysis = result["analysis"]

    if analysis:

        st.subheader("Sentiment")

        if analysis["sentiment"] == "positive":
            st.success("Positive 😊")

        elif analysis["sentiment"] == "negative":
            st.error("Negative 😞")

        else:
            st.warning("Mixed 😐")

        st.subheader("Summary")
        st.write(analysis["summary"])

        st.subheader("Key Themes")
        st.write(analysis["key_themes"])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Pros")
            st.write(analysis["pros"])

        with col2:
            st.subheader("Cons")
            st.write(analysis["cons"])

        st.metric(
            "Confidence",
            f"{analysis['confidence']:.0%}"
        )

        st.subheader("AI Feedback")
        st.info(analysis["feedback"])

    else:

        st.subheader("AI Response")
        st.write(result["answer"])


# Chat history
st.divider()
st.subheader("Conversation History")

history = get_session_history(
    st.session_state.session_id
)

for message in history.messages:

    if message.type == "human":

        with st.chat_message("user"):
            st.write(message.content)

    elif message.type == "ai":

        with st.chat_message("assistant"):
            st.write(message.content)


# Clear history
if st.button("Clear Conversation"):

    history.clear()
    st.session_state.last_result = None
    st.rerun()