
import streamlit as st
import uuid
,


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# 2. SESSION STATE
# --------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# 3. PAGE HEADER
# --------------------------------------------------

st.title("📚 AI Learning Assistant")
st.write("Learn any topic with your AI Tutor")


# --------------------------------------------------
# 4. SELECT LEARNING MODE
# --------------------------------------------------

mode = st.selectbox(
    "Choose Learning Mode",
    ["Explain", "Quiz", "Ask a Question"]
)


# --------------------------------------------------
# 5. EXPLAIN MODE
# --------------------------------------------------

if mode == "Explain":

    topic = st.text_input("Enter a topic you want to learn")

    if st.button("Submit", key="explain_submit"):

        if not topic.strip():
            st.warning("Please enter a topic")

        else:
            with st.spinner("AI is preparing your explanation..."):

                response = explanation_chain.invoke({
                    "topic": topic,
                    "format_instructions": parser.get_format_instructions()
                })

            st.subheader(response.topic)

            st.markdown("### Summary")
            st.write(response.summary)

            st.markdown("### Key Points")

            for point in response.key_points:
                st.markdown("- " + point)

            st.markdown("### Examples")

            for example in response.examples:
                st.markdown("- " + example)


# --------------------------------------------------
# 6. QUIZ MODE
# --------------------------------------------------

elif mode == "Quiz":

    topic = st.text_input("Enter a topic you want to learn")

    if st.button("Submit", key="quiz_submit"):

        if not topic.strip():
            st.warning("Please enter a topic")

        else:
            with st.spinner("AI is preparing your quiz..."):

                response = quiz_chain.invoke({
                    "topic": topic
                })

            st.markdown("### Your Quiz")
            st.markdown(response)


# --------------------------------------------------
# 7. ASK A QUESTION MODE
# --------------------------------------------------

elif mode == "Ask a Question":

    st.subheader("💬 Chat with AI Tutor")

    # Display previous chat messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    question = st.chat_input(
        "Ask your AI tutor a question..."
    )

    if question:

        # Save user question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # Display user question
        with st.chat_message("user"):
            st.write(question)

        # Generate AI response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = conversion_chain.invoke(
                    {"question": question},
                    config={
                        "configurable": {
                            "session_id": st.session_state.session_id
                        }
                    }
                )

            st.write(response)

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
