from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
history=InMemoryChatMessageHistory()
history.add_message(HumanMessage(content="How many casual leaves do employees get?"))
history.add_message(AIMessage(content="Employees get 8 casual leaves annually."))
history.add_message(HumanMessage(content="Can I carry them forward?"))
for hist in history.messages:
    print("role ",hist.type)
    print("content ",hist.content)
print("Length of Messages",len(history.messages))    