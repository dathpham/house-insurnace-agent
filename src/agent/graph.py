"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict , Literal
from langgraph.runtime import Runtime
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

import pandas as pd
import numpy as np
from langchain_core.tools import tool
from langchain_openai.chat_models.base import ChatOpenAI
# from langchain.agents import AgentExecutor#, create_react_agent
# from langchain_openai import OpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage , SystemMessage
import os
from typing import Optional
from dotenv import load_dotenv

import os
from langchain_openai import OpenAIEmbeddings
from supabase.client import Client, create_client

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# query = "Insurance Plan for Government Employee"
# vector = embeddings.embed_query(query)
# data = supabase.rpc('match_documents', params={'query_embedding': vector}).execute()
# data.data[0]['content']

# load_dotenv()



SYSTEM_PROMPT_DATA_ENTRY = """You are a helpful insurance assistant. You can extract insurance profile information from text into a structured format."""
SYSTEM_PROMPT_REVIEW_INSURANCE  = """You are a helpful insurance assistant. You will review the retrieved insurance plan and provide advice on how to maximize home insurance based on the insurance profile."""

system_instruction_data_entry = SystemMessage(content=SYSTEM_PROMPT_DATA_ENTRY)
system_instruction_review_insurance = SystemMessage(content=SYSTEM_PROMPT_REVIEW_INSURANCE)

class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


# Insurance Profile Data Model
class InsuranceProfile(BaseModel):
    """Universal insurance profile structure"""
    member_name: str = Field(description="Plan member name")
    benefit_type: str = Field(description="Type of home insurnace or type of service")
    member_id: str = Field(description="Member ID number")
    policy_number: str = Field(description="Policy number")
    employer: Optional[str] = Field(description="Employer name if applicable", default=None)
    insurance_provider: str = Field(description="Insurance company name")
    plan_type: str = Field(description="Plan type ")
    benefit_period: str = Field(description="Coverage period")
    out_of_pocket_maximum: float = Field(description="Annual out-of-pocket maximum", ge=0)
    location: str = Field(description="Member location")


# # Analysis Schema
# schema = {"Target service": "Type of service (e.g.,  massage therapy, physiotherapy, chiropractic, acupuncture, therapist, osteopath",
#             "Coverage": "How much the insurance covers",
#           "Insurance Pays": "How much the insurance pays",
#           "Your Cost": "How much you pay",
#             "Total Potential Cost": "Total potential cost of therapy sessions",
#             "Annual Maximum": "Annual maximum coverage",
#             "Est. Session Cost": "Estimated cost of the session",
#             "Your Cost per Session": "Your cost per session",
#             "Estimated Sessions Available":"Estimated number of sessions available based on coverage"   
#           }


# Pydantic
class AnalyzedProfile(BaseModel):
    """Structured analysis of insurance profile"""
    better_plan_details: str = Field(description="Details of the better plan if available")
    coverage: str = Field(description="How much the insurance covers")
    insurance_pays: str = Field(description="How much the insurance pays")
    your_cost: str = Field(description="How much you pay")
    annual_maximum: str = Field(description="Annual maximum coverage")


@tool
def query_doc(query_string: str) -> str:
   """Query the insurance documents in the database."""
   
   vector = embeddings.embed_query(query_string)
   data = supabase.rpc('match_documents', params={'query_embedding': vector}).execute()


   return data.data[0]['content']



# @dataclass
# class State:
#     """Input state for the agent.

#     Defines the initial structure of incoming data.
#     See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
#     """

#     changeme: str = "example"

class AgentState(MessagesState):
    # Final structured response from the agent
    final_response: InsuranceProfile


# async def data_entry(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
#     """Process input and returns output.

#     Can use runtime context to alter behavior.
#     """

#     model = ChatOpenAI(model="gpt-5", timeout=3600)
#     # tools={"type": "web_search"}
#     # client_with_tools = client.bind_tools([tools])

#     # messages = [("system", "Test: How can I maximize mental health benefits with insurance?"),
#     #         ("human", "Test: extract insurance profile ")]
#     # response = client_with_tools.invoke(messages)

#     return {
#         "changeme": "output from data_entry. "
#         f"Configured with {runtime.context.get('my_configurable_param')}"
#     }




model = ChatOpenAI(model="gpt-5", disable_streaming= True)
#model = init_chat_model("gpt-5",model_provider="openai",temperature=0)
# search_model = ChatOpenAI(model="gpt-5", timeout=3600)
tools = [InsuranceProfile]
model_with_response_tool = model.bind_tools(tools)

web_search_tool = {"type": "web_search"}
        #            "user_location": {
        #     "type": "approximate",
        #     "country": "US",
        #     "city": "Seattle",
        #     "region": "Seattle",
        # }}

model_with_rag_search= model.bind_tools([query_doc])

# Define the function that calls the model
def call_data_entry_model(state: AgentState):
    # state["messages"].append(system_instruction_data_entry)
    response = model_with_response_tool.invoke(state["messages"])
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def call_rag_model(state: AgentState):

    # state["messages"].append(system_instruction_review_insurance)
    # response = model_with_web_search.invoke([state["messages"]])
    response = model_with_rag_search.with_structured_output(AnalyzedProfile).invoke("Compare the retrieved insurance plan and recommend the best insurance that I can get?" + state["messages"][-2].content)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function that responds to the user
def respond(state: AgentState):
    # Construct the final answer from the arguments of the last tool call
    last_tool_call = state["messages"][-1].tool_calls[0]
    response = InsuranceProfile(**last_tool_call["args"])
    # Since we're using tool calling to return structured output,
    # we need to add  a tool message corresponding to the WeatherResponse tool call,
    # This is due to LLM providers' requirement that AI messages with tool calls
    # need to be followed by a tool message for each tool call
    tool_message = {
        "type": "tool",
        "content": "Here is your structured response",
        "tool_call_id": last_tool_call["id"],
    }
    # We return the final answer
    state["messages"].append(HumanMessage(content=state["messages"][-2].content))

    return {"final_response": response, "messages": [tool_message]}

def rag_agent_respond(state: AgentState):
    # We call the model with structured output in order to return the same format to the user every time
    # state['messages'][-2] is the last ToolMessage in the convo, which we convert to a HumanMessage for the model to use
    # We could also pass the entire chat history, but this saves tokens since all we care to structure is the output of the tool
    # review_response = model_with_web_search.invoke(
    #     [SystemMessage(content=SYSTEM_PROMPT_REVIEW_INSURE),HumanMessage(content=state["messages"][-2].content)]
    # )
    state["messages"].append(system_instruction_review_insurance)
    state["messages"].append(HumanMessage(content="How can I maximize mental health benefits with insurance?"))

    review_response = model_with_rag_search.invoke(state["messages"])
    # We return the final answer
    return {"insurance_response": review_response}


# Define the graph
# graph = (
#     StateGraph(State, context_schema=Context)
#     .add_node(data_entry)
#     .add_edge("__start__", "data_entry")
#     .compile(name="New Graph")
# )

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("data_entry_agent", call_data_entry_model)
workflow.add_node("rag_agent", call_rag_model)
workflow.add_node("structured_profile_response", respond)
workflow.add_node("insurance_review_response", rag_agent_respond)
# workflow.add_node("tools", ToolNode(tools))

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("data_entry_agent")

# # We now add a conditional edge
# workflow.add_conditional_edges(
#     "agent",
#     should_continue,
#     {
#         "continue": "tools",
#         "respond": "respond",
#     },
# )

# workflow.add_edge("tools", "data_entry_agent")
workflow.add_edge("data_entry_agent", "structured_profile_response")
workflow.add_edge("structured_profile_response", "rag_agent")
workflow.add_edge("rag_agent", "insurance_review_response")

workflow.add_edge("insurance_review_response", END)
graph = workflow.compile()


# graph = (
#     StateGraph(State, context_schema=Context)
#     .add_node(call_model)
#     .add_edge("__start__", "call_model")
#     .compile(name="New Graph")
# )
