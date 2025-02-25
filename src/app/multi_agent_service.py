from typing import Literal, Any
from typing import Annotated

from colorama import Fore, Style
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from langgraph_checkpoint_cosmosdb import CosmosDBSaver
import datetime
import azure_cosmos_db
from azure_cosmos_db import DATABASE_NAME, CHECKPOINT_CONTAINER, PRODUCTS_CONTAINER, userdata_container, patch_active_agent, \
    update_userdata_container, get_cosmos_client
from azure_open_ai import model, generate_embedding


########
# Tools ############################################################################################################
########

# Perform a vector search on the Cosmos DB container.
# Note: in this case we are using Cosmos DB's native SDKs to perform the vector search instead of
# using the langchain_cosmosdb vector store integration, because the latter's current implementation
# requires any additional fields that you want to return, other than the embedding field, to be defined
# within a metadata json field. This can mean duplicating fields that we need as part of the search,
# but that might already be present in the transactional store. As such, the langchain integration assumes
# that the vector store is being used ONLY as a vector store, and not also as a transactional store.
def vector_search(vectors, similarity_score=0.02, num_results=2):
    # Execute the query
    database = azure_cosmos_db.client.get_database_client(DATABASE_NAME)
    container = database.get_container_client(PRODUCTS_CONTAINER)
    results = container.query_items(
        query='''
        SELECT TOP @num_results c.product_id, c.product_name, c.category, c.price, c.product_description, VectorDistance(c.embedding, @embedding) as SimilarityScore 
        FROM c
        WHERE VectorDistance(c.embedding,@embedding) > @similarity_score
        ORDER BY VectorDistance(c.embedding,@embedding)
        ''',
        parameters=[
            {"name": "@embedding", "value": vectors},
            {"name": "@num_results", "value": num_results},
            {"name": "@similarity_score", "value": similarity_score}
        ],
        enable_cross_partition_query=True, populate_query_metrics=True)
    print("Executed vector search in Azure Cosmos DB... \n")
    results = list(results)
    # Extract the necessary information from the results
    formatted_results = []
    for result in results:
        score = result.pop('SimilarityScore')
        formatted_result = {
            'SimilarityScore': score,
            'document': result
        }
        formatted_results.append(formatted_result)
    return formatted_results

def transfer_to_agent_message(agent):
    print(Fore.LIGHTMAGENTA_EX + f"transfer_to_{agent}..." + Style.RESET_ALL)

def create_agent_transfer(*, agent_name: str):
    """Create a tool that can do agent transfer"""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def transfer_to_agent(
            state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Ask another agent for help."""
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        transfer_to_agent_message(agent_name)
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={"messages": state["messages"] + [tool_message]},
        )

    return transfer_to_agent
@tool
def refund_item(user_id, product_id):
    """Initiate a refund based on the user ID and product ID.
    Takes as input arguments in the format '{"user_id":1,"product_id":3}'
    """
    try:
        database = azure_cosmos_db.client.get_database_client(azure_cosmos_db.DATABASE_NAME)
        container = database.get_container_client(azure_cosmos_db.PURCHASE_HISTORY_CONTAINER)
        query = "SELECT c.amount FROM c WHERE c.user_id=@user_id AND c.product_id=@product_id"
        parameters = [
            {"name": "@user_id", "value": int(user_id)},
            {"name": "@product_id", "value": int(product_id)}
        ]
        items = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
        if items:
            amount = items[0]['amount']
            # Refund the amount to the user
            refund_message = f"Refunding ${amount} to user ID {user_id} for product ID {product_id}."
            return refund_message
        else:
            refund_message = f"No purchase found for user ID {user_id} and product ID {product_id}. Refund initiated."
            return refund_message
    except Exception as e:
        print(f"An error occurred during refund: {e}")


@tool
def notify_customer(user_id, method):
    """Notify a customer by their preferred method of either phone or email.
    Takes as input arguments in the format '{"user_id":1,"method":"email"}'"""
    try:
        database = azure_cosmos_db.client.get_database_client(azure_cosmos_db.DATABASE_NAME)
        container = database.get_container_client(azure_cosmos_db.USERS_CONTAINER)
        query = "SELECT c.email, c.phone FROM c WHERE c.user_id=@user_id"
        parameters = [{"name": "@user_id", "value": int(user_id)}]
        items = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
        if items:
            email, phone = items[0]['email'], items[0]['phone']
            if method == "email" and email:
                print(f"Emailed customer {email} a notification.")
            elif method == "phone" and phone:
                print(f"Texted customer {phone} a notification.")
            else:
                print(f"No {method} contact available for user ID {user_id}.")
        else:
            print(f"User ID {user_id} not found.")
    except Exception as e:
        print(f"An error occurred during notification: {e}")


@tool
def order_item(user_id, product_id):
    """Place an order for a product based on the user ID and product ID.
    Takes as input arguments in the format '{"user_id":1,"product_id":2}'"""
    try:
        date_of_purchase = datetime.datetime.now().strftime("%d/%m/%Y")

        database = azure_cosmos_db.client.get_database_client(azure_cosmos_db.DATABASE_NAME)
        container = database.get_container_client(azure_cosmos_db.PRODUCTS_CONTAINER)
        query = "SELECT c.product_id, c.product_name, c.price, c.category FROM c WHERE c.product_id=@product_id"
        parameters = [{"name": "@product_id", "value": int(product_id)}]
        items = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
        if items:
            product = items[0]
            product_id, product_name, price, category = product['product_id'], product['product_name'], product[
                'price'], product['category']
            print(f"Ordering product {product_name} for user ID {user_id}. The price is {price}.")
            # Add the purchase to the database
            azure_cosmos_db.add_purchase(int(user_id), date_of_purchase, product_id, price, product_name,
                                         category)
            order_item_message = f"Order placed for product {product_name} for user ID {user_id}. product ID: {product_id}."
            return order_item_message
        else:
            order_item_message = f"Product {product_id} not found."
            return order_item_message
    except Exception as e:
        print(f"An error occurred during order placement: {e}")


@tool
def product_information(user_prompt: str) -> list[dict[str, Any]]:
    """Provide information about a product based on the user prompt.
    Takes as input the user prompt as a string."""
    # Perform a vector search on the Cosmos DB container and return results to the agent
    vectors = generate_embedding(user_prompt)
    search_results = vector_search(vectors)
    return search_results

########
# Agents ############################################################################################################
########

triage_agent_tools = [
    create_agent_transfer(agent_name="product_agent"),
    create_agent_transfer(agent_name="sales_agent"),
    create_agent_transfer(agent_name="refunds_agent"),
]

local_interactive_mode = False

triage_agent = create_react_agent(
    model,
    triage_agent_tools,
    state_modifier=(
        "You are to triage a users request, and call a tool to transfer to the right intent."
        "Otherwise, once you are ready to transfer to the right intent, call the tool to transfer to the right intent."
        "You dont need to know specifics, just the topic of the request."
        "If the user asks for product information, transfer to product_agent."
        "If the user request is about making an order or purchasing an product, transfer to the sales_agent."
        "If the user request is about getting a refund on an product or returning a product, transfer to the refunds_agent."
        "When you need more information to triage the request to an agent, ask a direct question without explaining why you're asking it."
        "Do not share your thought process with the user! Do not make unreasonable assumptions on behalf of user."
    ),
)

product_agent_tools = [
    product_information,
    create_agent_transfer(agent_name="sales_agent"),
    create_agent_transfer(agent_name="refunds_agent"),
    create_agent_transfer(agent_name="triage_agent"),
]
product_agent = create_react_agent(
    model,
    product_agent_tools,
    state_modifier=(
        "You are a product agent that provides information about products in the database."
        "Always call the product_information tool when a user asks about products before transferring to another agent."
        "If the user asks for more information about any product, provide it by passing their question as input to the product_information tool."
        "When calling the product_information tool, do not make any assumptions"
        "Only give the user very basic information about the product; the product name, id, very short description, and the price."
        "If the user asks for more information about any product, provide it."
        "If the user asks you a question you cannot answer, transfer back to the triage agent."
    ),
)

refunds_agent_tools = [
    refund_item,
    notify_customer,
    create_agent_transfer(agent_name="sales_agent"),
]
refunds_agent = create_react_agent(
    model,
    refunds_agent_tools,
    state_modifier=(
        "You are a refund agent that handles all actions related to refunds after a return has been processed. "
        "If product id and user id is present in the context information, confirm with the user that they are correct. "
        "If not present, you must ask for both the user ID and product ID to initiate a refund. "
        "Do not use any other context information to determine whether the right user id or product id has been provided - just accept the input as is. "
        "If the user asks you to notify them, you must ask them what their preferred method of notification is. For notifications, you must "
        "ask them for user_id and method in one message."
        "If the user asks you a question you cannot answer, transfer back to the triage agent."
    ),
)

sales_agent_tools = [
    order_item,
    notify_customer,
    create_agent_transfer(agent_name="sales_agent"),
    create_agent_transfer(agent_name="triage_agent"),
    create_agent_transfer(agent_name="refunds_agent"),
    create_agent_transfer(agent_name="product_agent"),
]

sales_agent = create_react_agent(
    model,
    sales_agent_tools,
    state_modifier=(
        "You are a sales agent that handles all actions related to placing an order to purchase a product."
        "Regardless of what the user wants to purchase, must ask for the user ID. "
        "If the product id is present in the context information, use it. Otherwise, you must ask for the product ID as well. "
        "An order cannot be placed without these two pieces of information. Ask for both user_id and product_id in one message. "
        "If the user asks you to notify them, you must ask them what their preferred method is. For notifications, you must "
        "ask them for user_id and method in one message."
        "If the user asks you a question you cannot answer, transfer back to the triage agent."
    ),
)


def call_triage_agent(state: MessagesState, config) -> Command[Literal["triage_agent", "human"]]:
    thread_id = config["configurable"].get("thread_id", "UNKNOWN_THREAD_ID")  # Get thread_id from config

    # Get the active agent from the userdata container
    # the userdata container is a Cosmos DB container that stores each user session id.
    # The sessionId maps to the thread_id in the checkpointer store (named 'Chat').
    # The checkpointer store is used to store the state of the conversation.
    # the userdata container is modelled using hierarchical partitioning to allow multi-tenancy.
    # In this sample, a single tenant and user is hardcoded for simplicity.
    # This can be adapted to serve multiple tenants and users.
    activeAgent = userdata_container.query_items(
        query=f"SELECT c.activeAgent FROM c WHERE c.id = '{thread_id}' and c.tenantId = 'cli-test' and c.userId = 'cli-test' and c.sessionId = '{thread_id}'",
        enable_cross_partition_query=True
    )
    result = list(activeAgent)
    if result:
        active_agent_value = result[0]['activeAgent']
    else:
        if local_interactive_mode:
            update_userdata_container({
                "id": thread_id,
                "tenantId": "cli-test",
                "userId": "cli-test",
                "sessionId": thread_id,
                "name": "cli-test",
                "age": "cli-test",
                "address": "cli-test",
                "activeAgent": "unknown",
                "ChatName": "cli-test",
                "messages": []
            })
        active_agent_value = None  # or handle the case where no result is found

    # if active agent is something other than unknown or triage_agent,
    # then transfer directly to that agent to respond to the last collected user input
    if active_agent_value is not None and active_agent_value != "unknown" and active_agent_value != "triage_agent":
        return Command(update=state, goto=active_agent_value)
    else:
        response = triage_agent.invoke(state)
        return Command(update=response, goto="human")


def call_product_agent(state: MessagesState, config) -> Command[Literal["sales_agent", "human"]]:
    thread_id = config["configurable"].get("thread_id", "UNKNOWN_THREAD_ID")
    if local_interactive_mode:
        patch_active_agent(tenantId="cli-test", userId="cli-test", sessionId=thread_id,
                           activeAgent="product_agent")
    response = product_agent.invoke(state)
    return Command(update=response, goto="human")


def call_sales_agent(state: MessagesState, config) -> Command[Literal["sales_agent", "human"]]:
    thread_id = config["configurable"].get("thread_id", "UNKNOWN_THREAD_ID")
    # Get userId from state
    if local_interactive_mode:
        patch_active_agent(tenantId="cli-test", userId="cli-test", sessionId=thread_id,
                           activeAgent="sales_agent")
    response = sales_agent.invoke(state)  # Invoke sales agent with state
    return Command(update=response, goto="human")


def call_refunds_agent(state: MessagesState, config) -> Command[Literal["refunds_agent", "human"]]:
    thread_id = config["configurable"].get("thread_id", "UNKNOWN_THREAD_ID")
    if local_interactive_mode:
        patch_active_agent(tenantId="cli-test", userId="cli-test", sessionId=thread_id,
                           activeAgent="refunds_agent")
    response = refunds_agent.invoke(state)
    return Command(update=response, goto="human")


# The human_node with interrupt function serves as a mechanism to stop
# the graph and collect user input for multi-turn conversations.
def human_node(state: MessagesState, config) -> None:
    """A node for collecting user input."""
    interrupt(value="Ready for user input.")
    return None

builder = StateGraph(MessagesState)
builder.add_node("triage_agent", call_triage_agent)
builder.add_node("sales_agent", call_sales_agent)
builder.add_node("refunds_agent", call_refunds_agent)
builder.add_node("product_agent", call_product_agent)
builder.add_node("human", human_node)

builder.add_edge(START, "triage_agent")

checkpointer = CosmosDBSaver(database_name=DATABASE_NAME, container_name=CHECKPOINT_CONTAINER)
graph = builder.compile(checkpointer=checkpointer)

import uuid
from langchain.schema import AIMessage  # ✅ Import AIMessage

thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # Ensure thread_id is a string


def interactive_chat():
    global local_interactive_mode
    local_interactive_mode = True
    print("Welcome to the interactive multi-agent shopping assistant.")
    print("Type 'exit' to end the conversation.\n")

    user_input = input("You: ")
    conversation_turn = 1

    while user_input.lower() != "exit":

        input_message = {"messages": [{"role": "user", "content": user_input}]}

        response_found = False  # Track if we received an AI response

        for update in graph.stream(
                input_message,
                config=thread_config,
                stream_mode="updates",
        ):
            for node_id, value in update.items():
                if isinstance(value, dict) and value.get("messages"):
                    last_message = value["messages"][-1]  # Get last message
                    if isinstance(last_message, AIMessage):
                        print(f"{node_id}: {last_message.content}\n")
                        response_found = True

        if not response_found:
            print("DEBUG: No AI response received.")

        # Get user input for the next round
        user_input = input("You: ")
        conversation_turn += 1


if __name__ == "__main__":
    interactive_chat()
