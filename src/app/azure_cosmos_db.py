import json
import os
import sys
import uuid
import concurrent.futures
from datetime import time
from typing import List, Optional
from azure.cosmos import ContainerProxy
from azure.cosmos import exceptions
from azure.cosmos import CosmosClient, PartitionKey

from src.app import azure_open_ai
from pathlib import Path

# Azure Cosmos DB configuration
COSMOS_DB_URL = os.getenv("COSMOSDB_ENDPOINT")
COSMOS_DB_KEY = os.getenv("COSMOSDB_KEY")
DATABASE_NAME = "CosmosMultiAgentLangGraph"
CHECKPOINT_CONTAINER = "Chat"

cosmos_client = None
database = None
container = None

# Define Cosmos DB container for user data
USERDATA_CONTAINER = "UserData"
userdata_container = None
account_container = None

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
PRODUCTS_CONTAINER = "Products"
USERS_CONTAINER = "Users"
PURCHASE_HISTORY_CONTAINER = "PurchaseHistory"

client = None


def get_cosmos_client():
    global cosmos_client
    if cosmos_client is None:
        cosmos_client = CosmosClient(COSMOS_DB_URL, credential=COSMOS_DB_KEY)
    return cosmos_client


# Initialize Cosmos DB client
try:
    client = get_cosmos_client()
    database = client.create_database_if_not_exists(DATABASE_NAME)
    container = database.create_container_if_not_exists(
        id=CHECKPOINT_CONTAINER,
        partition_key=PartitionKey(path="/partition_key"),
        offer_throughput=400,
    )
    account_container = database.create_container_if_not_exists(
        id="Account",
        partition_key=PartitionKey(path="/accountId"),
        offer_throughput=400,
    )
    print(f"[DEBUG] Connected to Cosmos DB: {DATABASE_NAME}/{CHECKPOINT_CONTAINER}")

    database = client.get_database_client(DATABASE_NAME)
    userdata_container = database.create_container_if_not_exists(
        # Create a Cosmos DB container with hierarchical partition key
        id=USERDATA_CONTAINER, partition_key=PartitionKey(path=["/tenantId", "/userId", "/sessionId"], kind="MultiHash")
    )
except Exception as e:
    print(f"[ERROR] Error initializing Cosmos DB: {e}")
    raise e


# update the user data container
def update_userdata_container(data):
    try:
        userdata_container.upsert_item(data)
    except Exception as e:
        print(f"[ERROR] Error saving user data to Cosmos DB: {e}")
        raise e


# fetch the user data from the container by tenantId, userId
def fetch_userdata_container(tenantId, userId):
    try:
        query = f"SELECT * FROM c WHERE c.tenantId = '{tenantId}' AND c.userId = '{userId}'"
        items = list(userdata_container.query_items(query=query, enable_cross_partition_query=True))
        return items
    except Exception as e:
        print(f"[ERROR] Error fetching user data for tenantId: {tenantId}, userId: {userId}: {e}")
        raise e


# fetch the user data from the container by tenantId, userId, sessionId
def fetch_userdata_container_by_session(tenantId, userId, sessionId):
    try:
        query = f"SELECT * FROM c WHERE c.tenantId = '{tenantId}' AND c.userId = '{userId}' AND c.sessionId = '{sessionId}'"
        items = list(userdata_container.query_items(query=query, enable_cross_partition_query=True))
        print(
            f"[DEBUG] Fetched {len(items)} user data for tenantId: {tenantId}, userId: {userId}, sessionId: {sessionId}")
        return items
    except Exception as e:
        print(
            f"[ERROR] Error fetching user data for tenantId: {tenantId}, userId: {userId}, sessionId: {sessionId}: {e}")
        raise e


# patch the active agent in the user data container using patch operation
def patch_active_agent(tenantId, userId, sessionId, activeAgent):
    try:
        #filter = "from c WHERE p.used = false"

        operations = [
            {'op': 'replace', 'path': '/activeAgent', 'value': activeAgent}
        ]

        try:
            pk = [tenantId, userId, sessionId]
            userdata_container.patch_item(item=sessionId, partition_key=pk,
                                          patch_operations=operations)
        except Exception as e:
            print('\nError occurred. {0}'.format(e.message))
    except Exception as e:
        print(
            f"[ERROR] Error patching active agent for tenantId: {tenantId}, userId: {userId}, sessionId: {sessionId}: {e}")
        raise e

    # deletes the user data from the container by tenantId, userId, sessionId


def delete_userdata_item(tenantId, userId, sessionId):
    try:
        query = f"SELECT * FROM c WHERE c.tenantId = '{tenantId}' AND c.userId = '{userId}' AND c.sessionId = '{sessionId}'"
        items = list(userdata_container.query_items(query=query, enable_cross_partition_query=True))
        if len(items) == 0:
            print(f"[DEBUG] No user data found for tenantId: {tenantId}, userId: {userId}, sessionId: {sessionId}")
            return
        for item in items:
            userdata_container.delete_item(item, partition_key=[tenantId, userId, sessionId])
            print(f"[DEBUG] Deleted user data for tenantId: {tenantId}, userId: {userId}, sessionId: {sessionId}")
    except Exception as e:
        print(
            f"[ERROR] Error deleting user data for tenantId: {tenantId}, userId: {userId}, sessionId: {sessionId}: {e}")
        raise e


# Create the "Account" container partitioned by accountId


# Function to create an account record
# def create_account_record(account_data):
#     try:
#         account_container.upsert_item(account_data)
#         print(f"[DEBUG] Account record created: {account_data}")
#     except Exception as e:
#         print(f"[ERROR] Error creating account record: {e}")
#         raise e
#
#
#
#
#
# def fetch_latest_account_number():
#     try:
#         query = "SELECT VALUE MAX(c.accountId) FROM c"
#         items = list(account_container.query_items(query=query, enable_cross_partition_query=True))
#         print(f"[DEBUG] Fetched {len(items)} account numbers")
#         if items:
#             latest_account_id = items[0]
#             if latest_account_id is None:
#                 return 0
#             print(f"[DEBUG] Latest account ID: {latest_account_id}")
#             latest_account_number = int(latest_account_id[1:])  # Assuming accountId is in the format 'a<number>'
#             print(f"[DEBUG] Latest account number: {latest_account_number}")
#             return latest_account_number
#
#         else:
#             return None
#     except Exception as e:
#         print(f"[ERROR] Error fetching latest account number: {e}")
#         raise e
#
#
# # Function to create a transaction record
# def create_transaction_record(transaction_data):
#     try:
#         account_container.upsert_item(transaction_data)
#         print(f"[DEBUG] Transaction record created: {transaction_data}")
#     except Exception as e:
#         print(f"[ERROR] Error creating transaction record: {e}")
#         raise e

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "dimensions": 1536,
            "distanceFunction": "cosine",
        }
    ]
}

# full_text_policy = {
#     "defaultLanguage": "en-US",
#     "fullTextPaths": [
#         {
#             "path": "/product_description",
#             "language": "en-US",
#         }
#     ]
# }

indexing_policy = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
    #"fullTextIndexes": [{"path": "/product_description"}],
}


# Create Containers with Vector and Full-Text Indexing Policies
def create_containers():
    try:
        users_container = database.create_container_if_not_exists(
            id=USERS_CONTAINER,
            partition_key=PartitionKey(path="/user_id"),
            offer_throughput=400,
        )

        print(
            f"Container {USERS_CONTAINER} created."
        )

        purchase_history_container = database.create_container_if_not_exists(
            id=PURCHASE_HISTORY_CONTAINER,
            partition_key=PartitionKey(path="/user_id"),
            offer_throughput=400
        )

        print(
            f"Container {PURCHASE_HISTORY_CONTAINER} created."
        )

        products_container = database.create_container_if_not_exists(
            id=PRODUCTS_CONTAINER,
            partition_key=PartitionKey(path="/category"),
            offer_throughput=10000,
            vector_embedding_policy=vector_embedding_policy,
            #full_text_policy=full_text_policy,
            indexing_policy=indexing_policy,
        )

        print(
            f"Container {PRODUCTS_CONTAINER} created with vector and full-text search indexing."
        )
    except exceptions.CosmosHttpResponseError as e:
        print(f"Container creation failed: {e}")


def add_user(user_id, first_name, last_name, email, phone):
    container = database.get_container_client(USERS_CONTAINER)
    user = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone": phone
    }
    try:
        container.create_item(body=user)
    except exceptions.CosmosResourceExistsError:
        print(f"User with user_id {user_id} already exists.")


def add_purchase(user_id, date_of_purchase, item_id, amount, product_name, category):
    container = database.get_container_client(PURCHASE_HISTORY_CONTAINER)
    purchase = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "date_of_purchase": date_of_purchase,
        "product_id": item_id,
        "product_name": product_name,
        "category": category,
        "amount": amount
    }
    try:
        container.create_item(body=purchase)
    except exceptions.CosmosResourceExistsError:
        print(f"Purchase already exists for user_id {user_id} on {date_of_purchase} for item_id {item_id}.")


def add_product(product_id, product_name, category, product_description, price):
    container = database.get_container_client(PRODUCTS_CONTAINER)
    product_description_vector = azure_open_ai.generate_embedding(product_description)
    product = {
        "id": str(uuid.uuid4()),
        "product_id": product_id,
        "product_name": product_name,
        "category": category,
        "product_description": product_description,
        "product_description_vector": product_description_vector,
        "price": price
    }
    try:
        container.create_item(body=product)
    except exceptions.CosmosResourceExistsError:
        print(f"Product with product_id {product_id} already exists.")


import os
import json
import uuid
import sys
import time
import concurrent.futures
from typing import List, Optional
from azure.cosmos import ContainerProxy

import os
import json
import uuid
import sys
import time
import concurrent.futures
from typing import List, Optional
from azure.cosmos import ContainerProxy


def process_and_insert_data(
        filename: str,
        container: ContainerProxy,
        vector_field: Optional[str] = None,
        full_text_fields: Optional[List[str]] = None,
        max_concurrent_tasks: int = 3  # restrict concurrent tasks to avoid rate limiting
):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    with open(filename, "r") as f:
        data = json.load(f)

    total_records = len(data)
    print(f"Total records in file: {total_records}")
    print(f"Processing data from file...")

    processed_count = 0
    failed_processing_count = 0
    inserted_count = 0
    failed_insert_count = 0

    def process_entry(entry):
        nonlocal processed_count, failed_processing_count
        try:
            if full_text_fields is not None:
                for field in full_text_fields:
                    if field in entry and isinstance(entry[field], list):
                        entry[field] = [", ".join(map(str, entry[field]))]

            # Generate vector embedding
            if vector_field and vector_field in entry and isinstance(entry[vector_field], str):
                entry["embedding"] = azure_open_ai.generate_embedding(entry[vector_field])

            # Assign a unique ID
            entry["id"] = str(uuid.uuid4())

            # Add an empty text field
            entry["text"] = ""

            # Ensure metadata exists
            entry["metadata"] = entry.get("metadata", {})

            # Copy relevant fields into metadata
            for field in ["product_description", "product_id", "product_name", "category", "price", "id"]:
                if field in entry:
                    entry["metadata"][field] = entry[field]

            # Check document size
            size = sys.getsizeof(json.dumps(entry))
            if size > 2 * 1024 * 1024:  # 2MB in bytes
                print(f"Skipping document {entry['id']} - Too large: {size} bytes")
                failed_processing_count += 1
                return None
            processed_count += 1
            return entry
        except Exception as e:
            print(f"Failed processing entry: {e}")
            failed_processing_count += 1
            return None

    # Process entries concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
        processed_entries = list(executor.map(process_entry, data))

    # Remove None values (skipped due to errors or size constraints)
    processed_entries = [entry for entry in processed_entries if entry is not None]

    def upsert_entry(entry):
        nonlocal inserted_count, failed_insert_count
        retries = 5
        for attempt in range(retries):
            try:
                container.upsert_item(entry)
                inserted_count += 1
                return
            except Exception as e:
                if "TooManyRequests" in str(e) and attempt < retries - 1:
                    wait_time = (2 ** attempt) * 0.5  # Exponential backoff
                    print(f"Retry {attempt + 1} for entry {entry['id']} after {wait_time} seconds due to rate limit.")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to insert entry {entry['id']} after {attempt + 1} attempts: {e}")
                    failed_insert_count += 1
                    return

    print(f"Inserting data into {container.id}...")
    # Upsert items concurrently with a limited number of workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
        executor.map(upsert_entry, processed_entries)

    print(f"Finished inserting data from {filename} into {container.id}.")
    print(f"Summary:")
    print(f"  - Total records: {total_records}")
    print(f"  - Processed successfully: {processed_count}")
    print(f"  - Failed processing: {failed_processing_count}")
    print(f"  - Inserted successfully: {inserted_count}")
    print(f"  - Failed inserts: {failed_insert_count}")


# def process_and_insert_data(
#         filename: str,
#         container: ContainerProxy,
#         vector_field: Optional[str] = None,
#         full_text_fields: Optional[List[str]] = None,
# ):
#     if not os.path.exists(filename):
#         print(f"File {filename} not found.")
#         return
#
#     with open(filename, "r") as f:
#         data = json.load(f)
#
#     if len(data) > 300:
#         data = data[226:]
#
#     for entry in data:
#         if full_text_fields is not None:
#             for field in full_text_fields:
#                 if field in entry and isinstance(entry[field], list):
#                     entry[field] = [", ".join(map(str, entry[field]))]
#
#         # Generate vector embedding
#         if vector_field and vector_field in entry and isinstance(entry[vector_field], str):
#             entry["embedding"] = azure_open_ai.generate_embedding(entry[vector_field])
#
#         # Insert into CosmosDB
#         entry["id"] = str(uuid.uuid4())
#         size = sys.getsizeof(json.dumps(entry))
#         if size > 2 * 1024 * 1024:  # 2MB in bytes
#             print(f"Document {entry['id']} is too large: {size} bytes")
#         container.upsert_item(entry)
#
#     print(f"Inserted data from {filename} into {container.id}.")


def main():
    try:
        print("[DEBUG] Starting main function")

        # Create Containers
        print("[DEBUG] Creating containers")
        create_containers()

        products_container = database.get_container_client(PRODUCTS_CONTAINER)
        users_container = database.get_container_client(USERS_CONTAINER)
        purchase_history_container = database.get_container_client(PURCHASE_HISTORY_CONTAINER)

        # Insert data into CosmosDB with embedding and indexing
        current_directory = Path(__file__).parent  # Directory of the running script
        data_directory = current_directory / "data"  # Append "/data/"
        print(f"[DEBUG] Data directory: {data_directory}")

        file_prefix = data_directory

        print("[DEBUG] Processing and inserting data for final_products.json")
        process_and_insert_data(
            str(file_prefix / "final_products.json"),
            products_container,
            "product_description",
            ["product_description"],
        )

        print("[DEBUG] Processing and inserting data for users.json")
        process_and_insert_data(str(file_prefix / "users.json"), users_container)

        print("[DEBUG] Processing and inserting data for purchase_history.json")
        process_and_insert_data(
            str(file_prefix / "purchase_history.json"),
            purchase_history_container
        )

        print(
            "[DEBUG] Data successfully inserted into CosmosDB with embeddings, vector search, and full-text search indexing!")
    except Exception as e:
        print(f"[ERROR] An error occurred in the main function: {e}")
        raise e


if __name__ == "__main__":
    main()
