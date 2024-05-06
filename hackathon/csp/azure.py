import traceback

from hackathon.csp import CSPBase
from hackathon.embeddings import OpenAIEmbeddings, AzureEmbeddings
from hackathon.stt import OpenAISTT, AzureSTT
from hackathon.chat import OpenAIChat, AzureChat

from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex
)
from azure.search.documents.models import VectorizedQuery

from hackathon.glob.prompts import *

import pandas as pd

import os
import time

from dotenv import load_dotenv
load_dotenv()

from openai import AzureOpenAI


class AzureCSP(CSPBase):
    def __init__(self, embeddings=None, chat=None, stt=None, *args, **kwargs):
        super().__init__()

        if embeddings == 'openai' or embeddings is None:
            if embeddings is None: print("No embeddings client provided. Setting Default: OpenAI")
            self.embeddings_client = OpenAIEmbeddings(kwargs.get('openai_api_key', None))
        elif embeddings == 'azure':
            self.embeddings_client = AzureEmbeddings()
        else:
            raise NotImplementedError(f"Embedding Client {embeddings} not Implemented")

        if stt == 'openai' or stt is None:
            if stt is None: print("No stt client provided. Setting Default: OpenAI")
            self.stt_client = OpenAISTT(kwargs.get('openai_api_key', None))
        elif stt == 'azure':
            self.stt_client = AzureSTT(kwargs.get('openai_api_key', None))
        else:
            raise NotImplementedError(f"STT Client {stt} not Implemented")

        if chat == 'openai' or chat is None:
            if chat is None: print("No chat client provided. Setting Default: OpenAI")
            self.chat_client = OpenAIChat(kwargs.get('openai_api_key', None))
        elif chat == 'azure':
            self.chat_client = AzureChat(kwargs.get('openai_api_key', None))
            pass
        else:
            raise NotImplementedError(f"Chat Client {chat} not Implemented")

        # Requires correct key to initialize server
        self.search_credential = AzureKeyCredential(os.getenv("AZURE_KEY"))
        self.index_client = SearchIndexClient(endpoint=os.getenv("AZURE_ENDPOINT"), credential=self.search_credential)

    def index_data(self, file_path):
        for path in os.listdir(file_path):
            ind_path = os.path.join(file_path, path, "processed", "index.csv")
            if os.path.isfile(ind_path):
                df_ind = pd.read_csv(ind_path)
                df_ind["Type"] = "title"
            else:
                df_ind = None

            qna_path = os.path.join(file_path, path, "processed", "qna.csv")
            if os.path.isfile(qna_path):
                df_qna = pd.read_csv(qna_path)
                df_qna["Type"] = "qna"
            else:
                df_qna = None

            if df_ind is not None and df_qna is not None:
                df = pd.concat([df_ind, df_qna])
            else:
                df = df_ind or df_qna

            df.columns = df.columns.str.lower()

            input_documents = df.to_dict(orient='records')

            index_name = path.split("-")[-1].lstrip(" ").lower().replace(" ", "_")


            try:
                fields = [
                    SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True,
                                facetable=True),
                    SearchableField(name="title", type=SearchFieldDataType.String),
                    SearchableField(name="content", type=SearchFieldDataType.String),
                    SearchableField(name="category", type=SearchFieldDataType.String,
                                    filterable=True),
                    SimpleField(name="severity", type=SearchFieldDataType.Int32, filterable=True, sortable=True,
                                facetable=True),
                    SimpleField(name="type", type=SearchFieldDataType.String, filterable=True, sortable=True,
                                facetable=True),
                    SearchField(name="title_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                                searchable=True, vector_search_dimensions=1536,
                                vector_search_profile_name="myHnswProfile"),
                    SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                                searchable=True, vector_search_dimensions=1536,
                                vector_search_profile_name="myHnswProfile"),
                ]

                # Configure the vector search configuration
                vector_search = VectorSearch(
                    algorithms=[
                        HnswAlgorithmConfiguration(
                            name="myHnsw"
                        )
                    ],
                    profiles=[
                        VectorSearchProfile(
                            name="myHnswProfile",
                            algorithm_configuration_name="myHnsw",
                        )
                    ]
                )

                semantic_config = SemanticConfiguration(
                    name="my-semantic-config",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="title"),
                        keywords_fields=[SemanticField(field_name="category")],
                        content_fields=[SemanticField(field_name="content")]
                    )
                )

                semantic_search = SemanticSearch(configurations=[semantic_config])
                index = SearchIndex(name=index_name, fields=fields,
                                    vector_search=vector_search, semantic_search=semantic_search)
                result = self.index_client.create_or_update_index(index)
                print(f' {result.name} created')

                docs = self._get_sample_documents(input_documents)

                search_client = SearchClient(endpoint=os.getenv("AZURE_ENDPOINT"), index_name=index_name, credential=self.search_credential)
                result = search_client.upload_documents(docs)
                print(f"Uploaded {len(docs)} documents")

            except Exception as e:
                print(f"Failed to indexed {path}, {e}\n{traceback.print_exc()}")

        return "Successfully Indexed Data"

    def simple_hs(self, prompt, index_name):
        search_client = SearchClient(endpoint=os.getenv("AZURE_ENDPOINT"), index_name=index_name, credential=self.search_credential)
        query_embeddings = self.embeddings_client.get_embeddings(prompt)

        vector_query = VectorizedQuery(vector=query_embeddings, k_nearest_neighbors=5, fields="title_vector,content_vector")

        response_holder = []
        for t in ["qna", "title"]:
            response = search_client.search(search_text=None,
                                            vector_queries=[vector_query],
                                            select=["title", "content", "severity"],
                                            filter=f"type eq '{t}'")


            for result in response:
                search_result = {
                    "title": result.get('title'),
                    "content": result['content'],
                    "severity": result["severity"],
                    "score": result['@search.score'],
                    "type": t
                }

                response_holder.append(search_result)

        response_holder_total = len(response_holder)

        return response_holder, response_holder_total

    def start_conversation(self, prompt, history, state):
        response, history = self._get_category(prompt, history.copy())
        categories = [int(i) for i in response if i.isnumeric()]

        print(len(history))

        # Uncomment the return below to see the output of the above func on streamlit on prompt enter
        # print(categories)
        # return response, history, state

        print(categories)

        # Process categories
        index = "general"
        if len(categories) == 1:
            index = id_map[categories[0]]
        elif len(categories) > 1:
            response, history = self._narrow_category_follow_up(prompt, history.copy(), categories)
            # Uncomment the return below to see the output of the above func on streamlit on prompt enter
            return response, history, state

        # Narrow Category function is not required if we are not using state
        print("253", index)

        if index == "out_of_category" or index == "general":
            response, history = self._process_out_of_category(prompt, history.copy(), index)
            # Uncomment the return below to see the output of the above func on streamlit on prompt enter
            return response, history, state

        response_holder, holder_length = self.simple_hs(prompt, index_name=index)
        final_prompt = self._create_context(prompt, response_holder)  # Create the context and final prompt in this

        response, history = self.chat_client.get_response(final_prompt, history.copy())

        # Uncomment the return below to see the output of the above func on streamlit on prompt enter
        return response, history, state

    def _get_category(self, prompt, history):
        if len(history) == 0:
            system_prompt = get_category_prompt_if_no_history
        else:
            system_prompt = get_category_prompt_if_history

        response, history = self.chat_client.get_response(prompt, history.copy(), system_prompt)
        return response, history

    def _process_out_of_category(self, prompt, history, index):
        if index == "out_of_category":
            return self.chat_client.get_response(prompt, history.copy(), system_prompt=out_of_category_prompt)
        else:
            return self.chat_client.get_response(prompt, history.copy())

    def _narrow_category_follow_up(self, prompt, history, categories):
        system_prompt = f"""
        {generate_follow_up_prompt}
         Here are the categories we think this question could apply to: {categories}
        """

        return self.chat_client.get_response(prompt, history.copy(), system_prompt)

    def _narrow_category(self, prompt, history):
        return self.chat_client.get_response(prompt, history.copy())

    def _create_context(self, prompt, response_holder):
        context = "Here are 5 pieces of source material that you can use to help formulate your response:\n"

        i = 1
        for item in response_holder:
            if {item['type']} == 'title':
                context += f'{i}; Title: {item["title"]}; Content: {item["content"]}; Severity: {item["severity"]}; \n'
                i += 1

        context += "\nand Here are 5 Questions and Answers relating to this topic that you can also use to help you formulate your response:\n"

        i = 1
        for item in response_holder:
            if {item['type']} == 'qna':
                context += f'{i}; Question: {item["title"]}; Answer: {item["content"]}; \n'
                i += 1

        system_prompt = f"""
        System prompt: {system_prompt1}
        {context}\n
        """

        return system_prompt

    def _check_change_in_index(self):
        pass

    def _get_hist_string(history):
        hist_string = ""
        for item in history:
            hist_string += f"{item['role']}: {item['content']}\n"

        return hist_string

    def speech_to_text(self, audio_data):
        return self.stt_client.get_text(audio_data)

    def _get_sample_documents(self, documents):
        sample_documents = []
        failed_ids = []

        for document in documents:
            try:
                content = document.get("content", "")
                title = document.get("title", "")

                for i in range(5):
                    try:
                        content_embeddings = self.embeddings_client.get_embeddings(content)
                        break
                    except Exception as e:
                        if i == 4:
                            raise
                        else:
                            time.sleep(1)

                for i in range(5):
                    try:
                        title_embeddings = self.embeddings_client.get_embeddings(title)
                        break
                    except Exception as e:
                        if i == 4:
                            raise
                        else:
                            time.sleep(1)

                document["title_vector"] = title_embeddings
                document["content_vector"] = content_embeddings
                document["id"] = str(document["id"])

                sample_documents.append(document)
            except Exception as ex:
                question_id = document.get("ID", "Unknown")
                failed_ids.append(str(question_id))
                print(f"Error processing document ID {question_id}: {ex} \n {traceback.format_exc()}")

        if failed_ids:
            with open('failedIds.txt', 'w') as f:
                f.write('\n'.join(failed_ids))

        return sample_documents
