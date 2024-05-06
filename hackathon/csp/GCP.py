import pandas as pd
import traceback

from hackathon.csp import CSPBase
from hackathon.embeddings import OpenAIEmbeddings, GCPEmbeddings
from hackathon.stt import OpenAISTT, GCPSTT
from hackathon.chat import OpenAIChat, GeminiChat
from hackathon.glob.prompts import *

from google.cloud import bigquery
from google.cloud import aiplatform
from google.oauth2 import service_account

import os
import time

from dotenv import load_dotenv

load_dotenv()

alpha = 0.5


class GCPCSP(CSPBase):
    def __init__(self, embeddings=None, chat=None, stt=None, *args, **kwargs):
        super().__init__()

        if embeddings == 'openai' or embeddings is None:
            if embeddings is None: print("No embeddings client provided. Setting Default: OpenAI")
            self.embeddings_client = OpenAIEmbeddings(kwargs.get('openai_api_key', None))
        elif embeddings == 'gcp':
            self.embeddings_client = GCPEmbeddings(kwargs.get('gcp_api_key', None))
        else:
            raise NotImplementedError(f"Embedding Client {embeddings} not Implemented")

        if stt == 'openai' or stt is None:
            if stt is None: print("No stt client provided. Setting Default: OpenAI")
            self.stt_client = OpenAISTT(kwargs.get('openai_api_key', None))
        elif stt == 'gcp':
            self.stt_client = GCPSTT(kwargs.get('gcp_api_key', None))
        else:
            raise NotImplementedError(f"STT Client {stt} not Implemented")

        if chat == 'openai' or chat is None:
            if chat is None: print("No chat client provided. Setting Default: OpenAI")
            self.chat_client = OpenAIChat(kwargs.get('openai_api_key', None))
        elif chat == 'gcp':
            self.chat_client = GeminiChat(kwargs.get('gcp_api_key', None))
        else:
            raise NotImplementedError(f"Chat Client {chat} not Implemented")

        self.credentials = service_account.Credentials.from_service_account_file(os.getenv("GCP_CREDENTIALS_PATH"))
        # self.index_client = SearchIndexClient(endpoint=os.getenv("AZURE_ENDPOINT"), credential=search_credential)

    def index_data(self, file_path, project_id, dataset_name="alcohol", location="us-central1"):
        for path in os.listdir(file_path):
            bigquery_client = bigquery.Client(credentials=self.credentials, project=project_id)

            dataset_id = f"{project_id}.{dataset_name}"
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = location
            bigquery_client.create_dataset(dataset, exists_ok=True)

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

            df['ID'] = range(1, len(df) + 1)
            input_documents = df.to_dict(orient='records')[:10]

            index_name = path.split("-")[-1].lstrip(" ").lower().replace(" ", "_")

            schema = [
                bigquery.SchemaField("ID", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("Type", "STRING"),
                bigquery.SchemaField("Category", "STRING"),
                bigquery.SchemaField("Title", "STRING"),
                bigquery.SchemaField("Content", "STRING"),
                bigquery.SchemaField("Severity", "INTEGER"),
                bigquery.SchemaField("Title_Vector", "FLOAT64", mode="REPEATED"),
                bigquery.SchemaField("Content_Vector", "FLOAT64", mode="REPEATED")
            ]

            table_id = f"{dataset.project}.{dataset.dataset_id}.{index_name}"
            table = bigquery.Table(table_id, schema=schema)
            table = bigquery_client.create_table(table, exists_ok=True)

            sample_documents = self._get_sample_documents(input_documents)

            errors = None

            for i in range(5):
                try:
                    errors = bigquery_client.insert_rows_json(table_id, sample_documents)
                    time.sleep(5)
                    break
                except Exception as e:
                    print(e)
                    bigquery_client = bigquery.Client(credentials=self.credentials, project=project_id)

                    dataset_id = f"{project_id}.{dataset_name}"
                    dataset = bigquery.Dataset(dataset_id)
                    dataset.location = location
                    bigquery_client.create_dataset(dataset, exists_ok=True)

                    table_id = f"{dataset.project}.{dataset.dataset_id}.{index_name}"
                    table = bigquery.Table(table_id, schema=schema)
                    table = bigquery_client.create_table(table, exists_ok=True)

            if errors:
                print("Encountered errors while uploading data:", errors)
            else:
                print(f"{index_name} loaded into BigQuery: {table_id}")

    # def index_data(self, index_name, file_path, project_id, location, vector_length):
    #     aiplatform.init(credentials=self.credentials, project=project_id, location=location)
    #
    #     my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    #         display_name=index_name,
    #         contents_delta_uri=file_path, # GS URI
    #         dimensions=768,
    #         approximate_neighbors_count=10,
    #     )
    #
    #     my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    #         display_name=index_name,
    #         public_endpoint_enabled=True
    #     )
    #
    #     DEPLOYED_INDEX_ID = f"deployed_{index_name}"
    #
    #     my_index_endpoint.deploy_index(
    #         index=my_index, deployed_index_id=DEPLOYED_INDEX_ID
    #     )

    def simple_hs(self, prompt, index_name, project_id):
        dataset = "alcohol"
        location = "us-central1"

        response_holder = []

        bigquery_client = bigquery.Client(credentials=self.credentials, location=location)

        query_embedding = self.embeddings_client.get_embeddings(prompt)

        query = f"""
                CREATE TEMPORARY FUNCTION td(a ARRAY<FLOAT64>, b ARRAY<FLOAT64>, idx INT64) AS (
                  (a[OFFSET(idx)] - b[OFFSET(idx)]) * (a[OFFSET(idx)] - b[OFFSET(idx)])
                );
                CREATE TEMPORARY FUNCTION term_distance(a ARRAY<FLOAT64>, b ARRAY<FLOAT64>) AS ((
                  SELECT SQRT(SUM( td(a, b, idx))) FROM UNNEST(GENERATE_ARRAY(0, ARRAY_LENGTH(a)-1)) idx
                ));
                CREATE TEMPORARY FUNCTION combined_distance(content_distance FLOAT64, title_distance FLOAT64, alpha FLOAT64) AS (
                  alpha * content_distance + (1 - alpha) * title_distance
                );
                
                SELECT *,
                  term_distance(@query_embedding, Content_Vector) AS content_distance,
                  term_distance(@query_embedding, Title_Vector) AS title_distance,
                  combined_distance(
                    term_distance(@query_embedding, Content_Vector), 
                    term_distance(@query_embedding, Title_Vector), 
                    @alpha
                  ) AS final_score
                FROM `{project_id}.{dataset}.{index_name}`
                WHERE Type = @type
                ORDER BY final_score ASC
                LIMIT 10;
                """

        for t in ["qna", 'title']:

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", query_embedding),
                    bigquery.ScalarQueryParameter("alpha", "FLOAT64", alpha),
                    bigquery.ScalarQueryParameter("type", "STRING", t),
                ]
            )

            query_job = bigquery_client.query(query, job_config=job_config)
            results = query_job.result()

            # for row in results:
            #     print(f"ID: {row['ID']}, Content: {row['Content']}")

            matching_ids = tuple(row['ID'] for row in results)
            if not matching_ids:
                results = []
            else:
                # Convert list to a tuple and format for SQL
                ids_tuple = tuple(matching_ids)
                ids_str = str(ids_tuple)
                if len(matching_ids) == 1:
                    ids_str = f"({matching_ids[0]})"  # Ensure a trailing comma for a single-element tuple

                sql_query = f"""
                    SELECT ID, Category, Title, Content, Severity, Type
                    FROM `{project_id}.{dataset}.{index_name}`
                    WHERE ID IN {ids_str}
                """
                query_job = bigquery_client.query(sql_query)
                results = query_job.result()

            for row in results:
                search_result = {
                    "id": row["ID"],
                    "category": row["Category"],
                    "title": row["Title"],
                    "content": row["Content"],
                    "severity": row["Severity"],
                    "type": row["Type"]
                }
                response_holder.append(search_result)

        return response_holder, len(response_holder)

    def start_conversation(self, prompt, history, state):
        response, history = self._get_category(prompt, history.copy())
        categories = [int(i) for i in response if i.isnumeric()]

        # Uncomment the return below to see the output of the above func on streamlit on prompt enter
        # print(categories)
        # return response, history, state

        print(categories)

        # Process categories
        index = None
        if len(categories) == 1:
            index = id_map[categories[0]]
        elif len(categories) > 1:
            response, history = self._narrow_category_follow_up(prompt, history.copy(), categories)
            print(response)
            # Uncomment the return below to see the output of the above func on streamlit on prompt enter
            return response, history, state

        # Narrow Category function is not required if we are not using state
        print("253", index)

        if index == "out_of_category" or index == "general":
            response, history = self._process_out_of_category(prompt, history.copy(), index)
            # Uncomment the return below to see the output of the above func on streamlit on prompt enter
            return response, history, state

        response_holder, holder_length = self.simple_hs(prompt, index_name=index, project_id='hallowed-air-418016')
        final_prompt = self._create_context(prompt, response_holder)  # Create the context and final prompt in this

        response, history = self.chat_client.get_response(final_prompt, history.copy())

        # Uncomment the return below to see the output of the above func on streamlit on prompt enter
        return response, history, state

    # def start_conversation(self, prompt, history, state):
    #     if state == "started":
    #         response, history = self._get_category(prompt, history.copy())
    #         return response, history, state
    #
    #         categories = response.split(",")
    #
    #         if len(categories) == 1:
    #             state = categories[0]
    #         elif len(categories) > 1:
    #             response, history = self._narrow_category_follow_up(prompt, history.copy(), categories)
    #             state = "followup"
    #             return response, history, state
    #
    #     if state == "follow_up":
    #         response, history = self._narrow_category()
    #
    #         categories = response.split(",")
    #
    #         if len(categories) == 1:
    #             state = categories[0]
    #         elif len(categories) > 1:
    #             response, history = self._narrow_category_follow_up(prompt, history.copy(), categories)
    #             state = "followup"
    #             return response, history, state
    #
    #     if state in ["categores go here"]:
    #         self._check_change_in_index()
    #         response_holder, holder_length = self.simple_hs(prompt, index_name=state, project_id='hallowed-air-418016')
    #         final_prompt = self._create_context(prompt, response_holder, state)
    #
    #         response, history = self.chat_client.get_response(final_prompt, history.copy())
    #
    #         return response, history, state
    #
    #     if state == "None":
    #         self._process_out_of_category()
    #
    #     response, history = self.chat_client.get_response(prompt, history.copy())
    #     return response, history, state

    def _get_category(self, prompt, history):
        if len(history) == 0:
            final_prompt = f"""
            System Prompt: {get_category_prompt_if_no_history}\n\n
            User Prompt: {prompt}
            """
        else:
            final_prompt = f"""
            {get_category_prompt_if_history}
            {prompt}
            """

        response, history = self.chat_client.get_response(final_prompt, history.copy())
        return response, history

    def _process_out_of_category(self, prompt, history, index):
        if index == "out_of_category":
            final_prompt = f"""
                            {out_of_category_prompt}
                            {prompt}
                            """
            return self.chat_client.get_response(final_prompt, history.copy())
        else:
            final_prompt = f"""
                            {prompt}
                            """
            return self.chat_client.get_response(final_prompt, [])



    def _narrow_category_follow_up(self, prompt, history, categories):
        final_prompt = f"""
        {generate_follow_up_prompt}
         Here are the categories we think this question could apply to: {categories}
        Here is the user prompt: {prompt}
        """

        return self.chat_client.get_response(final_prompt, history.copy())

    def _narrow_category(self, prompt, history):
        final_prompt = f"""
                {prompt}
                """

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

        final_prompt = f"""
        System prompt: {system_prompt1}
        {context}\n
        and here is the user prompt: {prompt}
        """

        return final_prompt

    def _check_change_in_index(self):
        pass

    def speech_to_text(self, audio_data):
        return self.stt_client.get_text(audio_data)

    def _get_sample_documents(self, documents):
        sample_documents = []
        failed_ids = []

        for document in documents:
            try:
                content = document.get("Content", "")
                title = document.get("Title", "")

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

                document["Title_Vector"] = title_embeddings
                document["Content_Vector"] = content_embeddings

                sample_documents.append(document)
            except Exception as ex:
                question_id = document.get("ID", "Unknown")
                failed_ids.append(str(question_id))
                print(f"Error processing document ID {question_id}: {ex}")

        if failed_ids:
            with open('failedIds.txt', 'w') as f:
                f.write('\n'.join(failed_ids))

        return sample_documents
