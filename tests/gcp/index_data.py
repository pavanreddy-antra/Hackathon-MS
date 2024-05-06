import os
from hackathon.csp import AzureCSP

import sys
sys.path.append("C:\\Users\\Pavan Reddy\\Desktop\\Hackathon")

gcpscp = AzureCSP(embeddings='azure', chat='azure', stt='azure')

# file_path = "gs://hackathon-12341/hackathon/index.json"
file_path = "..\..\Data"


print(gcpscp.index_data(file_path=file_path))

# print(gcpscp.simple_hs("My friend is hiding his drinking habits", index_name='warning_signs'))

# print(gcpscp.chat_client.get_response("Hello!")[0])