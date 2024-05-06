from flask import Flask, jsonify, request
from hackathon.csp import AzureCSP


class HackathonServer:
    def __init__(self, name, csp='azure', embeddings_client=None, stt_client=None):
        self.app = Flask(name)

        if csp == 'azure':
            self.csp = AzureCSP(embeddings_client=embeddings_client, stt_client=stt_client)
        else:
            raise NotImplemented(f"Cloud Service Provide {csp} not implemented")

        self.setup_routes()

    def setup_routes(self):

        @self.app.route('/api/Support/IndexData', methods=['POST'])
        def support_index_data():
            index_name = request.args.get('index_name')
            file_path = request.args.get('file_path')

            for item, ty in zip([index_name, file_path], [str, str]):
                if not isinstance(item, ty):
                    return jsonify({'errors': f'{item} must be of type {ty.__name__}'}), 400

            self.csp.index_data(index_name, file_path)

            return jsonify({'message': 'Success'}), 200

        @self.app.route('/api/Support/SimpleHS', methods=['GET'])
        def support_simple_hs():
            index_name = request.args.get('index_name')
            prompt = request.args.get('prompt')

            for item, ty in zip([index_name, prompt], [str, str]):
                if not isinstance(item, ty):
                    return jsonify({'errors': f'{item} must be of type {ty.__name__}'}), 400

            self.csp.simple_hs(prompt, index_name)

            return jsonify({'message': 'Success'}), 200

        @self.app.route('/api/Support/StartConversation', methods=['POST'])
        # Start Conversation
        def start_conversation():
            name = request.args.get('name')
            prompt = request.args.get('prompt')

            for item, ty in zip([name, prompt], [str, str]):
                if not isinstance(item, ty):
                    return jsonify({'errors': f'{item} must be of type {ty.__name__}'}), 400
                
            

            self.csp.start_conversation()

            return jsonify({'message': 'Success'}), 200

        @self.app.route('/api/Support/SpeechToText', methods=['POST'])
        def speech_to_text():
            speech = request.args.get('speech')

            for item, ty in zip([speech], [str]):
                if not isinstance(item, ty):
                    return jsonify({'errors': f'{item} must be of type {ty.__name__}'}), 400

            self.csp.speech_to_text(speech)

            return jsonify({'message': 'Success'}), 200


    def run(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    server = HackathonServer("Google hackathon")
    app = server.app
    server.run()
