import argparse
from hackathon.server import HackathonServer

def main():
    parser = argparse.ArgumentParser(description="Run the Flask Server for the hackathon project.")
    parser.add_argument("--name", type=str, default="Hackathon Server",
                        help="Name of the Flask application.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="The hostname to listen on. Set this to '0.0.0.0' to have the server available externally.")
    parser.add_argument("--port", type=int, default=5000,
                        help="The port of the webserver.")
    parser.add_argument("--csp", type=str, default="azure", required=False,
                        help="The cloud service provider to use, e.g., 'azure'.")
    parser.add_argument("--embeddingsclient", type=str, default=None, required=False,
                        help="The embeddings client to be used by the CSP.")
    parser.add_argument("--sttclient", type=str, default=None, required=False,
                        help="The speech-to-text client to be used by the CSP.")


    args = parser.parse_args()

    server = HackathonServer(args.name, args.csp)
    server.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
