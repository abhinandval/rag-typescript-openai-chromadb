# Simple RAG chat bot using OpenAI and Pinecone Vector DB with langChain Project
 
Follow the instructions below to get started.

## Prerequisites

Before you begin, make sure you have the following installed:

1. **Node.js**: You'll need Node.js installed on your system. If you don't have it, download and install it from here.

2. **Yarn**: Yarn is a package manager for Node.js. Install it globally by running:

    ```bash
    npm install -g yarn
    ```
3. **Python**: This project requires Python version 3.8 to 3.11. You can download the appropriate installer from [https://www.python.org/downloads/](https://www.python.org/downloads/). Verify your installation by running `python --version`.

## Setup

1. Clone this repository:

    ```bash
    git clone https://github.com/your-username/my-awesome-node-project.git
    cd my-awesome-node-project
    ```

2. Copy the `.env.sample` file to `.env` and replace the placeholder values with your actual configuration. This file contains environment variables needed for your application.

   **NOTE** goto [https://ai.google.dev/gemini-api/docs/api-key] to get your Google API key and replace the placeholder values in `.env` file with them.

3. Install project dependencies:

    ```bash
    yarn install
    ```
4. Install ChromaDB
   ```bash 
   pip install chromadb
   ```

## Usage
1. Start ChromaDB using the following command:
   ```bash
   chroma run --path chroma 
   ```

2. Create a vector database index by running:

    ```bash
    yarn indexDocuments
    ```

   This command will create an index for your documents in the database.

3. Start querying your data using:

    ```bash
    yarn makeQuery
    ```

   Customize the query logic according to your project requirements.

## Contributing

Feel free to contribute to this project by opening pull requests or reporting issues. Let's make this project even more awesome together!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
