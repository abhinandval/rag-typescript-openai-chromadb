# Simple RAG chat bot using OpenAI and Pinecone Vector DB with langChain Project
 
Follow the instructions below to get started.

## Prerequisites

Before you begin, make sure you have the following installed:

1. **Node.js**: You'll need Node.js installed on your system. If you don't have it, download and install it from here.

2. **Yarn**: Yarn is a package manager for Node.js. Install it globally by running:

    ```bash
    npm install -g yarn
    ```

## Setup

1. Clone this repository:

    ```bash
    git clone https://github.com/your-username/my-awesome-node-project.git
    cd my-awesome-node-project
    ```

2. Copy the `.env.sample` file to `.env` and replace the placeholder values with your actual configuration. This file contains environment variables needed for your application.

3. Install project dependencies:

    ```bash
    yarn install
    ```

## Usage

1. Create a vector database index by running:

    ```bash
    yarn indexDocuments
    ```

   This command will create an index for your documents in the database.

2. Start querying your data using:

    ```bash
    yarn makeQuery
    ```

   Customize the query logic according to your project requirements.

## Contributing

Feel free to contribute to this project by opening pull requests or reporting issues. Let's make this project even more awesome together!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
