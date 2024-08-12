# Golden-Retriever

Golden-Retriever is a framework for high-fidelity retrieval augmented generation in industrial knowledge bases. It integrates jargon identification, context recognition, and question augmentation to overcome challenges in specialized domains.

## Features

- Jargon identification and definition retrieval
- Context recognition for domain-specific questions
- Dynamic question augmentation
- Retrieval-augmented generation using DSPy
- Adaptive answer generation with reasoning
- Extensible and customizable architecture

## Installation

```bash
git clone https://github.com/yourusername/golden-retriever.git
cd golden-retriever
pip install -r requirements.txt
```

## Configuration

Set your OpenAI API key in a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

```python
from golden_retriever import GoldenRetrieverRAG

# Initialize the framework
rag = GoldenRetrieverRAG()

# Set up the necessary modules
rag.identify_jargon = dspy.Predict("question -> jargon_terms")
rag.identify_context = dspy.Predict("question -> context")
rag.augment_question = dspy.ChainOfThought("question, jargon_definitions, context -> augmented_question")
rag.generate_answer = ImprovedAnswerGenerator()

# Compile the RAG instance (optional)
compiled_rag = teleprompter.compile(rag, trainset=trainset, valset=devset)

# Ask a question
question = "What is the role of wear leveling in SSDs?"
result = compiled_rag(question)

print(result.answer)
```

## Training and Evaluation

The framework includes functionality for generating training data, compiling the RAG instance using teleprompter, and evaluating the model's performance.

## Interactive Mode

Run the script to enter an interactive mode where you can ask questions and receive detailed responses, including jargon definitions, context, reasoning, and retrieved passages.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This implementation is based on the DSPy library and the concepts from the paper "Golden-Retriever: High-Fidelity Agentic Retrieval Augmented Generation for Industrial Knowledge Base".
