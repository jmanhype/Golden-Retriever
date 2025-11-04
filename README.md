# Golden-Retriever: A Framework for High-Fidelity Agentic Retrieval Augmented Generation 

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

Create a `.env` file in the project root with your OpenAI API key:

```bash
# .env file
OPENAI_API_KEY=your_api_key_here
```

**Important:** The OPENAI_API_KEY environment variable is required. The application will display a helpful error message if it's not set.

## Usage

### Basic Usage

```python
import dspy
from app import GoldenRetrieverRAG, ImprovedAnswerGenerator

# Initialize the framework
rag = GoldenRetrieverRAG(num_passages=5)

# Set up the necessary modules
rag.identify_jargon = dspy.Predict("question -> jargon_terms")
rag.identify_context = dspy.Predict("question -> context")
rag.augment_question = dspy.ChainOfThought("question, jargon_definitions, context -> augmented_question")
rag.generate_answer = ImprovedAnswerGenerator()

# Ask a question
question = "What is the role of wear leveling in SSDs?"
result = rag(question)

# Access different parts of the response
print(f"Answer: {result.answer}")
print(f"Reasoning: {result.reasoning}")
print(f"Context: {result.context}")
print(f"Jargon Definitions: {result.jargon_definitions}")
```

### Advanced Usage with Training

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from app import generate_and_load_trainset, improved_answer_evaluation

# Generate training data
dataset = generate_and_load_trainset(num_examples=20)
trainset = dataset[:-5]
devset = dataset[-5:]

# Configure teleprompter
teleprompter = BootstrapFewShotWithRandomSearch(
    metric=improved_answer_evaluation,
    num_candidate_programs=10,
    max_bootstrapped_demos=4
)

# Compile the RAG instance
compiled_rag = teleprompter.compile(rag, trainset=trainset, valset=devset)

# Use the compiled version
result = compiled_rag(question)
print(result.answer)
```

## Training and Evaluation

The framework includes built-in functionality for:

- **Generating synthetic training data** from predefined Q&A pairs
- **Compiling the RAG model** using DSPy's teleprompter
- **Evaluating performance** using ROUGE and semantic similarity metrics

The evaluation combines ROUGE-L scores with semantic similarity (using sentence transformers) to provide a comprehensive quality assessment.

## Interactive Mode

Run the main script to enter an interactive mode:

```bash
python app.py
```

In interactive mode, you can:
- Ask technical questions about storage technology
- View jargon definitions from multiple sources (local dictionary, Wikipedia, GPT-3)
- See the reasoning process behind each answer
- Review retrieved passages used for context
- Type 'quit' to exit

## Architecture

### Components

1. **QueryJargonDictionary**: Multi-source term definition retrieval
   - Local dictionary for common terms
   - Wikipedia API integration
   - GPT-3 fallback for undefined terms
   - TTL caching for performance

2. **ImprovedAnswerGenerator**: Chain-of-thought answer generation
   - Integrates question context, jargon, and retrieved passages
   - Provides step-by-step reasoning
   - Generates comprehensive, accurate answers

3. **GoldenRetrieverRAG**: Main pipeline orchestrator
   - Jargon identification
   - Context recognition
   - Question augmentation
   - Passage retrieval (ColBERT)
   - Answer generation

### Pipeline Flow

```
User Question
    ↓
Identify Jargon Terms
    ↓
Retrieve Definitions (Local/Wikipedia/GPT)
    ↓
Identify Context
    ↓
Augment Question
    ↓
Retrieve Relevant Passages (ColBERT)
    ↓
Generate Answer with Reasoning
    ↓
Return Complete Response
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This implementation is based on the DSPy library and the concepts from the paper "Golden-Retriever: High-Fidelity Agentic Retrieval Augmented Generation for Industrial Knowledge Base".
