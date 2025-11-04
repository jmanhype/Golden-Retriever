"""
Golden-Retriever: High-Fidelity Agentic Retrieval Augmented Generation

This module implements a sophisticated RAG framework for technical question-answering
in specialized domains. It combines jargon identification, context recognition, and
question augmentation to provide accurate, well-reasoned answers.

Key Components:
    - QueryJargonDictionary: Retrieves technical term definitions from multiple sources
    - ImprovedAnswerGenerator: Generates comprehensive answers with chain-of-thought reasoning
    - GoldenRetrieverRAG: Main orchestrator integrating all components

Usage:
    from app import GoldenRetrieverRAG
    import dspy

    # Initialize and configure
    rag = GoldenRetrieverRAG()
    rag.identify_jargon = dspy.Predict("question -> jargon_terms")
    rag.identify_context = dspy.Predict("question -> context")
    rag.augment_question = dspy.ChainOfThought("question, jargon_definitions, context -> augmented_question")
    rag.generate_answer = ImprovedAnswerGenerator()

    # Ask a question
    result = rag("What is the role of wear leveling in SSDs?")
    print(result.answer)
"""

import dspy
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from cachetools import TTLCache
import logging
import json
import random
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy import ColBERTv2
import backoff
import nest_asyncio
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple, Optional, Any

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables and setup logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Validate required environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY environment variable is not set. "
        "Please create a .env file with your OpenAI API key: OPENAI_API_KEY=your_key_here"
    )

# Configure DSPy
llm = dspy.OpenAI(
    model='gpt-3.5-turbo',
    api_key=OPENAI_API_KEY,
    max_tokens=2000
)
dspy.settings.configure(lm=llm)

# Initialize ColBERTv2 retriever
retriever = ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=retriever)

class QueryJargonDictionary(dspy.Module):
    """
    A module for retrieving jargon definitions from multiple sources.

    This class queries local dictionaries, Wikipedia, and GPT-3 to provide
    comprehensive definitions for technical terms. It includes caching and
    rate limiting to optimize API usage.

    Attributes:
        cache: TTL cache for storing retrieved definitions
        rate_limit: Rate limit delay in seconds between API calls
        local_dictionary: Pre-defined technical terms and their definitions
    """

    def __init__(self) -> None:
        """Initialize the jargon dictionary with cache and local definitions."""
        super().__init__()
        self.cache: TTLCache = TTLCache(maxsize=1000, ttl=3600)
        self.rate_limit: float = 1.0
        self.local_dictionary: Dict[str, str] = {
            # ... [previous dictionary entries remain unchanged] ...
            "Wear leveling": "A technique used in SSDs to distribute write operations evenly across all the flash memory blocks, extending the lifespan of the drive by preventing premature wear-out of specific areas.",
            "SSDs": "Solid State Drives, storage devices that use integrated circuit assemblies to store data persistently, offering faster access times and improved reliability compared to traditional hard disk drives.",
            "Traditional storage interfaces": "Conventional methods of connecting storage devices to computers, such as SATA (Serial ATA) or SAS (Serial Attached SCSI), which are generally slower and less efficient than newer interfaces like NVMe.",
        }

    async def forward(self, jargon_terms: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Retrieve definitions for multiple jargon terms concurrently.

        Args:
            jargon_terms: List of technical terms to define

        Returns:
            Dictionary mapping terms to their definitions from various sources
        """
        jargon_definitions: Dict[str, Dict[str, str]] = {}

        async with aiohttp.ClientSession() as session:
            tasks = [self.get_jargon_definition(term, session) for term in jargon_terms]
            results = await asyncio.gather(*tasks)

        for term, definitions in results:
            jargon_definitions[term] = definitions

        return jargon_definitions

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def get_jargon_definition(
        self, term: str, session: aiohttp.ClientSession
    ) -> Tuple[str, Dict[str, str]]:
        """
        Get definition for a single jargon term from cache or external sources.

        Args:
            term: The technical term to define
            session: Aiohttp client session for making HTTP requests

        Returns:
            Tuple of (term, definitions_dict) where definitions_dict maps
            source names to definition strings
        """
        if term in self.cache:
            return term, self.cache[term]

        logging.info(f"Querying for term: {term}")

        # Check local dictionary first
        if term.lower() in self.local_dictionary:
            self.cache[term] = {"local": self.local_dictionary[term.lower()]}
            return term, self.cache[term]

        definitions: Dict[str, str] = {
            "wikipedia": await self.query_wikipedia(term, session),
        }

        # Remove None values
        definitions = {k: v for k, v in definitions.items() if v is not None}

        if not definitions:
            # Use GPT-3 as a fallback for definition
            gpt_def = await self.query_gpt(term)
            if gpt_def:
                definitions["gpt"] = gpt_def

        self.cache[term] = definitions
        return term, definitions

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def query_wikipedia(
        self, term: str, session: aiohttp.ClientSession
    ) -> Optional[str]:
        """
        Query Wikipedia API for term definition.

        Args:
            term: The term to look up
            session: Aiohttp client session

        Returns:
            Wikipedia extract text or None if not found
        """
        try:
            await asyncio.sleep(self.rate_limit)  # Rate limiting
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{term}"
            async with session.get(url, headers={"User-Agent": "GoldenRetrieverBot/1.0"}) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('extract')
                else:
                    logging.warning(f"Wikipedia returned status {response.status} for term {term}")
        except aiohttp.ClientError as e:
            logging.error(f"Network error querying Wikipedia for {term}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error querying Wikipedia for {term}: {e}")
        return None

    async def query_gpt(self, term: str) -> Optional[str]:
        """
        Query GPT-3 for term definition as fallback when other sources fail.

        Args:
            term: The term to define

        Returns:
            GPT-3 generated definition or None if all retries fail
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = f"Provide a brief definition for the term '{term}' in the context of computer storage technology:"
                response = dspy.Predict("term -> definition")(term=prompt).definition
                return response.strip()
            except Exception as e:
                logging.warning(f"Error querying GPT for {term} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logging.error(f"Failed to query GPT for {term} after {max_retries} attempts")
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

class ImprovedAnswerGenerator(dspy.Module):
    """
    Generates comprehensive answers using chain-of-thought reasoning.

    This module takes the original question, augmented question, jargon definitions,
    context, and retrieved passages to generate a well-reasoned answer.
    """

    def __init__(self) -> None:
        """Initialize the answer generator with chain-of-thought module."""
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(
            "original_question, augmented_question, jargon_definitions, context, retrieved_passages -> reasoning, comprehensive_answer"
        )

    def forward(
        self,
        original_question: str,
        augmented_question: str,
        jargon_definitions: str,
        context: str,
        retrieved_passages: str
    ) -> Tuple[str, str]:
        """
        Generate a comprehensive answer with reasoning.

        Args:
            original_question: The user's original question
            augmented_question: Enhanced question with context
            jargon_definitions: JSON string of term definitions
            context: Domain context for the question
            retrieved_passages: JSON string of relevant passages

        Returns:
            Tuple of (reasoning, comprehensive_answer)
        """
        result = self.generate_answer(
            original_question=original_question,
            augmented_question=augmented_question,
            jargon_definitions=jargon_definitions,
            context=context,
            retrieved_passages=retrieved_passages
        )
        return result.reasoning, result.comprehensive_answer

class GoldenRetrieverRAG(dspy.Module):
    """
    Main RAG (Retrieval Augmented Generation) framework for technical Q&A.

    This class orchestrates jargon identification, context recognition, question
    augmentation, passage retrieval, and answer generation to provide high-fidelity
    responses to domain-specific questions.

    Attributes:
        query_jargon_dictionary: Module for retrieving term definitions
        retrieve: DSPy retriever for finding relevant passages
        identify_jargon: Module for identifying technical terms (set externally)
        identify_context: Module for recognizing domain context (set externally)
        augment_question: Module for enhancing questions (set externally)
        generate_answer: Module for creating answers (set externally)
    """

    def __init__(self, num_passages: int = 5) -> None:
        """
        Initialize the Golden Retriever RAG framework.

        Args:
            num_passages: Number of passages to retrieve for context (default: 5)
        """
        super().__init__()
        self.query_jargon_dictionary = QueryJargonDictionary()
        self.retrieve = dspy.Retrieve(k=num_passages)

        # Initialize these as None, they will be set later
        self.identify_jargon: Optional[Any] = None
        self.identify_context: Optional[Any] = None
        self.augment_question: Optional[Any] = None
        self.generate_answer: Optional[Any] = None

    async def forward(self, question: str) -> dspy.Prediction:
        """
        Process a question through the complete RAG pipeline.

        Args:
            question: User's technical question

        Returns:
            DSPy Prediction containing the complete analysis and answer

        Raises:
            ValueError: If required modules have not been initialized
        """
        if not all([self.identify_jargon, self.identify_context, self.augment_question, self.generate_answer]):
            raise ValueError(
                "Not all required modules have been set. Please configure: "
                "identify_jargon, identify_context, augment_question, and generate_answer"
            )

        jargon_terms = self.identify_jargon(question=question).jargon_terms.strip().split(',')
        jargon_terms = [term.strip() for term in jargon_terms if len(term.strip().split()) <= 3]  # Limit to terms with 3 words or less
        jargon_definitions = await self.query_jargon_dictionary(jargon_terms)
        context = self.identify_context(question=question).context.strip()

        augmented_question = self.augment_question(
            question=question,
            jargon_definitions=json.dumps(jargon_definitions),
            context=context
        ).augmented_question.strip()

        retrieved_passages = self.retrieve(augmented_question).passages

        reasoning, answer = self.generate_answer(
            original_question=question,
            augmented_question=augmented_question,
            jargon_definitions=json.dumps(jargon_definitions),
            context=context,
            retrieved_passages=json.dumps(retrieved_passages)
        )

        return dspy.Prediction(
            original_question=question,
            augmented_question=augmented_question,
            jargon_definitions=jargon_definitions,
            context=context,
            reasoning=reasoning,
            answer=answer,
            retrieved_passages=retrieved_passages
        )

    def __call__(self, question: str) -> dspy.Prediction:
        """
        Synchronous wrapper for forward method.

        Args:
            question: User's technical question

        Returns:
            DSPy Prediction with the complete response
        """
        return asyncio.run(self.forward(question))

def generate_and_load_trainset(num_examples: int = 20) -> List[dspy.Example]:
    """
    Generate a synthetic training dataset for the RAG model.

    Args:
        num_examples: Number of training examples to generate (default: 20)

    Returns:
        List of DSPy Example objects with question-answer pairs
    """
    questions = [
        "What is Flash Translation Layer (FTL) in computer storage technology?",
        "How does Error Correction Code (ECC) work in data storage?",
        "What are the advantages of NVMe over traditional storage interfaces?",
        "Explain the concept of wear leveling in SSDs.",
        "What is the difference between NOR and NAND flash memory?",
        "How does TRIM command improve SSD performance?",
        "What is the role of a controller in an SSD?",
        "Explain the concept of garbage collection in SSDs.",
        "What is over-provisioning in SSDs and why is it important?",
        "How does QLC NAND differ from TLC NAND?",
    ]
    
    answers = [
        "FTL is a layer that translates logical block addresses to physical addresses in flash memory, managing wear leveling and garbage collection.",
        "ECC detects and corrects errors in data storage by adding redundant bits, improving data reliability.",
        "NVMe offers lower latency, higher throughput, and more efficient queuing than traditional interfaces like SATA.",
        "Wear leveling distributes write operations evenly across all blocks of an SSD, preventing premature wear-out of specific areas.",
        "NOR flash allows random access to any memory location, while NAND flash reads and writes data in blocks, offering higher density.",
        "TRIM informs the SSD which blocks of data are no longer in use, improving garbage collection and write performance.",
        "An SSD controller manages data transfer between the computer and flash memory chips, handling tasks like wear leveling and error correction.",
        "Garbage collection in SSDs consolidates valid data and erases invalid data blocks, freeing up space for new writes.",
        "Over-provisioning reserves extra space in an SSD, improving performance, endurance, and allowing for more efficient garbage collection.",
        "QLC NAND stores 4 bits per cell, offering higher capacity but lower endurance compared to TLC NAND, which stores 3 bits per cell.",
    ]
    
    trainset = []
    for _ in range(num_examples):
        idx = random.randint(0, len(questions) - 1)
        example = dspy.Example(question=questions[idx], answer=answers[idx])
        trainset.append(example.with_inputs('question'))  # Specify 'question' as input
    
    return trainset

def improved_answer_evaluation(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional[Any] = None,
    frac: float = 0.5
) -> bool:
    """
    Evaluate answer quality using ROUGE and semantic similarity metrics.

    Args:
        example: Ground truth example with expected answer
        pred: Model prediction containing generated answer
        trace: Optional trace information (unused)
        frac: Threshold for combined score (default: 0.5)

    Returns:
        True if combined score meets threshold, False otherwise
    """
    rouge = Rouge()
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def normalize_text(text: str) -> str:
        """Normalize text by lowercasing and removing extra whitespace."""
        return ' '.join(text.lower().split())

    def calculate_rouge(prediction: str, ground_truth: str) -> float:
        """Calculate ROUGE-L F1 score."""
        scores = rouge.get_scores(prediction, ground_truth)
        return scores[0]['rouge-l']['f']

    def calculate_semantic_similarity(prediction: str, ground_truth: str) -> float:
        """Calculate cosine similarity between semantic embeddings."""
        embeddings1 = model.encode([prediction], convert_to_tensor=True)
        embeddings2 = model.encode([ground_truth], convert_to_tensor=True)
        return util.pytorch_cos_sim(embeddings1, embeddings2).item()

    prediction = normalize_text(pred.answer)
    ground_truth = normalize_text(example.answer)

    rouge_score = calculate_rouge(prediction, ground_truth)
    semantic_similarity = calculate_semantic_similarity(prediction, ground_truth)

    combined_score = (rouge_score + semantic_similarity) / 2

    return combined_score >= frac

async def async_evaluate(compiled_rag: GoldenRetrieverRAG, devset: List[dspy.Example]) -> float:
    """
    Asynchronously evaluate RAG model on development set.

    Args:
        compiled_rag: Compiled RAG model to evaluate
        devset: List of evaluation examples

    Returns:
        Average evaluation score across all examples
    """
    results: List[bool] = []
    for example in devset:
        pred = await compiled_rag.forward(example.question)
        score = improved_answer_evaluation(example, pred)
        results.append(score)
    return sum(results) / len(results) if results else 0.0

def evaluate(compiled_rag: GoldenRetrieverRAG, devset: List[dspy.Example]) -> float:
    """
    Synchronous wrapper for async_evaluate.

    Args:
        compiled_rag: Compiled RAG model to evaluate
        devset: List of evaluation examples

    Returns:
        Average evaluation score
    """
    return asyncio.run(async_evaluate(compiled_rag, devset))

# Run the main event loop
if __name__ == "__main__":
    # Setup and compilation
    dataset = generate_and_load_trainset()
    trainset = dataset[:-5]  # Use all but last 5 examples as train set
    devset = dataset[-5:]  # Use last 5 examples as dev set

    # Define the modules
    modules = [
        ("identify_jargon", dspy.Predict("question -> jargon_terms")),
        ("identify_context", dspy.Predict("question -> context")),
        ("augment_question", dspy.ChainOfThought("question, jargon_definitions, context -> augmented_question")),
        ("generate_answer", ImprovedAnswerGenerator())
    ]

    # Create a new GoldenRetrieverRAG instance
    rag_instance = GoldenRetrieverRAG()

    # Set the modules
    for name, module in modules:
        setattr(rag_instance, name, module)

    # Set instructions separately
    rag_instance.identify_jargon.instructions = "Identify technical jargon or abbreviations in the following question. Output only individual terms or short phrases, separated by commas."
    rag_instance.identify_context.instructions = "Identify the relevant context or domain for the given question."
    rag_instance.augment_question.instructions = "Given the original question, jargon definitions, and context, create an augmented version of the question that incorporates this additional information."
    rag_instance.generate_answer.generate_answer.instructions = """
    Given the original question, augmented question, jargon definitions, context, and retrieved passages:
    1. Analyze the question and identify the key concepts and requirements.
    2. Review the jargon definitions and context to understand the specific domain knowledge needed.
    3. Examine the retrieved passages and extract relevant information.
    4. Reason step-by-step about how to construct a comprehensive answer.
    5. Synthesize the information into a clear, concise, and accurate answer.
    6. Ensure the answer directly addresses the original question and incorporates relevant jargon and context.
    7. Provide your step-by-step reasoning in the 'reasoning' output.
    8. Provide your final comprehensive answer in the 'comprehensive_answer' output.
    """

    teleprompter = BootstrapFewShotWithRandomSearch(
        metric=improved_answer_evaluation,
        num_candidate_programs=10,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=2,
        num_threads=1,  # Set this to 1 to avoid multi-threading issues
        max_errors=10
    )

    try:
        compiled_rag = teleprompter.compile(rag_instance, trainset=trainset, valset=devset)
    except Exception as e:
        logging.error(f"Error during compilation: {e}")
        compiled_rag = rag_instance

    # Save the compiled program
    compiled_program_json = compiled_rag.save("compiled_goldenretriever_rag.json")
    print("Program saved to compiled_goldenretriever_rag.json")

    # Evaluate the compiled program
    try:
        results = evaluate(compiled_rag, devset)
        print("Evaluation Results:")
        print(results)
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        print("An error occurred during evaluation. Please check the logs for details.")

    # Interactive loop
    while True:
        question = input("Enter a question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        try:
            prediction = asyncio.run(compiled_rag.forward(question))
            print(f"Original Question: {prediction.original_question}")
            print(f"Augmented Question: {prediction.augmented_question}")
            print(f"Identified Jargon Terms:")
            for term, definitions in prediction.jargon_definitions.items():
                print(f"  - {term}:")
                for source, definition in definitions.items():
                    print(f"    {source}: {definition}")
            print(f"Identified Context: {prediction.context}")
            print(f"Reasoning:")
            print(prediction.reasoning)
            print(f"Answer: {prediction.answer}")
            print("Retrieved Passages:")
            for i, passage in enumerate(prediction.retrieved_passages, 1):
                print(f"Passage {i}: {passage[:200]}...")  # Print first 200 characters of each passage
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            print("An error occurred while processing the question. Please try again.")

    print("Thank you for using GoldenRetrieverRAG. Goodbye!")
