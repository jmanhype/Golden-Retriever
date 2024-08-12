import dspy
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from cachetools import TTLCache
import logging
import json
import wikipedia
import time
import random
import requests
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate.evaluate import Evaluate
from dspy import ColBERTv2
import backoff
import nest_asyncio
import copy
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables and setup logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure DSPy
llm = dspy.OpenAI(
    model='gpt-3.5-turbo',
    api_key=os.environ['OPENAI_API_KEY'],
    max_tokens=2000
)
dspy.settings.configure(lm=llm)

# Initialize ColBERTv2 retriever
retriever = ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=retriever)

class QueryJargonDictionary(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        self.rate_limit = 1.0
        self.local_dictionary = {
            # ... [previous dictionary entries remain unchanged] ...
            "Wear leveling": "A technique used in SSDs to distribute write operations evenly across all the flash memory blocks, extending the lifespan of the drive by preventing premature wear-out of specific areas.",
            "SSDs": "Solid State Drives, storage devices that use integrated circuit assemblies to store data persistently, offering faster access times and improved reliability compared to traditional hard disk drives.",
            "Traditional storage interfaces": "Conventional methods of connecting storage devices to computers, such as SATA (Serial ATA) or SAS (Serial Attached SCSI), which are generally slower and less efficient than newer interfaces like NVMe.",
        }

    async def forward(self, jargon_terms):
        jargon_definitions = {}

        async with aiohttp.ClientSession() as session:
            tasks = [self.get_jargon_definition(term, session) for term in jargon_terms]
            results = await asyncio.gather(*tasks)

        for term, definitions in results:
            jargon_definitions[term] = definitions

        return jargon_definitions

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def get_jargon_definition(self, term, session):
        if term in self.cache:
            return term, self.cache[term]

        logging.info(f"Querying for term: {term}")
        
        # Check local dictionary first
        if term.lower() in self.local_dictionary:
            self.cache[term] = {"local": self.local_dictionary[term.lower()]}
            return term, self.cache[term]

        definitions = {
            "wikipedia": await self.query_wikipedia(term, session),
        }

        # Remove None values
        definitions = {k: v for k, v in definitions.items() if v is not None}

        if not definitions:
            # Use GPT-3 as a fallback for definition
            definitions["gpt"] = await self.query_gpt(term)

        self.cache[term] = definitions
        return term, definitions

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def query_wikipedia(self, term, session):
        try:
            await asyncio.sleep(self.rate_limit)  # Rate limiting
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{term}"
            async with session.get(url, headers={"User-Agent": "GoldenRetrieverBot/1.0"}) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('extract')
                else:
                    logging.warning(f"Wikipedia returned status {response.status} for term {term}")
        except Exception as e:
            logging.error(f"Error querying Wikipedia for {term}: {e}")
        return None

    async def query_gpt(self, term):
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
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("original_question, augmented_question, jargon_definitions, context, retrieved_passages -> reasoning, comprehensive_answer")

    def forward(self, original_question, augmented_question, jargon_definitions, context, retrieved_passages):
        result = self.generate_answer(
            original_question=original_question,
            augmented_question=augmented_question,
            jargon_definitions=jargon_definitions,
            context=context,
            retrieved_passages=retrieved_passages
        )
        return result.reasoning, result.comprehensive_answer

class GoldenRetrieverRAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.query_jargon_dictionary = QueryJargonDictionary()
        self.retrieve = dspy.Retrieve(k=num_passages)
        
        # Initialize these as None, they will be set later
        self.identify_jargon = None
        self.identify_context = None
        self.augment_question = None
        self.generate_answer = None

    async def forward(self, question):
        if not all([self.identify_jargon, self.identify_context, self.augment_question, self.generate_answer]):
            raise ValueError("Not all required modules have been set.")

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

    def __call__(self, question):
        return asyncio.run(self.forward(question))

def generate_and_load_trainset(num_examples=20):
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

def improved_answer_evaluation(example, pred, trace=None, frac=0.5):
    rouge = Rouge()
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def normalize_text(text):
        return ' '.join(text.lower().split())

    def calculate_rouge(prediction, ground_truth):
        scores = rouge.get_scores(prediction, ground_truth)
        return scores[0]['rouge-l']['f']

    def calculate_semantic_similarity(prediction, ground_truth):
        embeddings1 = model.encode([prediction], convert_to_tensor=True)
        embeddings2 = model.encode([ground_truth], convert_to_tensor=True)
        return util.pytorch_cos_sim(embeddings1, embeddings2).item()

    prediction = normalize_text(pred.answer)
    ground_truth = normalize_text(example.answer)

    rouge_score = calculate_rouge(prediction, ground_truth)
    semantic_similarity = calculate_semantic_similarity(prediction, ground_truth)

    combined_score = (rouge_score + semantic_similarity) / 2

    return combined_score >= frac

async def async_evaluate(compiled_rag, devset):
    results = []
    for example in devset:
        pred = await compiled_rag.forward(example.question)
        score = improved_answer_evaluation(example, pred)
        results.append(score)
    return sum(results) / len(results)

def evaluate(compiled_rag, devset):
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
