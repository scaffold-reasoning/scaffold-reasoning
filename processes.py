from chain_of_agents import ChainOfAgents
from chain_of_agents.utils import split_into_chunks, split_into_chunks_tokenwise
# from chain_of_agents.gpt_agents import WorkerAgent, ManagerAgent, ActionAgent, SingleAgent, ReactAgent, ReviewAgent, ExtractAgent
from chain_of_agents.debug_agents import WorkerAgent, ManagerAgent, ActionAgent, SingleAgent, ReactAgent, ReviewAgent, ExtractAgent
import concurrent.futures
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import tiktoken

class CoAProcessRunner:
    """
    Wraps multiple versions of Chain of Agents pipelines.
    """

    def __init__(self, coa: ChainOfAgents):
        self.coa = coa  # holds your config, chunk size, models, etc.

        # Load NLP and summarizer once when you initialize the runner
        # self.nlp = spacy.load("en_core_web_sm")

        # # Use LED for longer input context window
        # self.model_name = "allenai/led-base-16384"
        # self.led_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        # self.summarizer = pipeline("summarization", model=model, tokenizer=self.led_tokenizer)
        # self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def single(self, query, code, bug_type,  explanation):
        """
        Standard process: Single Agent
        """
        manager = SingleAgent(self.coa.manager_model, self.coa.single_prompt)
        final_output = manager.synthesize(query, code, bug_type,  explanation)

        return final_output, final_output

    def debug_react(self, query, code, bug_type, explanation):
        """
        Standard process: Single Agent
        """
        manager = ReactAgent(self.coa.manager_model, self.coa.react_prompt)
        final_output = manager.synthesize(query, code, bug_type,  explanation)

        return final_output, final_output

    def single_react(self, input_text, query, answer):
        """
        Standard process: Single Agent
        """
        import time
        from openai._exceptions import RateLimitError, APIError

        manager = ReactAgent(self.coa.manager_model)
        for attempt in range(4):  # Retry up to 5 times
            try:
                predict = manager.synthesize(input_text, query)               
                break  # success, break the retry loop
            except RateLimitError:
                print("⚠️ Rate limit hit. Retrying in 5 second...")
                time.sleep(5)
            except APIError as e:
                print("❌ API error:", e)
                break

        # checker = ReviewAgent(self.coa.manager_model)
        # for attempt in range(10):  # Retry up to 5 times
        #     try:
        #         output = checker.synthesize(predict, answer)   
        #         break         
        #     except RateLimitError:
        #         print("⚠️ Rate limit hit. Retrying in 1 second...")
        #         time.sleep(1)
        #     except APIError as e:
        #         print("❌ API error:", e)
        #         break
        
        # checker = ExtractAgent(self.coa.manager_model)
        # for attempt in range(10):  # Retry up to 5 times
        #     try:
        #         output = checker.synthesize(predict)   
        #         break         
        #     except RateLimitError:
        #         print("⚠️ Rate limit hit. Retrying in 1 second...")
        #         time.sleep(1)
        #     except APIError as e:
        #         print("❌ API error:", e)
        #         break

        return predict, predict # Return both final output and answer

    def single_with_action(self, input_text, query):
        """
        Standard process: Single Agent
        """
        action = ActionAgent(self.coa.action_model, self.coa.action_prompt)
        action_output = action.analyze_question(query)
        query = f"Question: {query}\nStep to reach the answer: {action_output}"
        manager = SingleAgent(self.coa.manager_model, self.coa.single_prompt)
        final_output = manager.synthesize(input_text, query)

        return final_output, final_output

    def base_process(self, input_text, query):
        """
        Standard process: Worker ➜ Manager
        """
        chunks = split_into_chunks(input_text, self.coa.chunk_size, self.coa.worker_model)
        worker_outputs = []
        previous_cu = None

        for chunk in chunks:
            worker = WorkerAgent(self.coa.worker_model, self.coa.worker_prompt, self.coa.max_new_tokens)
            output = worker.process_chunk(chunk, query, previous_cu)
            worker_outputs.append(output)
            previous_cu = output

        manager = ManagerAgent(self.coa.manager_model, self.coa.manager_prompt)
        final_output = manager.synthesize(worker_outputs, query)

        return final_output, worker_outputs

    def with_action(self, input_text, query):
        """
        Standard process: Worker ➜ Manager
        """
        action = ActionAgent(self.coa.action_model, self.coa.action_prompt)
        action_output = action.analyze_question(query)
        query = f"Question: {query}\nStep to reach the answer: {action_output}"
        chunks = split_into_chunks(input_text, self.coa.chunk_size, self.coa.worker_model)
        worker_outputs = []
        previous_cu = None

        for chunk in chunks:
            worker = WorkerAgent(self.coa.worker_model, self.coa.worker_prompt)
            output = worker.process_chunk(chunk, query, previous_cu)
            worker_outputs.append(output)
            previous_cu = output

        manager = ManagerAgent(self.coa.manager_model, self.coa.manager_prompt)
        final_output = manager.synthesize(worker_outputs, query)

        return final_output, worker_outputs

    def with_critic(self, input_text, query):
        """
        Process with Critic Agent inserted: Worker ➜ Critic ➜ Manager
        """
        chunks = split_into_chunks(input_text, self.coa.chunk_size, self.coa.worker_model)
        worker_outputs = []
        previous_cu = None

        for chunk in chunks:
            # chunk = self.mark_article_elements(chunk)  # Mark entities and conclusions
            worker = WorkerAgent(self.coa.worker_model, self.coa.worker_prompt)
            worker_output = worker.process_chunk(chunk, query, previous_cu)

            critic = CriticAgent(self.coa.critic_model, self.coa.critic_prompt)
            critic_output = critic.critique([worker_output], query)

            worker_outputs.append(critic_output)
            previous_cu = critic_output

        manager = ManagerAgent(self.coa.manager_model, self.coa.manager_prompt)
        final_output = manager.synthesize(worker_outputs, query)

        return final_output, worker_outputs
    # Key entity mark process
    """
    Mark the entity and conclusion in article before chunking
    """
    

    def mark_article_elements(self, article):
        doc = self.nlp(article)

        # Safer entity marking to avoid overlaps
        new_text = ""
        last_idx = 0
        for ent in doc.ents:
            start_char, end_char = ent.start_char, ent.end_char
            new_text += article[last_idx:start_char]
            new_text += f"<entity>{article[start_char:end_char]}</entity>"
            last_idx = end_char
        new_text += article[last_idx:]

        # Summarize — LED can handle long input
        summary = self.summarizer(
            article,
            max_length=200,
            min_length=50,
            do_sample=False
        )[0]['summary_text']

        new_text += f"\n<conclusion>{summary}</conclusion>"

        return new_text


    def ring_process(self, input_text, query):
        """
        Ring process: Workers form a ring and loop for 2 rounds ➜ Manager
        """
        chunks = split_into_chunks(input_text, self.coa.chunk_size, self.coa.worker_model)
        num_chunks = len(chunks)
        worker_outputs = []

        # Initialize current unit for each chunk
        current_units = [None] * num_chunks

        # Loop for 2 rounds
        for round_num in range(2):
            next_units = []

            for i, chunk in enumerate(chunks):
                worker = WorkerAgent(self.coa.worker_model, self.coa.worker_prompt)

                # Each worker receives:
                # - Its own chunk
                # - The query
                # - The previous output in the ring (previous chunk's output)
                prev_idx = (i - 1) % num_chunks
                previous_cu = current_units[prev_idx]

                output = worker.process_chunk(chunk, query, previous_cu)
                next_units.append(output)

            current_units = next_units

        # After 2 rounds, send all final outputs to the manager
        manager = ManagerAgent(self.coa.manager_model, self.coa.manager_prompt)
        final_output = manager.synthesize(current_units, query)

        return final_output, current_units

    def debug_process(self, input_text, query):
        """
        Example: same as base_process but prints extra logs
        """
        chunks = split_into_chunks(input_text, self.coa.chunk_size, self.coa.worker_model)
        worker_outputs = []
        previous_cu = None

        print(f"Debug: total chunks = {len(chunks)}")

        for idx, chunk in enumerate(chunks):
            print(f"Processing chunk {idx + 1}/{len(chunks)}...")
            worker = WorkerAgent(self.coa.worker_model, self.coa.worker_prompt)
            output = worker.process_chunk(chunk, query, previous_cu)
            print(f"Worker output:\n{output}\n")
            worker_outputs.append(output)
            previous_cu = output

        manager = ManagerAgent(self.coa.manager_model, self.coa.manager_prompt)
        final_output = manager.synthesize(worker_outputs, query)

        print(f"Manager final:\n{final_output}\n")

        return final_output, worker_outputs

    # Parallel then Verified by critic agent
    def tree_process_with_critic(self, input_text, query):
        """
        Swarm + Binary Tree Reduce with CriticAgent:
        Parallel workers ➜ pairwise critique/merge ➜ tree ➜ Manager
        """
        chunks = split_into_chunks(input_text, self.coa.chunk_size, self.coa.worker_model)

        def run_worker(chunk):
            worker = WorkerAgent(self.coa.worker_model, self.coa.worker_prompt)
            return worker.process_chunk(chunk, query)

        # Run workers in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_worker, chunk) for chunk in chunks]
            worker_outputs = [f.result() for f in futures]

        # Tree reduce with CriticAgent
        verified_outputs = self.tree_reduce_with_critic(worker_outputs, query)

        # Final manager synthesis
        manager = ManagerAgent(self.coa.manager_model, self.coa.manager_prompt)
        final_output = manager.synthesize(verified_outputs, query)

        return final_output, worker_outputs


    def tree_reduce_with_critic(self, outputs, query):
        """
        Pairwise critique/merge using CriticAgent in binary tree fashion.
        """
        current = outputs
        while len(current) > 1:
            next_level = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    # Merge the pair using CriticAgent
                    merged = self.merge_outputs_with_critic(current[i], current[i+1], query)
                    next_level.append(merged)
                else:
                    # Odd leftover, push up unchanged
                    next_level.append(current[i])
            current = next_level
        return current  # Should be one final verified output


    def merge_outputs_with_critic(self, output1, output2, query):
        """
        Uses CriticAgent to critique and merge two outputs.
        """
        critic = CriticAgent(self.coa.critic_model, self.coa.critic_prompt)
        # Critique takes a list of outputs plus the query for context
        merged_output = critic.critique([output1, output2], query)
        return merged_output

# improved with critic
    def improved_with_critic(self, input_text, query, rounds=1):
        """
        Worker → Critic → Worker → Critic → ... → Manager
        With enforced:
        - Restatement
        - Alternative check
        - Evidence check
        """

        chunks = split_into_chunks(input_text, self.coa.chunk_size, self.coa.worker_model)
        num_chunks = len(chunks)
        num_workers = num_chunks

        workers = [
            WorkerAgent(self.coa.worker_model, self.coa.worker_prompt)
            for _ in range(num_workers)
        ]
        critic = CriticAgent(self.coa.manager_model, self.coa.critic_prompt)

        all_outputs = []
        previous_cu = None

        for idx, chunk in enumerate(chunks):
            output = None
            trace = []

            total_steps = rounds * num_workers

            worker_idx = idx  # each chunk starts with its assigned worker

            for step in range(total_steps):
                worker = workers[worker_idx % num_workers]

                if step == 0:
                    input_text_for_worker = f"""
    Chunk: {chunk}
    Query: {query}
    Previous CU: {previous_cu or 'None'}
    """
                    output = worker.process_chunk(input_text_for_worker, query)
                    trace.append(f"Worker {worker_idx+1} Step {step+1}: {output}")

                else:
                    # Critic checks
                    critique = critic.critique(
                        chunk=chunk,
                        query=query,
                        worker_outputs=output
                    )
                    trace.append(f"Critic Step {step}: {critique}")

                    # Next worker responds to critique
                    input_text_for_worker = f"""
    Chunk: {chunk}
    Query: {query}
    Previous CU: {previous_cu or 'None'}
    Prior Worker Output: {output}
    Critic Feedback: {critique}
    """
                    output = worker.process_chunk(input_text_for_worker, query)
                    trace.append(f"Worker {worker_idx+1} Step {step+1}: {output}")

                worker_idx += 1

            previous_cu = output
            all_outputs.append("\n".join(trace))

        manager = ManagerAgent(self.coa.manager_model, self.coa.manager_prompt)
        final_output = manager.synthesize(all_outputs, query)

        return final_output, all_outputs
