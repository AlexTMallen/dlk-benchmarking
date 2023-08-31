from transformers import AutoTokenizer
import numpy as np
import openai
import pandas as pd
from tqdm import tqdm
from itertools import islice
import threading
import queue
import datetime
from typing import Literal
import re
import time


system_prompt = \
"""You are a widely knowledgeable, epistemically modest, and wise reviewer who is extraordinarily vigilant about getting facts right. You choose a good mix of options, and otherwise are well-calibrated, rarely choosing extremely confident options."""
prompt_template = \
"""### TRANSCRIPT OF UNKNOWN FACTUAL ACCURACY ###
{}
###

First, decide whether each statement can even be given an accuracy label, writing "Truth-apt" if it can, and "N/A" if the statement doesn't make any claims that can be judged on their accuracy. Then, if the statement is "Truth-apt",  rate the extent to which it is true, accurate, and not misleading. Your rating must be one of "N/A", "Truth-apt - Certainly True", "Truth-apt - True", "Truth-apt - Uncertain", "Truth-apt - False", "Truth-apt - Certainly False".

Fill in the following list with your rating for each of the statements marked, without any explanations or anything other than the ratings.

"""
score_list_template = "[[{}]] Score:"
API_costs = {
    "gpt-3.5-turbo": {"prompt_tokens": 0.0015 / 1000, "completion_tokens": 0.002 / 1000},
    "gpt-4": {"prompt_tokens": 0.03 / 1000, "completion_tokens": 0.06 / 1000},
}

class TruthLabeler:

    def __init__(self, 
                model_name="gpt-3.5-turbo",
                temperature=1,
                n_samples=5,
                uncertainty_bias=0,
                na_bias=0,
                score_to_p_apt=None,
                score_to_p_true=None,
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model_name = model_name
        self.temperature = temperature
        self.n_samples = n_samples
        self.uncertainty_bias = uncertainty_bias
        self.na_bias = na_bias
        self.score_to_p_apt = score_to_p_apt or {
                "N/A": 0,
                "Truth-apt - Certainly True": 1,
                "Truth-apt - True": 1,
                "Truth-apt - Uncertain": 0.3,
                "Truth-apt - False": 1,
                "Truth-apt - Certainly False": 1,
            }
        assert all(score in prompt_template for score in score_to_p_apt.keys())
        self.score_to_p_true = score_to_p_true or {
                "N/A": 0.5,
                "Truth-apt - Certainly True": 1,
                "Truth-apt - True": 0.8,
                "Truth-apt - Uncertain": 0.5,
                "Truth-apt - False": 0.2,
                "Truth-apt - Certainly False": 0,
            }
        assert all(score in prompt_template for score in score_to_p_apt.keys())
        self.logit_bias = {self.tokenizer.encode(" Unc")[0]: uncertainty_bias, self.tokenizer.encode(" N")[0]: na_bias}
        self.stop_seq = "\n\n"
        self.total_cost = 0

    def aggregate_sample_scores(self, score_set, result: Literal["p_apt", "p_true"]):
        score_to_p = self.score_to_p_apt if result == "p_apt" else self.score_to_p_true
        p = np.mean([score_to_p[score] for score in score_set])
        return p

    @staticmethod
    def make_input(annotated_transcript):
        pattern = re.compile(r"\[\[(\d+)\]\]")  # find all annotations
        ann_count = len(pattern.findall(annotated_transcript))
            
        input = prompt_template.format(annotated_transcript)
        score_list = "\n".join(score_list_template.format(i) for i in range(1, ann_count + 1))
        input += score_list
        return input, ann_count
    
    def label_example(self, i, id, annotated_transcript, results, num_tries=5):
        try:
            
            input, ann_count = TruthLabeler.make_input(annotated_transcript)
            if ann_count == 0:
                print("SKIPPING: no truth-apt statements")
                return
            
            for i in range(num_tries):
                try:
                    if i > 0:
                        print("Retrying request")
                    completion = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": input},
                        ],
                        temperature=self.temperature,
                        max_tokens=len(self.tokenizer.encode(score_list)) * 3,
                        logit_bias=self.logit_bias,
                        stop=self.stop_seq,
                        n=self.n_samples,
                    )
                    break
                except Exception as e:
                    print("Error completing request:", e)
                    time.sleep(2)
            
            usage = completion["usage"]
            prompt_tokens, completion_tokens = usage["prompt_tokens"], usage["completion_tokens"]
            cost = API_costs[self.model_name]["prompt_tokens"] * prompt_tokens + API_costs[self.model_name]["completion_tokens"] * completion_tokens
            self.total_cost += cost
            
            # check that there's the rightn number of choices
            if len(completion["choices"]) != self.n_samples:
                print("SKIPPING: multiple choices")
                return
            
            score_samples = []
            responses = []
            for choice in completion["choices"]:
                # check that finish reason is not for a content filter, not for length, not for function_call and that it is "stop"
                if choice["finish_reason"] != "stop":
                    print(f"SKIPPING: finish reason is {completion['choices'][0]['finish_reason']}, not stop")
                    print("RESPONSE:", choice["message"]["content"])
                    return

                response = choice["message"]["content"]
                if response.endswith(self.stop_seq):
                    print(f"Removing stop sequence from response: {self.stop_seq}")
                    response = response[:-len(self.stop_seq)]

                response = response.strip()

                pred_scores = self.get_scores_from_response(response, ann_count)
                if pred_scores is None:
                    continue

                score_samples.append(pred_scores)
                responses.append(response)

            if len(score_samples) == 0:
                print("SKIPPING: no valid samples")
                return

            # transpose the list of lists, so that each list contains the scores for a single annotation
            score_samples = list(zip(*score_samples))
            p_apts = [self.aggregate_sample_scores(scores, "p_apt") for scores in score_samples]
            p_trues = [self.aggregate_sample_scores(scores, "p_true") for scores in score_samples]
            
            result = {
                "idx": i,
                "message_id": id,
                "input": input,
                "annotated_transcript": annotated_transcript,
                "responses": responses,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "dollars": cost,
                "pred_scores": score_samples,
                "p_apts": p_apts,
                "p_trues": p_trues,
                "ann_count": ann_count,
            }
            results.put(result)
        except Exception as e:
            print("SKIPPING DUE TO MAIN ERROR", e)
            return
        
    def label(self, ids, annotated_transcripts, n_threads=10):
        """
        ids: list of unique ids for each text
        annotated_transcripts: list of strings containing annotated transcripts:
                e.g. 
                "USER: A penguin is a bird. Is this correct?
                 
                 ASSISTANT: This is correct[[1]], penguins are birds[[2]]."
        """
        results = queue.Queue()
        n_iters = (len(ids) // n_threads) * n_threads
        iterator = islice(enumerate(zip(ids, annotated_transcripts)), n_iters)

        while True:
            threads = []
            for _ in range(n_threads):
                i, args = next(iterator)
                t = threading.Thread(target=self.label_example, args=(i, *args, results))
                threads.append(t)
                t.start()

            for t in threads:
                t.join(timeout=60)
            
            for t in threads:
                if t.is_alive():
                    print("THREAD TIMED OUT")
                    try:
                        t._stop()
                    except AssertionError:
                        print("Thread could not be terminated")
                    
            print(f"Total cost: ${self.total_cost:.4f}")
            if (i + 1) % 200 == 0 or i == n_iters - 1:
                out_df = pd.DataFrame(list(results.queue))
                out_df = out_df.sort_values("idx")
                out_df.to_json(f"data/checkpoints/{self.model_name}_{i + 1}.json")
            if i == n_iters - 1:
                break
            
        out_df = pd.DataFrame(list(results.queue))
        out_df = out_df.sort_values("idx")
        return out_df

    @staticmethod   
    def get_scores_from_response(response, ann_count):
        target = "]] Score:" 
        options = ["N/A", "Truth-apt - Certainly True", "Truth-apt - True", "Truth-apt - Uncertain", "Truth-apt - False", "Truth-apt - Certainly False"]
        assert target in score_list_template  # make sure I didn't change the template
        assert all(option in prompt_template for option in options)  # make sure I didn't change the template

        if score_list_template.format(1) in response:        
            scores = []
            idx = 0
            while idx := (response.index(target, idx) if target in response[idx:] else None):
                idx += len(target)
                try:
                    newline_idx = response.index("\n", idx)
                except ValueError:
                    newline_idx = len(response)
                score = response[idx:newline_idx].strip()
                scores.append(score)
        else:
            scores = response.split("\n")
            scores = [score.strip() for score in scores if score.strip()]

        if any(score not in options for score in scores):
            print(f"SKIPPING: scores must be one of {options}, but found {scores}")
            return
        if len(scores) != ann_count:
            print(f"SKIPPING: {len(scores)} scores found, but {ann_count} annotations were expected.")
            return

        return scores