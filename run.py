import time
from fairtestllm import FairAgent

start_time = time.time()

agent = FairAgent(
    # Required
    dataset_path      = "bank-full-biased.csv",
    target_col        = "y",
    api_key           = "sk-your-openai-key-here",  # ← OpenAI key

    # LLM
    llm_model         = "gpt-4o-mini",              # ← OpenAI model

    # Model
    model_type        = "mlp",

    # Protected attributes
    protected_attrs   = ["age", "marital"],

    # Search settings
    n_seeds           = 30,
    n_counterfactuals = 8,
    max_retries       = 0,
    n_iterations      = 3,

    fairness_threshold= -1.0,   # always search

    output_dir        = "fairagent_results"
)

results = agent.run()

end_time   = time.time()
total_time = end_time - start_time

print(f"\nDone! Found {results.good} discriminatory pairs")
print(f"G={results.good} F={results.failed} U={results.useless}")
print(f"Cost       : ${results.total_cost:.3f}")
print(f"Total time : {total_time:.1f}s ({total_time/60:.1f} min)")