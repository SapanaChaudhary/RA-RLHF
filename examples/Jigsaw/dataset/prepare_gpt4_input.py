import pandas as pd

# Load the generations

jigsaw_generations = pd.read_csv("jigsaw_generations_8_32_alpha_20_seed_2.csv")
gpt4_input = jigsaw_generations[["query", "response", "response_risk"]].copy()
gpt4_input.rename(columns={"query": "Prompt", "response": "Agent A", "response_risk": "Agent B"}, inplace=True)
gpt4_input.to_csv("input_to_gpt4.csv", index=False)
